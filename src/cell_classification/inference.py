import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries


def calculate_normalization(channel_path, quantile):
    """Calculates the normalization value for a given channel
    Args:
        channel_path (str): path to channel
        quantile (float): quantile to use for normalization
    Returns:
        normalization_value (float): normalization value
    """
    mplex_img = io.imread(channel_path)
    normalization_value = np.quantile(mplex_img, quantile)
    chan = os.path.basename(channel_path).split(".")[0]
    return chan, normalization_value


def prepare_normalization_dict(
        fov_paths, output_dir, quantile=0.999, exclude_channels=[], n_subset=10, n_jobs=1,
        output_name="normalization_dict.json"
    ):
    """Prepares the normalization dict for a list of fovs
    Args:
        fov_paths (list): list of paths to fovs
        output_dir (str): path to output directory
        quantile (float): quantile to use for normalization
        exclude_channels (list): list of channels to exclude
        n_subset (int): number of fovs to use for normalization
        n_jobs (int): number of jobs to use for joblib multiprocessing
        output_name (str): name of output file
    Returns:
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
    """
    normalization_dict = {}
    if n_subset is not None:
        random.shuffle(fov_paths)
        fov_paths = fov_paths[:n_subset]
    print("Iterate over fovs...")
    for fov_path in tqdm(fov_paths):
        channels = os.listdir(fov_path)
        channels = [
            channel for channel in channels if channel.split(".")[0] not in exclude_channels
        ]
        channel_paths = [os.path.join(fov_path, channel) for channel in channels]
        if n_jobs > 1:
            normalization_values = Parallel(n_jobs=n_jobs)(
            delayed(calculate_normalization)(channel_path, quantile)
            for channel_path in channel_paths
            )
        else:
            normalization_values = [
                calculate_normalization(channel_path, quantile)
                for channel_path in channel_paths
            ]
        for channel, normalization_value in normalization_values:
            if channel not in normalization_dict:
                normalization_dict[channel] = []
            normalization_dict[channel].append(normalization_value)
    for channel in normalization_dict.keys():
        normalization_dict[channel] = np.mean(normalization_dict[channel])
    # save normalization dict
    with open(os.path.join(output_dir, output_name), 'w') as f:
        json.dump(normalization_dict, f)
    return normalization_dict


def prepare_input_data(mplex_img, instance_mask):
    """Prepares the input data for the segmentation model
    Args:
        mplex_img (np.array): multiplex image
        instance_mask (np.array): instance mask
    Returns:
        input_data (np.array): input data for segmentation model
    """
    edge = find_boundaries(instance_mask, mode="inner").astype(np.uint8)
    binary_mask = np.logical_and(edge == 0, instance_mask > 0).astype(np.float32)
    input_data = np.stack([mplex_img, binary_mask], axis=-1)[np.newaxis,...] # bhwc
    return input_data


def segment_mean(instance_mask, prediction):
    """Calculates the mean prediction per instance
    Args:
        instance_mask (np.array): instance mask
        prediction (np.array): prediction
    Returns:
        uniques (np.array): unique instance ids
        mean_per_cell (np.array): mean prediction per instance
    """
    instance_mask_flat = tf.cast(tf.reshape(instance_mask, -1), tf.int32)  # (h*w)
    pred_flat = tf.cast(tf.reshape(prediction, -1), tf.float32)
    sort_order = tf.argsort(instance_mask_flat)
    instance_mask_flat = tf.gather(instance_mask_flat, sort_order)
    uniques, _ = tf.unique(instance_mask_flat)
    pred_flat = tf.gather(pred_flat, sort_order)
    mean_per_cell = tf.math.segment_mean(pred_flat, instance_mask_flat)
    mean_per_cell = tf.gather(mean_per_cell, uniques)
    return [uniques.numpy()[1:], mean_per_cell.numpy()[1:]] # discard background


def test_time_aug(
        input_data, channel, app, normalization_dict, rotate=True, flip=True, batch_size=4
    ):
    """Performs test time augmentation
    Args:
        input_data (np.array): input data for segmentation model, mplex_img and binary mask
        channel (str): channel name
        app (tf.keras.Model): segmentation model
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
        rotate (bool): whether to rotate
        flip (bool): whether to flip
        batch_size (int): batch size
    Returns:
        seg_map (np.array): predicted segmentation map
    """
    forward_augmentations = []
    backward_augmentations = []
    if rotate:
        for k in [0,1,2,3]:
            forward_augmentations.append(lambda x: tf.image.rot90(x, k=k))
            backward_augmentations.append(lambda x: tf.image.rot90(x, k=-k))
    if flip:
        forward_augmentations += [
            lambda x: tf.image.flip_left_right(x),
            lambda x: tf.image.flip_up_down(x)
        ]
        backward_augmentations += [
            lambda x: tf.image.flip_left_right(x),
            lambda x: tf.image.flip_up_down(x)
        ]
    input_batch = []
    for forw_aug in forward_augmentations:
        input_data_tmp = forw_aug(input_data).numpy() # bhwc
        input_batch.append(np.concatenate(input_data_tmp))
    input_batch = np.stack(input_batch, 0)
    seg_map = app._predict_segmentation(
        input_batch,
        batch_size=batch_size,
        preprocess_kwargs={
            "normalize": True,
            "marker": channel,
            "normalization_dict": normalization_dict},
        )
    tmp = []
    for backw_aug, seg_map_tmp in zip(backward_augmentations, seg_map):
        seg_map_tmp = backw_aug(seg_map_tmp[np.newaxis,...])
        seg_map_tmp = np.squeeze(seg_map_tmp)
        tmp.append(seg_map_tmp)
    seg_map = np.stack(tmp, -1)
    seg_map = np.mean(seg_map, axis = -1, keepdims = True)
    return seg_map


def predict_fovs(
        fov_paths, cell_classification_output_dir, app, normalization_dict,
        segmentation_naming_convention, exclude_channels=[], save_predictions=True,
        half_resolution=False, batch_size=4, test_time_augmentation=True
    ):
    """Predicts the segmentation map for each mplex image in each fov
    Args:
        fov_paths (list): list of fov paths
        cell_classification_output_dir (str): path to cell classification output dir
        app (deepcell.applications.Application): segmentation model
        normalization_dict (dict): dict with channel names as keys and norm factors  as values
        segmentation_naming_convention (function): function to get instance mask path from fov path
        exclude_channels (list): list of channels to exclude
        save_predictions (bool): whether to save predictions
        half_resolution (bool): whether to use half resolution
        batch_size (int): batch size
        test_time_augmentation (bool): whether to use test time augmentation
    Returns:
        cell_table (pd.DataFrame): cell table with predicted confidence scores per fov and cell
    """
    fov_dict_list = []
    for fov_path in tqdm(fov_paths):
        out_fov_path = os.path.join(
            os.path.normpath(cell_classification_output_dir), os.path.basename(fov_path)
        )
        fov_dict = {}
        for channel in os.listdir(fov_path):
            channel_path = os.path.join(fov_path, channel)
            if not channel.endswith(".tiff"):
                continue
            if channel[:2] == "._":
                continue
            channel = channel.split(".")[0]
            if channel in exclude_channels:
                continue
            mplex_img = np.squeeze(io.imread(channel_path))
            instance_path = segmentation_naming_convention(fov_path)
            instance_mask = np.squeeze(io.imread(instance_path))
            input_data = prepare_input_data(mplex_img, instance_mask)
            if half_resolution:
                scale = 0.5
                input_data = np.squeeze(input_data)
                h,w,_ = input_data.shape
                img = cv2.resize(input_data[...,0], [int(h*scale), int(w*scale)])
                binary_mask = cv2.resize(
                    input_data[...,1], [int(h*scale), int(w*scale)], interpolation=0
                )
                input_data = np.stack([img, binary_mask], axis=-1)[np.newaxis,...]
            if test_time_augmentation:
                prediction = test_time_aug(
                    input_data, channel, app, normalization_dict, batch_size=batch_size
                )
            else:
                prediction = app._predict_segmentation(
                    input_data,
                    preprocess_kwargs={
                        "normalize": True, "marker": channel,
                        "normalization_dict": normalization_dict
                    },
                    batch_size=batch_size
                )
            prediction = np.squeeze(prediction)
            if half_resolution:
                prediction = cv2.resize(prediction, (h, w))
            instance_mask = np.expand_dims(instance_mask, axis=-1)
            labels, mean_per_cell = segment_mean(instance_mask, prediction)
            if "label" not in fov_dict.keys():
                fov_dict["fov"] = [os.path.basename(fov_path)]*len(labels)
                fov_dict["label"] = labels
            fov_dict[channel+"_pred"] = mean_per_cell
            if save_predictions:
                os.makedirs(out_fov_path, exist_ok=True)
                pred_int = tf.cast(prediction*255.0, tf.uint8).numpy()
                io.imsave(
                    os.path.join(out_fov_path, channel+".tiff"), pred_int,
                    photometric="minisblack", compression="zlib"
                )
        fov_dict_list.append(pd.DataFrame(fov_dict))
    cell_table = pd.concat(fov_dict_list, ignore_index=True)
    return cell_table
