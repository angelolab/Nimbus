import os
import random
import numpy as np
from skimage import io
from joblib import Parallel, delayed
import json


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
        fov_paths, output_dir, quantile=0.999, exclude_channels=[], n_subset=10, n_jobs=8,
    ):
    """Prepares the normalization dict for a list of fovs
    Args:
        fov_paths (list): list of paths to fovs
        quantile (float): quantile to use for normalization
        exclude_channels (list): list of channels to exclude
        n_subset (int): number of fovs to use for normalization
    Returns:
        normalization_dict (dict): dict with fov names as keys and normalization values as values
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
    if not os.path.exists(os.path.join(output_dir, 'normalization_dict.json')):
        with open(os.path.join(output_dir, 'normalization_dict.json'), 'w') as f:
            json.dump(normalization_dict, f)    
    return normalization_dict