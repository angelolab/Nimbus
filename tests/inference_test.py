import os
import json
import pytest
import tempfile
import numpy as np
from skimage import io
from cell_classification.application import Nimbus
from segmentation_data_prep_test import prepare_test_data_folders
from cell_classification.inference import calculate_normalization, prepare_normalization_dict
from cell_classification.inference import prepare_input_data, segment_mean, predict_fovs
from cell_classification.inference import test_time_aug as tt_aug


def test_calculate_normalization():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_paths = prepare_test_data_folders(
            1, temp_dir, ["CD4"], random=True,
            scale=[0.5]
        )
        channel = "CD4"
        channel_path = os.path.join(fov_paths[0], channel + ".tiff")
        channel_out, norm_val = calculate_normalization(channel_path, 0.999)
        # test if we get the correct channel and normalization value
        assert channel_out == channel
        assert np.isclose(norm_val, 0.5, 0.01)


def test_prepare_normalization_dict():
    with tempfile.TemporaryDirectory() as temp_dir:
        scales = [0.5, 1.0, 1.5, 2.0, 5.0]
        channels = ["CD4", "CD11c", "CD14", "CD56", "CD57"]
        fov_paths = prepare_test_data_folders(
            5, temp_dir, channels, random=True,
            scale=scales
        )
        normalization_dict = prepare_normalization_dict(
            fov_paths, temp_dir, quantile=0.999, exclude_channels=["CD57"], n_subset=10, n_jobs=1,
            output_name="normalization_dict.json"
        )
        # test if normalization dict got saved
        assert os.path.exists(os.path.join(temp_dir, "normalization_dict.json"))
        assert normalization_dict == json.load(
            open(os.path.join(temp_dir, "normalization_dict.json"))
        )
        # test if channel got excluded
        assert "CD57" not in normalization_dict.keys()
        # test if normalization dict is correct
        for channel, scale in zip(channels, scales):
            if channel == "CD57":
                continue
            assert np.isclose(normalization_dict[channel], scale, 0.01)

        # test if multiprocessing yields approximately the same results
        normalization_dict_mp = prepare_normalization_dict(
            fov_paths, temp_dir, quantile=0.999, exclude_channels=["CD57"], n_subset=10, n_jobs=2,
            output_name="normalization_dict.json"
        )
        for key in normalization_dict.keys():
            assert np.isclose(normalization_dict[key], normalization_dict_mp[key], 1e-6)


def test_prepare_input_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        scales = [0.5]
        channels = ["CD4"]
        fov_paths = prepare_test_data_folders(
            1, temp_dir, channels, random=True,
            scale=scales
        )
        mplex_img = io.imread(os.path.join(fov_paths[0], "CD4.tiff"))
        instance_mask = io.imread(os.path.join(fov_paths[0], "cell_segmentation.tiff"))
        input_data = prepare_input_data(mplex_img, instance_mask)
        # check shape
        assert input_data.shape == (1, 256, 256, 2)
        # check if instance mask got binarized and eroded
        assert np.alltrue(np.unique(input_data[..., 1]) == np.array([0, 1]))
        assert np.sum(input_data[..., 1]) < np.sum(instance_mask)
        # check if mplex image is the same as before
        assert np.alltrue(input_data[0, ..., 0] == mplex_img)


def test_segment_mean():
    with tempfile.TemporaryDirectory() as temp_dir:
        scales = [0.5]
        channels = ["CD4"]
        fov_paths = prepare_test_data_folders(
            1, temp_dir, channels, random=True,
            scale=scales
        )
        mplex_img = io.imread(os.path.join(fov_paths[0], "CD4.tiff"))
        prediction = (mplex_img > 0.5).astype(np.float32)
        instance_mask = io.imread(os.path.join(fov_paths[0], "cell_segmentation.tiff"))


def test_tt_aug():
    pass


def test_predict_fovs():
    pass
