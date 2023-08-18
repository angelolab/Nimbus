import os
import json
import pytest
import tempfile
import numpy as np
from skimage import io
from cell_classification.application import Nimbus
from segmentation_data_prep_test import prepare_test_data_folders, prep_object_and_inputs
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
        prediction = (mplex_img > 0.25).astype(np.float32)
        instance_mask = io.imread(os.path.join(fov_paths[0], "cell_segmentation.tiff"))
        instance_ids, mean_per_cell = segment_mean(instance_mask, prediction)
        # check if we get the correct number of cells
        assert len(instance_ids) == len(np.unique(instance_mask)[1:])
        # check if we get the correct mean per cell
        for i in np.unique(instance_mask)[1:]:
            assert mean_per_cell[i-1] == np.mean(prediction[instance_mask == i])


def test_tt_aug():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            return os.path.join(fov_path, "cell_segmentation.tiff")

        _, fov_paths, _, _ = prep_object_and_inputs(temp_dir)
        os.remove(os.path.join(temp_dir, 'normalization_dict.json'))
        output_dir = os.path.join(temp_dir, "nimbus_output")
        fov_paths = fov_paths[:1]
        nimbus = Nimbus(
            fov_paths, segmentation_naming_convention, output_dir,
            exclude_channels=["CD57", "CD11c", "XYZ"]
        )
        nimbus.prepare_normalization_dict()
        channel = "CD4"
        mplex_img = io.imread(os.path.join(fov_paths[0], channel+".tiff"))
        instance_mask = io.imread(os.path.join(fov_paths[0], "cell_segmentation.tiff"))
        input_data = prepare_input_data(mplex_img, instance_mask)
        pred_map = tt_aug(
            input_data, channel, nimbus, nimbus.normalization_dict, rotate=True, flip=True,
            batch_size=32
        )
        # check if we get the correct shape
        assert pred_map.shape == (256, 256, 1)

        pred_map_2 = tt_aug(
            input_data, channel, nimbus, nimbus.normalization_dict, rotate=False, flip=True,
            batch_size=32
        )
        pred_map_3 = tt_aug(
            input_data, channel, nimbus, nimbus.normalization_dict, rotate=True, flip=False,
            batch_size=32
        )
        pred_map_no_tt_aug = nimbus._predict_segmentation(
            input_data,
            batch_size=1,
            preprocess_kwargs={
                "normalize": True,
                "marker": channel,
                "normalization_dict": nimbus.normalization_dict},
        )
        # check if we get roughly the same results for non augmented and augmented predictions
        assert np.allclose(pred_map, pred_map_no_tt_aug, atol=0.05)
        assert np.allclose(pred_map_2, pred_map_no_tt_aug, atol=0.05)
        assert np.allclose(pred_map_3, pred_map_no_tt_aug, atol=0.05)


def test_predict_fovs():
    with tempfile.TemporaryDirectory() as temp_dir:
        def segmentation_naming_convention(fov_path):
            return os.path.join(fov_path, "cell_segmentation.tiff")

        exclude_channels = ["CD57", "CD11c", "XYZ"]
        _, fov_paths, _, _ = prep_object_and_inputs(temp_dir)
        os.remove(os.path.join(temp_dir, 'normalization_dict.json'))
        output_dir = os.path.join(temp_dir, "nimbus_output")
        fov_paths = fov_paths[:1]
        nimbus = Nimbus(
            fov_paths, segmentation_naming_convention, output_dir,
            exclude_channels=exclude_channels
        )
        output_dir = os.path.join(temp_dir, "nimbus_output")
        nimbus.prepare_normalization_dict()
        cell_table = predict_fovs(
            fov_paths, output_dir, nimbus, nimbus.normalization_dict,
            segmentation_naming_convention, exclude_channels=exclude_channels,
            save_predictions=False, half_resolution=True,
        )
        # check if we get the correct number of cells
        assert len(cell_table) == 15
        # check if we get the correct columns (fov, segmentation_label, CD4_pred, CD56_pred)
        assert np.alltrue(
            set(cell_table.columns) == set(["fov", "segmentation_label", "CD4_pred", "CD56_pred"])
        )
        # check if predictions don't get written to output_dir
        assert not os.path.exists(os.path.join(output_dir, "fov_0", "CD4.tiff"))
        assert not os.path.exists(os.path.join(output_dir, "fov_0", "CD56.tiff"))
        #
        # run again with save_predictions=True and check if predictions get written to output_dir
        cell_table = predict_fovs(
            fov_paths, output_dir, nimbus, nimbus.normalization_dict,
            segmentation_naming_convention, exclude_channels=exclude_channels,
            save_predictions=True, half_resolution=True,
        )
        assert os.path.exists(os.path.join(output_dir, "fov_0", "CD4.tiff"))
        assert os.path.exists(os.path.join(output_dir, "fov_0", "CD56.tiff"))
