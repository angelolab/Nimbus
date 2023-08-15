import os
import pytest
import tempfile
import numpy as np
import tensorflow as tf
from cell_classification.application import Nimbus, nimbus_preprocess
from cell_classification.model_builder import ModelBuilder
from segmentation_data_prep_test import prep_object_and_inputs


def test_cell_preprocess():
    input_data = np.random.rand(1, 1024, 1024, 2)
    expected_output = np.copy(input_data)
    expected_output[..., 0] = input_data[..., 0] / 1.2

    output = nimbus_preprocess(
        input_data, normalize=True, marker="test", normalization_dict={"test": 1.2}
    )
    # check if shape and values are in the expected range
    assert output.shape == (1, 1024, 1024, 2)
    assert output.max() <= 1.0
    assert output.min() >= 0.0
    # check if normalization was applied
    assert np.allclose(expected_output, output, atol=1e-5)

    # check if normalization works when not dict is given
    output = nimbus_preprocess(input_data, normalize=True, marker="test")
    expected_output[..., 0] = input_data[..., 0] / np.quantile(input_data[..., 0], 0.999)
    expected_output = expected_output.clip(0, 1)

    assert np.allclose(expected_output, output, atol=1e-5)

    # check if normalization works when marker is not in dict
    output = nimbus_preprocess(
        input_data, normalize=True, marker="test2", normalization_dict={"test": 1.2}
    )
    assert np.allclose(expected_output, output, atol=1e-5)

    # check if normalization works when normalization is set to False
    output = nimbus_preprocess(input_data, normalize=False)
    assert np.array_equal(input_data, output)


def test_initialize_model():
        nimbus = Nimbus(None, None, None)
        assert type(nimbus.model) == tf.keras.functional.Functional


def test_check_inputs():
    with tempfile.TemporaryDirectory() as temp_dir:
        _, fov_paths, _, _ = prep_object_and_inputs(temp_dir)
        segmentation_naming_convention = lambda x: os.path.join(x, "cell_segmentation.tiff")
        output_dir = temp_dir
        
        # check if no errors are raised when all inputs are valid
        nimbus = Nimbus(fov_paths, segmentation_naming_convention, output_dir)
        nimbus.check_inputs()

        # check if error is raised when a path in fov_paths does not exist on disk
        nimbus.fov_paths.append("invalid_path")
        with pytest.raises(FileNotFoundError, match="invalid_path"):
            nimbus.check_inputs()
        
        # check if error is raised when segmentation_name_convention does not return a valid path
        nimbus.fov_paths = fov_paths
        nimbus.segmentation_naming_convention = lambda x: "invalid_path"
        with pytest.raises(FileNotFoundError, match="invalid_path"):
            nimbus.check_inputs()

        # check if error is raised when output_dir does not exist on disk
        nimbus.segmentation_naming_convention = segmentation_naming_convention
        nimbus.output_dir = "invalid_path"
        with pytest.raises(FileNotFoundError, match="invalid_path"):
            nimbus.check_inputs()


def test_prepare_normalization_dict():
    # test if normalization dict gets prepared and saved, in-depth tests are in inference_test.py
    with tempfile.TemporaryDirectory() as temp_dir:
        _, fov_paths, _, _ = prep_object_and_inputs(temp_dir)
        segmentation_naming_convention = lambda x: os.path.join(x, "cell_segmentation.tiff")
        output_dir = temp_dir
        nimbus = Nimbus(
            fov_paths, segmentation_naming_convention, output_dir, exclude_channels = ["CD57"]
        )
        # test if normalization dict gets prepared and saved
        nimbus.prepare_normalization_dict(overwrite=True)
        assert os.path.exists(os.path.join(output_dir, "normalization_dict.json"))
        assert "CD57" not in nimbus.normalization_dict.keys()

        # test if normalization dict gets loaded
        nimbus_2 = Nimbus(
            fov_paths, segmentation_naming_convention, output_dir, exclude_channels = ["CD57"]
        )
        nimbus_2.prepare_normalization_dict()
        assert nimbus_2.normalization_dict == nimbus.normalization_dict


def test_predict_fovs():
    with tempfile.TemporaryDirectory() as temp_dir:
        _, fov_paths, _, _ = prep_object_and_inputs(temp_dir)
        segmentation_naming_convention = lambda x: os.path.join(x, "cell_segmentation.tiff")
        os.remove(os.path.join(temp_dir, 'normalization_dict.json'))
        output_dir = os.path.join(temp_dir, "nimbus_output")
        fov_paths = fov_paths[:1]
        nimbus = Nimbus(
            fov_paths, segmentation_naming_convention, output_dir,
            exclude_channels=["CD57", "CD11c", "XYZ"]
        )
        cell_table = nimbus.predict_fovs()

        # check if all channels are in the cell_table
        for channel in nimbus.normalization_dict.keys():
            assert channel+"_pred" in cell_table.columns
        # check if cell_table was saved
        assert os.path.exists(os.path.join(output_dir, "nimbus_cell_table.csv"))

        # check if fov folders were created in output_dir
        for fov_path in fov_paths:
            fov = os.path.basename(fov_path)
            assert os.path.exists(os.path.join(output_dir, fov))
            # check if all predictions were saved
            for channel in nimbus.normalization_dict.keys():
                assert os.path.exists(os.path.join(output_dir, fov, channel+".tiff"))
            # check if CD57 was excluded
            assert not os.path.exists(os.path.join(output_dir, fov, "CD57.tiff"))
            assert not os.path.exists(os.path.join(output_dir, fov, "CD11c.tiff"))
            assert not os.path.exists(os.path.join(output_dir, fov, "XYZ.tiff"))
