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


def test_check_inputs():
    with tempfile.TemporaryDirectory() as temp_dir:
        _, fov_paths, _, _ = prep_object_and_inputs(temp_dir)
        exclude_channels = ["CD57"]
        segmentation_naming_convention = lambda x: os.path.join(x, "cell_segmentation.tiff")
        output_dir = temp_dir
        save_predictions = True
        
        # check if no errors are raised when all inputs are valid
        nimbus = Nimbus(
            fov_paths, exclude_channels, segmentation_naming_convention, output_dir,
            save_predictions
        )
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


def test_initialize_model():
        nimbus = Nimbus(None, None, None, None, None)
        assert type(nimbus.model) == tf.keras.functional.Functional


def test_prepare_normalization_dict():
    # test if normalization dict gets prepared and saved, in-depth tests are in inference_test.py


def Nimbus_test(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = tf_record_path
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["num_steps"] = 20
        config_params["num_validation"] = 2
        config_params["batch_size"] = 2
        config_params["test"] = True
        config_params["snap_steps"] = 5000
        config_params["val_steps"] = 5000
        trainer = ModelBuilder(config_params)
        trainer.train()
        input_data = np.random.rand(1, 1024, 1024, 2)
        prediction = Nimbus(trainer.model).predict(
            input_data, marker="test", normalization_dict={"test": 1.0}
        )
        # check if shape and values are in the expected range
        assert prediction.shape == (1, 1024, 1024, 1)
        assert prediction.max() <= 1.0
        assert prediction.min() >= 0.0
