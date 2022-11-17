from application import CellClassification
from model_builder import ModelBuilder
from segmentation_data_prep_test import prep_object_and_inputs
import numpy as np
import tempfile
import toml
import os


def predict_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["num_steps"] = 20
        params["num_validation"] = 2
        params["batch_size"] = 2
        params["test"] = True
        params["snap_steps"] = 5000
        params["val_steps"] = 5000
        trainer = ModelBuilder(params)
        trainer.train()
        input_data = np.random.rand(1, 1024, 1024, 2)
        prediction = CellClassification(trainer.model).predict(
            input_data, marker="test", normalization_dict={"test": 1.0}
        )
        # check if shape and values are in the expected range
        assert prediction.shape == (1, 1024, 1024, 1)
        assert prediction.max() <= 1.0
        assert prediction.min() >= 0.0
