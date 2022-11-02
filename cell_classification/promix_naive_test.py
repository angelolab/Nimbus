from segmentation_data_prep_test import prep_object_and_inputs
from promix_naive import PromixNaive
import toml
import tempfile
import numpy as np
import tensorflow as tf
import os


def test_reduce_to_cells():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    pred = np.random.rand(16, 256, 266)
    instance_mask = np.random.randint(0, 100, (16, 256, 266))
    instance_mask[-1, instance_mask[-1] == 1] = 0
    marker_activity_mask = np.zeros_like(instance_mask)
    marker_activity_mask[instance_mask > 90] = 1
    trainer = PromixNaive(params)
    uniques, mean_per_cell = tf.map_fn(
        trainer.reduce_to_cells,
        (pred, instance_mask),
        infer_shape=False,
        fn_output_signature=[
            tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0),
            tf.RaggedTensorSpec(shape=[None], dtype=tf.float32, ragged_rank=0),
        ],
    )

    # check that the output has the right dimension
    assert uniques.shape[0] == instance_mask.shape[0]
    assert mean_per_cell.shape[0] == instance_mask.shape[0]

    # check that the output is correct
    assert set(np.unique(instance_mask[0])) == set(uniques[0].numpy())
    for i in np.unique(instance_mask[0]):
        assert np.isclose(
            np.mean(pred[0][instance_mask[0] == i]),
            mean_per_cell[0][uniques[0] == i].numpy().max(),
        )


def test_matched_high_confidence_selection_thresholds():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    trainer = PromixNaive(params)
    thresholds = trainer.matched_high_confidence_selection_thresholds()

    # check that the output has the right dimension
    assert len(thresholds) == 2
    assert thresholds["positive"] > 0.0
    assert thresholds["negative"] > 0.0


def test_train():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["num_steps"] = 7
        params["num_validation"] = 2
        params["batch_size"] = 2
        params["test"] = True
        params["weight_decay"] = 1e-4
        params["snap_steps"] = 5
        params["val_steps"] = 5
        params["quantile"] = 0.3
        params["ema"] = 0.01
        params["confidence_thresholds"] = [0.1, 0.9]
        trainer = PromixNaive(params)
        trainer.train()

        # check params.toml is dumped to file and contains the created paths
        assert "params.toml" in os.listdir(trainer.params["model_dir"])
        loaded_params = toml.load(os.path.join(trainer.params["model_dir"], "params.toml"))
        for key in ["model_dir", "log_dir", "model_path"]:
            assert key in list(loaded_params.keys())

        # check if model can be loaded from file
        trainer.model = None
        trainer.load_model(trainer.params["model_path"])
        assert isinstance(trainer.model, tf.keras.Model)


def test_prep_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["num_steps"] = 3
        params["num_validation"] = 2
        params["batch_size"] = 2
        trainer = PromixNaive(params)
        trainer.prep_data()

        # check if train and validation datasets exists and are of the right type
        assert isinstance(trainer.validation_dataset, tf.data.Dataset)
        assert isinstance(trainer.train_dataset, tf.data.Dataset)


def test_class_wise_loss_selection():
    pass


def test_matched_high_confidence_selection():
    pass


def test_batchwise_loss_selection():
    pass
