from segmentation_data_prep_test import prep_object_and_inputs
from promix_naive import PromixNaive
import toml
import tempfile
import numpy as np
import tensorflow as tf


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
        params["num_epochs"] = 2
        params["num_validation"] = 2
        params["batch_size"] = 2
        params["test"] = True
        params["weight_decay"] = 1e-4
        params["steps_per_epoch"] = 5000
        trainer = PromixNaive(params)
        trainer.train()


def test_reduce_to_cells():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    params["quantile"] = 0.3
    pred = np.random.rand(16,256,266)
    instance_mask = np.random.randint(0, 100, (16,256,266))
    instance_mask[-1, instance_mask[-1]==1] = 0
    marker_activity_mask = np.zeros_like(instance_mask)
    marker_activity_mask[instance_mask > 90] = 1
    trainer = PromixNaive(params)
    uniques, mean_per_cell = tf.map_fn(trainer.reduce_to_cells, (pred, instance_mask),
        infer_shape=False, fn_output_signature=(
            tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0), tf.RaggedTensorSpec(shape=[None], dtype=tf.float32, ragged_rank=0)
        )
    )

    # check that the output has the right dimension
    assert uniques.shape[0] == instance_mask.shape[0]
    assert mean_per_cell.shape[0] == instance_mask.shape[0]

    # check that the output is correct
    assert set(np.unique(instance_mask[0])) == set(uniques[0].numpy())
    for i in np.unique(instance_mask[0]):
        assert np.isclose(
            np.mean(pred[0][instance_mask[0]==i]), mean_per_cell[0][uniques[0]==i].numpy().max()
        )


def test_train():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    params["quantile"] = 0.3
    params["ema"] = 0.01
    params["confidence_thresholds"] = [0.1, 0.9]
    params["num_steps"] = 40
    trainer = PromixNaive(params)
    trainer.train()

test_train()