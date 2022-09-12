import pytest
import tempfile
import numpy as np
import tensorflow as tf
from segmentation_data_prep_test import prep_object_and_inputs
import os
import toml
from train import Trainer


def test_prep_loss():
    with tempfile.TemporaryDirectory() as temp_dir:
        params = toml.load("cell_classification/configs/params.toml")
        trainer = Trainer(params)
        loss_fn = trainer.prep_loss(2)

        # sanity check outputs of loss function
        loss = loss_fn(
            tf.constant(np.random.randint(0, 2, (1, 256, 256, 1))),
            tf.sigmoid(tf.constant(np.random.rand(1, 256, 256, 1), dtype=tf.float32)),
        )
        assert loss.shape == (1, 256, 256)
        assert loss.dtype == tf.float32
        assert tf.reduce_mean(loss) > 0
        assert tf.reduce_min(loss) >= 0


def test_prep_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        # trainer, params = prep_trainer(temp_dir)
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["num_epochs"] = 1
        params["num_validation"] = 2
        params["batch_size"] = 2
        trainer = Trainer(params)
        trainer.prep_data()

        # check if correct number of samples per batch is returned
        assert next(iter(trainer.train_dataset))[0].shape[0] == params["batch_size"]
        assert next(iter(trainer.validation_dataset))[0].shape[0] == params["batch_size"]

        # check if samples only contains two files (inputs, targets)
        assert len(next(iter(trainer.train_dataset))) == 2
        assert len(next(iter(trainer.validation_dataset))) == 2


def test_prep_model():
    params = toml.load("cell_classification/configs/params.toml")
    trainer = Trainer(params)
    trainer.prep_model()

    # check if right objects are instantiated
    assert isinstance(trainer.model, tf.keras.Model)
    assert isinstance(trainer.optimizer, tf.keras.optimizers.Optimizer)

    # check if all the directories were created
    assert os.path.exists(trainer.log_dir)
    assert os.path.exists(trainer.model_dir)

    # check if callbacks were created
    assert isinstance(trainer.train_callbacks[0], tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(trainer.train_callbacks[1], tf.keras.callbacks.LearningRateScheduler)
    assert isinstance(trainer.train_callbacks[2], tf.keras.callbacks.TensorBoard)


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
        trainer = Trainer(params)
        trainer.train()

        # check if loss_history is of type tf.keras.callbacks.History
        assert isinstance(trainer.loss_history, tf.keras.callbacks.History)

        # check if model is trained for correct number of epochs
        assert len(trainer.loss_history.history["loss"]) == params["num_epochs"]

        # check if loss history is written to file
        assert "tfevents" in os.listdir(os.path.join(trainer.log_dir, "train"))[0]
        assert "tfevents" in os.listdir(os.path.join(trainer.log_dir, "validation"))[0]

        # check if model checkpoint is written to file
        assert params["experiment"] + ".h5" in os.listdir(trainer.model_dir)
