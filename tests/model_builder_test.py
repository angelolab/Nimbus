import json
import os
import tempfile

import time
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import toml

from cell_classification.model_builder import ModelBuilder
from cell_classification.segmentation_data_prep import (feature_description,
                                                        parse_dict)

from segmentation_data_prep_test import prep_object_and_inputs

tf.config.run_functions_eagerly(True)


def test_prep_loss(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = ModelBuilder(config_params)
        loss_fn = trainer.prep_loss()

        # sanity check outputs of loss function
        loss = loss_fn(
            tf.constant(np.random.randint(0, 2, (1, 256, 256, 1))),
            tf.sigmoid(tf.constant(np.random.rand(1, 256, 256, 1), dtype=tf.float32)),
        )
        assert loss.shape == (1, 256, 256)
        assert loss.dtype == tf.float32
        assert tf.reduce_mean(loss) > 0
        assert tf.reduce_min(loss) >= 0


def test_prep_data(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        print(os.path.exists(temp_dir))
        tf_record_paths = []
        for i in range(2):
            data_prep, _, _, _ = prep_object_and_inputs(
                temp_dir, dataset="testdata_{}".format(i), num_folders=10,
                scale=[0.5, 1.0, 1.5, 2.0, 5.0]*2
            )
            data_prep.tf_record_path = temp_dir
            data_prep.make_tf_record()
            tf_record_path = os.path.join(
                data_prep.tf_record_path, "testdata_{}.tfrecord".format(i)
            )
            tf_record_paths.append(tf_record_path)
        config_params["record_path"] = tf_record_paths
        config_params["dataset_names"] = ["test1", "test2"]
        config_params["dataset_sample_probs"] = [0.5, 0.5]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["num_steps"] = 20
        config_params["num_validation"] = [2, 2]
        config_params["num_test"] = [2, 2]
        config_params["batch_size"] = 2
        trainer = ModelBuilder(config_params)
        trainer.prep_data()

        # check if correct number of datasets are loaded
        assert len(trainer.validation_datasets) == 2
        assert len(trainer.train_datasets) == 2
        assert len(trainer.test_datasets) == 2

        # check if train_dataset batches consist of samples from both datasets
        batch_list = []
        for batch in trainer.train_dataset:
            batch_dset = [b.decode() for b in batch['dataset'].numpy()]
            batch_list += batch_dset
        assert set(batch_list) == set(["testdata_0", "testdata_1"])

        # check if correct number of samples per batch is returned
        trainer.validation_datasets = [validation_dataset.map(
            trainer.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        ) for validation_dataset in trainer.validation_datasets]
        trainer.test_datasets = [test_dataset.map(
            trainer.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        ) for test_dataset in trainer.test_datasets]
        trainer.train_dataset = trainer.train_dataset.map(
            trainer.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        )
        assert next(iter(trainer.train_dataset))[0].shape[0] == config_params["batch_size"]
        for validation_dataset in trainer.validation_datasets:
            assert next(iter(validation_dataset))[0].shape[0] == config_params["batch_size"]
        for test_dataset in trainer.test_datasets:
            assert next(iter(test_dataset))[0].shape[0] == config_params["batch_size"]

        # check if samples only contains two files (inputs, targets)
        assert len(next(iter(trainer.train_dataset))) == 2
        for validation_dataset in trainer.validation_datasets:
            assert len(next(iter(validation_dataset))) == 2
        for test_dataset in trainer.test_datasets:
            assert len(next(iter(test_dataset))) == 2

        # check if in eval mode validation samples contain all original example keys
        trainer.params["eval"] = True
        trainer.prep_data()
        val_dset = iter(trainer.validation_datasets[0])
        val_batch = next(val_dset)
        assert set(val_batch.keys()) == set(
            [
                "mplex_img", "binary_mask", "instance_mask", "nuclei_img", "membrane_img",
                "folder_name", "marker", "dataset", "imaging_platform", "marker_activity_mask",
                "activity_df", 'tissue_type'
            ]
        )
        # check if in eval mode validation samples from one validation dataset only contain samples
        # from one dataset
        batch_list = []
        for batch in trainer.validation_datasets[0]:
            batch_dset = [b.decode() for b in batch['dataset'].numpy()]
            batch_list += batch_dset
        assert set(batch_list) == set(["testdata_0"])

        batch_list = []
        for batch in trainer.validation_datasets[1]:
            batch_dset = [b.decode() for b in batch['dataset'].numpy()]
            batch_list += batch_dset
        assert set(batch_list) == set(["testdata_1"])

        # check if fov_filter works correctly when called inside prep_data
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["batch_size"] = 1
        # prepare data splits json and add to params
        split = {"train": ["fov_0", "fov_1", "fov_2"], "validation": ["fov_3"], "test": ["fov_4"]}
        with open(os.path.join(temp_dir, "data_splits.json"), "w") as f:
            json.dump(split, f)
        config_params["data_splits"] = [os.path.join(temp_dir, "data_splits.json")]
        trainer = ModelBuilder(config_params)
        trainer.prep_data()
        # check if we filtered the fovs correctly
        fovs = {"train": [], "validation": [], "test": []}
        for example in trainer.train_dataset:
            fovs["train"].append(example["folder_name"].numpy()[0].decode())

        for example in trainer.validation_datasets[0]:
            fovs["validation"].append(example["folder_name"].numpy()[0].decode())

        for example in trainer.test_datasets[0]:
            fovs["test"].append(example["folder_name"].numpy()[0].decode())

        for key in split.keys():
            assert set(split[key]) == set(fovs[key])


def test_prep_model(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_params["path"] = temp_dir

        trainer = ModelBuilder(config_params)
        trainer.prep_model()

        # check if right objects are instantiated
        assert isinstance(trainer.model, tf.keras.Model)
        assert isinstance(trainer.optimizer, tf.keras.optimizers.Optimizer)

        # check if all the directories were created
        assert os.path.exists(trainer.params["log_dir"])
        assert os.path.exists(trainer.params["model_dir"])

        # check if model path is taken from params.toml if it exists
        trainer.params["model_path"] = os.path.join(temp_dir, "test_dir", "test.h5")
        trainer.prep_model()
        assert trainer.params["model_path"] == os.path.join(temp_dir, "test_dir", "test.h5")

        # test vanilla unet
        config_params["model"] = "VanillaUNet"
        trainer = ModelBuilder(config_params)
        trainer.prep_model()
        assert isinstance(trainer.model, tf.keras.Model)


def test_train_step(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["num_validation"] = [2]
        config_params["num_test"] = [2]
        config_params["batch_size"] = 2
        config_params["test"] = True
        config_params["weight_decay"] = 1e-4
        config_params["snap_steps"] = 5
        config_params["val_steps"] = 5
        trainer = ModelBuilder(config_params)
        trainer.prep_data()
        trainer.prep_model()
        trainer.train_dataset = trainer.train_dataset.map(
            trainer.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        )
        x, y = next(iter(trainer.train_dataset))

        # check if train_step returns correct loss
        loss = trainer.train_step(trainer.model, x, y)
        assert loss.dtype == tf.float32
        assert loss > 0

        # check with vanilla unet
        config_params["model"] = "VanillaUNet"
        trainer = ModelBuilder(config_params)
        trainer.prep_data()
        trainer.prep_model()
        trainer.train_dataset = trainer.train_dataset.map(
            trainer.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        )
        x, y = next(iter(trainer.train_dataset))
        loss = trainer.train_step(trainer.model, x, y)
        assert loss.dtype == tf.float32


def test_train(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["num_validation"] = [2]
        config_params["num_test"] = [2]
        config_params["batch_size"] = 2
        config_params["test"] = True
        config_params["weight_decay"] = 1e-4
        config_params["snap_steps"] = 5
        config_params["val_steps"] = 5

        trainer = ModelBuilder(config_params)
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


def test_tensorboard_callbacks(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 6
        config_params["dataset_sample_probs"] = [1.0]
        config_params["num_validation"] = [2]
        config_params["num_test"] = [2]
        config_params["batch_size"] = 2
        config_params["test"] = True
        config_params["weight_decay"] = 1e-4
        config_params["snap_steps"] = 5
        config_params["val_steps"] = 5
        #
        trainer = ModelBuilder(config_params)
        trainer.train()
        # check if loss history is written to file
        import time
        time.sleep(5)
        assert "wandb" in os.listdir(trainer.params["log_dir"])[0]
        # check if model checkpoint is written to file
        assert os.path.split(trainer.params["model_path"])[-1] in os.listdir(
            trainer.params["model_dir"]
        )


def test_predict(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["num_validation"] = [2]
        config_params["num_test"] = [2]
        config_params["batch_size"] = 2
        config_params["test"] = True
        config_params["snap_steps"] = 5000
        config_params["val_steps"] = 5000

        trainer = ModelBuilder(config_params)
        trainer.train()
        val_dset = trainer.validation_datasets[0].map(
            trainer.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        )
        val_batch = next(iter(val_dset))
        predictions = trainer.predict(val_batch[0])

        # check if predictions have the right shape, format and range
        assert predictions.shape == (2, 256, 256, 1)
        assert predictions.dtype == np.float32
        assert np.max(predictions) <= 1
        assert np.min(predictions) >= 0

        # check if predictions work for a single image
        predictions = trainer.predict(val_batch[0][0])
        assert predictions.shape == (1, 256, 256, 1)


def test_predict_dataset(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 2
        config_params["dataset_sample_probs"] = [1.0]
        config_params["num_validation"] = [2]
        config_params["num_test"] = [2]
        config_params["batch_size"] = 2
        config_params["snap_steps"] = 5000
        config_params["val_steps"] = 5000
        trainer = ModelBuilder(config_params)
        trainer.train()

        val_dset = trainer.validation_datasets[0]
        single_example_list = trainer.predict_dataset(val_dset)

        # check if predict returns a list with the right number of items
        assert [len(single_example_list)] == config_params["num_validation"]

        # check if params were saved to file
        assert "params.toml" in os.listdir(config_params["model_dir"])

        # check if examples get serialized correctly
        single_example_list = trainer.predict_dataset(val_dset, save_predictions=True)
        params = trainer.params
        for i in range(config_params["num_validation"][0]):
            assert str(i).zfill(4) + "_pred.hdf" in list(os.listdir(config_params["eval_dir"]))
        eval_dir = os.path.join(config_params["eval_dir"], str(0).zfill(4) + "_pred.hdf")
        with h5py.File(eval_dir, "r") as f:
            assert f["prediction"].shape == (256, 256, 1)
            assert f["marker_activity_mask"].shape == (256, 256, 1)
            assert set(list(f.keys())) == set(list(single_example_list[0].keys()))


def test_add_weight_decay(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["num_validation"] = [2]
        config_params["num_test"] = [2]
        config_params["batch_size"] = 2
        config_params["test"] = True
        config_params["weight_decay"] = 1e-3

        trainer = ModelBuilder(config_params)
        trainer.prep_model()

        # check if weight decay is added to the model losses
        assert len(trainer.model.losses) > 1

        # check if loss is higher with weight decay than without weight decay
        tf.random.set_seed(42)
        trainer = ModelBuilder(config_params)
        trainer.prep_data()
        trainer.prep_model()
        loss_with_weight_decay = trainer.validate(trainer.validation_datasets[0])

        config_params["weight_decay"] = False
        tf.random.set_seed(42)
        trainer_no_decay = ModelBuilder(config_params)
        trainer_no_decay.prep_data()
        trainer_no_decay.prep_model()
        loss_without_weight_decay = trainer_no_decay.validate(trainer.validation_datasets[0])

        assert loss_with_weight_decay > loss_without_weight_decay


def test_quantile_filter(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["num_validation"] = [0]
        config_params["num_test"] = [0]
        config_params["batch_size"] = 1
        trainer = ModelBuilder(config_params)
        trainer.prep_data()
        unfiltered_num_cells = []
        for example in trainer.train_dataset:
            df = pd.read_json(example["activity_df"].numpy()[0].decode())
            unfiltered_num_cells.append(np.sum(df.activity))
        config_params["filter_quantile"] = 0.8
        trainer = ModelBuilder(config_params)
        trainer.prep_data()
        filtered_num_cells = []
        for example in trainer.train_dataset:
            df = pd.read_json(example["activity_df"].numpy()[0].decode())
            filtered_num_cells.append(np.sum(df.activity))

        # check if we really reduced the number of examples
        assert len(unfiltered_num_cells) > len(filtered_num_cells)

        # check if filtered examples contain more cells than unfiltered examples
        diff = [
            num_cells for num_cells in unfiltered_num_cells if num_cells not in filtered_num_cells
        ]
        assert np.max(diff) < np.min(filtered_num_cells)

        # check if dataset_num_pos_dict.json was saved and contains the right values
        assert os.path.exists(trainer.num_pos_dict_path)

        with open(trainer.num_pos_dict_path, "r") as f:
            num_pos_dict = json.load(f)

        assert np.array_equal(sorted(num_pos_dict["CD4"]), sorted(unfiltered_num_cells))


def test_gen_prep_batches_fn(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["batch_constituents"] = ["mplex_img", "binary_mask", "nuclei_img",
                                               "membrane_img"]
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["batch_size"] = 1
        config_params["test"] = True
        config_params["num_validation"] = [2]
        config_params["num_test"] = [2]
        trainer = ModelBuilder(config_params)
        trainer.prep_data()
        example = next(iter(trainer.train_dataset))
        prep_batches_4 = trainer.prep_batches
        prep_batches_2 = trainer.gen_prep_batches_fn(keys=["mplex_img", "binary_mask"])

        # check if each batch contains the above specified constituents
        batch_2 = prep_batches_2(example)
        assert batch_2[0].shape[-1] == 2
        assert np.array_equal(batch_2[0], tf.concat([
            example["mplex_img"], tf.cast(example["binary_mask"], tf.float32)
            ], axis=-1)
        )

        batch_4 = prep_batches_4(example)
        assert batch_4[0].shape[-1] == 4
        assert np.array_equal(batch_4[0], tf.concat([
            example["mplex_img"], tf.cast(example["binary_mask"], tf.float32),
            example["nuclei_img"], example["membrane_img"]
            ], axis=-1)
        )


def test_fov_filter(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        config_params["record_path"] = [tf_record_path]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["dataset_names"] = ["test1"]
        config_params["num_steps"] = 20
        config_params["dataset_sample_probs"] = [1.0]
        config_params["batch_size"] = 1
        # prepare data splits json and add to params
        split = {"train": ["fov_0", "fov_1", "fov_2"], "validation": ["fov_3"], "test": ["fov_4"]}
        trainer = ModelBuilder(config_params)
        dataset = tf.data.TFRecordDataset(tf_record_path)
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
        dataset = dataset.map(parse_dict)
        for fov in split.values():
            fov_list = []
            dataset_filtered = trainer.fov_filter(dataset, fov)
            for example in dataset_filtered:
                fov_list.append(example["folder_name"].numpy().decode())
            assert set(fov) == set(fov_list)


def test_dset_marker_filter(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        tf_record_paths = []
        for i in range(2):
            data_prep, _, _, _ = prep_object_and_inputs(
                temp_dir, dataset="testdata_{}".format(i), num_folders=10,
                scale=[0.5, 1.0, 1.5, 2.0, 5.0]*2
            )
            data_prep.tf_record_path = temp_dir
            data_prep.make_tf_record()
            tf_record_path = os.path.join(
                data_prep.tf_record_path, "testdata_{}.tfrecord".format(i)
            )
            tf_record_paths.append(tf_record_path)
        config_params["record_path"] = tf_record_paths
        config_params["dataset_names"] = ["test1", "test2"]
        config_params["dataset_sample_probs"] = [0.5, 0.5]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["num_steps"] = 20
        config_params["num_validation"] = [2, 2]
        config_params["num_test"] = [2, 2]
        config_params["batch_size"] = 2
        config_params["exclude_dset_marker"] = [["testdata_0"], ["CD4"]]
        trainer = ModelBuilder(config_params)
        # check if CD4 is filtered for dataset test1
        dataset = tf.data.TFRecordDataset(tf_record_paths[0])
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
        dataset = dataset.map(parse_dict)
        dataset_filtered = trainer.dset_marker_filter(
            dataset, config_params["exclude_dset_marker"]
        )
        for example in dataset_filtered:
            assert example["marker"].numpy().decode() != "CD4"
        # check if CD4 is not filtered for dataset test2
        dataset = tf.data.TFRecordDataset(tf_record_paths[1])
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
        dataset = dataset.map(parse_dict)
        exclude_dset_marker = [["testdata_0"], ["CD4"]]
        dataset_filtered = trainer.dset_marker_filter(dataset, exclude_dset_marker)
        markers = []
        for example in dataset_filtered:
            markers.append(example["marker"].numpy().decode())
        assert "CD4" in markers


def test_predict_dataset_list(config_params):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(
            temp_dir, dataset="testdata", num_folders=10,
            scale=[0.5, 1.0, 1.5, 2.0, 5.0]*2
        )
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(
            data_prep.tf_record_path, "testdata.tfrecord"
        )
        config_params["record_path"] = [tf_record_path]
        config_params["dataset_names"] = ["test1"]
        config_params["dataset_sample_probs"] = [0.5]
        config_params["path"] = temp_dir
        config_params["experiment"] = "test"
        config_params["num_validation"] = [2]
        config_params["batch_size"] = 2
        config_params["model_path"] = os.path.join(temp_dir, "test.pkl")
        trainer = ModelBuilder(config_params)
        trainer.prep_model()
        trainer.prep_data()
        trainer.model.save_weights(config_params["model_path"])
        df = trainer.predict_dataset_list(
            datasets=trainer.validation_datasets, fname="validation_predictions")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns.tolist()) == set([
            'activity', 'prediction', 'cell_type', 'marker', 'labels', 'fov', 'dataset'])
        # check if df got stored
        assert os.path.exists(
            os.path.join(trainer.params["model_dir"], "validation_predictions.csv")
        )
