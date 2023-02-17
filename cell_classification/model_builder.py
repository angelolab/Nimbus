import os
import argparse
import tensorflow as tf
import toml
from augmentation_pipeline import prepare_tf_aug, py_aug, get_augmentation_pipeline
from post_processing import merge_activity_df, process_to_cells
from segmentation_data_prep import parse_dict, feature_description
from deepcell.model_zoo.panopticnet import PanopticNet
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from deepcell.utils.train_utils import rate_scheduler, get_callbacks, count_gpus
from deepcell import losses
from semantic_head import create_semantic_head
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd
from loss import Loss
from tqdm import tqdm
from time import time
import json


class ModelBuilder:
    """Builds, trains and writes validation metrics for models"""

    def __init__(self, params):
        """Initialize the trainer with the parameters from the config file
        Args:
            params (dict): Dictionary of parameters from the config file
        """
        self.params = params
        self.params["model"] = "ModelBuilder"
        self.num_gpus = count_gpus()
        if "batch_constituents" in list(self.params.keys()):
            self.prep_batches = self.gen_prep_batches_fn(self.params["batch_constituents"])
        else:
            self.prep_batches = self.gen_prep_batches_fn()
        # make prep_batches a callable static method
        self.prep_batches = staticmethod(self.prep_batches).__func__

    def prep_data(self):
        """Prepares training and validation data"""
        # make datasets and splits
        datasets = [
            tf.data.TFRecordDataset(record_path) for record_path in self.params["record_path"]
        ]
        datasets = [
            dataset.map(
                lambda x: tf.io.parse_single_example(x, feature_description),
                num_parallel_calls=tf.data.AUTOTUNE,
            ) for dataset in datasets
        ]
        datasets = [
            dataset.map(parse_dict, num_parallel_calls=tf.data.AUTOTUNE) for dataset in datasets
        ]

        # filter out sparse samples
        if "filter_quantile" in self.params.keys():
            datasets = [
                self.quantile_filter(dataset, record_path) for dataset, record_path in
                zip(datasets, self.params["record_path"])
            ]

        # split into train, validation and test
        self.validation_datasets = [
            dataset.take(num_validation) for dataset, num_validation in zip(
                datasets, self.params["num_validation"])
            ]
        datasets = [dataset.skip(num_validation) for dataset, num_validation in zip(
            datasets, self.params["num_validation"])
        ]
        self.test_datasets = [
            dataset.take(num_test) for dataset, num_test in zip(
                datasets, self.params["num_test"])
            ]
        self.train_datasets = [
            dataset.skip(num_test) for dataset, num_test in zip(
                datasets, self.params["num_test"])
        ]
        # add external validation datasets
        if "external_validation_path" in self.params.keys():
            external_validation_datasets = [
                tf.data.TFRecordDataset(record_path) for record_path in
                self.params["external_validation_path"]
            ]
            external_validation_datasets = [
                dataset.map(
                    lambda x: tf.io.parse_single_example(x, feature_description),
                    num_parallel_calls=tf.data.AUTOTUNE,
                ) for dataset in external_validation_datasets
            ]
            external_validation_datasets = [
                dataset.map(parse_dict, num_parallel_calls=tf.data.AUTOTUNE) for dataset in
                external_validation_datasets
            ]
            self.external_validation_datasets = external_validation_datasets
            self.external_validation_names = self.params["external_validation_names"]

        if "num_training" in self.params.keys() and self.params["num_training"] is not None:
            self.train_datasets = [
                train_dataset.take(num_training) for train_dataset, num_training
                in zip(self.train_datasets, self.params["num_training"])
            ]

        # merge datasets with tf.data.Dataset.sample_from_datasets
        self.train_dataset = tf.data.Dataset.sample_from_datasets(
            datasets=self.train_datasets, weights=self.params["dataset_sample_probs"]
        )

        # shuffle, batch and augment the datasets
        self.train_dataset = self.train_dataset.shuffle(self.params["shuffle_buffer_size"]).batch(
            self.params["batch_size"] * np.max([self.num_gpus, 1])
        )
        self.validation_datasets = [validation_dataset.batch(
            self.params["batch_size"] * np.max([self.num_gpus, 1])
        ) for validation_dataset in self.validation_datasets]
        self.test_datasets = [test_dataset.batch(
            self.params["batch_size"] * np.max([self.num_gpus, 1])
        ) for test_dataset in self.test_datasets]

        self.dataset_names = self.params["dataset_names"]

    def prep_model(self):
        """Prepares the model for training"""
        # prepare folders
        self.params["model_dir"] = os.path.join(
            os.path.normpath(self.params["path"]), self.params["experiment"]
        )
        self.params["log_dir"] = os.path.join(self.params["model_dir"], "logs", str(int(time())))
        os.makedirs(self.params["model_dir"], exist_ok=True)
        os.makedirs(self.params["log_dir"], exist_ok=True)
        if "model_path" not in self.params.keys() or self.params["model_path"] is None:
            self.params["model_path"] = os.path.join(
                self.params["model_dir"], "{}.h5".format(self.params["experiment"])
            )
        self.params["loss_path"] = os.path.join(
            self.params["model_dir"], "{}.npz".format(self.params["experiment"])
        )

        # initialize optimizer and lr scheduler
        # replace with AdamW when available
        self.lr_sched = CosineDecay(
            initial_learning_rate=self.params["lr"],
            decay_steps=self.params["num_steps"],
            alpha=1e-6,
        )
        self.optimizer = Adam(learning_rate=self.lr_sched, clipnorm=0.001)

        # initialize model
        if "test" in self.params.keys() and self.params["test"]:
            self.model = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(
                        1, (3, 3), input_shape=self.params["input_shape"], padding="same",
                        name="semantic_head", activation="sigmoid", data_format="channels_last",
                    )]
            )
        else:
            self.model = PanopticNet(
                backbone=self.params["backbone"], input_shape=self.params["input_shape"],
                norm_method="std", num_semantic_classes=self.params["classes"],
                create_semantic_head=create_semantic_head, location=self.params["location"],
            )

        loss = {}
        # Give losses for all of the semantic heads
        for layer in self.model.layers:
            if layer.name.startswith("semantic_"):
                loss[layer.name] = self.prep_loss()

        if "weight_decay" in self.params.keys():
            self.add_weight_decay()
        self.model.compile(loss=loss, optimizer=self.optimizer)

    @staticmethod
    @tf.function
    def train_step(model, x, y):
        """Trains the model for one step"""
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = model.compute_loss(x, y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def distributed_train_step(self, model, x, y):
        """Trains the model for one step on multiple GPUs"""
        loss = self.strategy.run(self.train_step, args=(model, x, y))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    def train(self):
        """Calls prep functions and starts training loops"""
        print("Training on", self.num_gpus, "GPUs.")
        # initialize data and model
        self.prep_data()

        # make transformations on the training dataset
        augmentation_pipeline = get_augmentation_pipeline(self.params)
        tf_aug = prepare_tf_aug(augmentation_pipeline)
        self.train_dataset = self.train_dataset.map(
            lambda x: py_aug(x, tf_aug), num_parallel_calls=tf.data.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.map(
            self.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.prefetch(tf.data.AUTOTUNE)

        if self.num_gpus > 1:
            # set up distributed training
            self.strategy = tf.distribute.MirroredStrategy()
            self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            with self.strategy.scope():
                self.prep_model()
                checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            print("Distributed training on {} devices".format(self.strategy.num_replicas_in_sync))
            train_step = self.distributed_train_step
        else:
            self.prep_model()
            train_step = self.train_step
        #
        with open(os.path.join(self.params["model_dir"], "params.toml"), "w") as f:
            toml.dump(self.params, f)

        self.summary_writer = tf.summary.create_file_writer(self.params["log_dir"])
        self.step = 0
        self.global_val_loss = []
        self.val_loss_history = {}
        self.train_loss_tmp = []
        while self.step < self.params["num_steps"]:
            for x, y in tqdm(self.train_dataset):
                train_loss = train_step(self.model, x, y)
                self.train_loss_tmp.append(train_loss)
                self.step += 1
                self.tensorboard_callbacks(x, y)
                if self.step > self.params["num_steps"]:
                    break

    def tensorboard_callbacks(self, x, y):
        """Logs training metrics to Tensorboard
        Args:
            x (tf.Tensor): input image
            y (tf.Tensor): ground truth labels
        """
        if self.step % 10 == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    "train_loss", tf.reduce_mean(self.train_loss_tmp), step=self.step
                )
                tf.summary.scalar(
                    "lr", self.model.optimizer._decayed_lr(tf.float32), step=self.step
                )
            print(
                "Step: {step}, loss {loss}".format(
                    step=self.step, loss=tf.reduce_mean(self.train_loss_tmp))
            )
            self.train_loss_tmp = []
        if self.step % self.params["snap_steps"] == 0:
            print("Saving training snapshots")
            if self.num_gpus > 1:
                x = self.strategy.experimental_local_results(x)[0]
                y_pred = self.model(x, training=False)
                y_pred = self.strategy.experimental_local_results(y_pred)[0]
                y = self.strategy.experimental_local_results(y)[0]
            else:
                y_pred = self.model(x, training=False)
            with self.summary_writer.as_default():
                tf.summary.image(
                    "x_0 | y | y_pred",
                    tf.concat([
                        x[:1, ..., :1],
                        x[:1, ..., 1:2] * 0.25 + tf.cast(y[:1, ..., :1], tf.float32),
                        y_pred[:1, ..., :1]],  axis=0,
                    ),
                    step=self.step,
                )
        # run validation and write to tensorboard
        if self.step % self.params["val_steps"] == 0:
            print("Running validation...")
            for validation_dataset, dataset_name in zip(
                self.validation_datasets, self.dataset_names
            ):
                validation_dataset = validation_dataset.map(
                    self.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
                )
                val_loss = self.model.evaluate(validation_dataset, verbose=1)
                print("Validation loss:", val_loss)
                if dataset_name not in self.val_loss_history.keys():
                    self.val_loss_history[dataset_name] = []
                self.val_loss_history[dataset_name].append(val_loss)
                with self.summary_writer.as_default():
                    tf.summary.scalar(dataset_name + "_val", val_loss, step=self.step)
            val_loss = np.mean([val_loss[-1] for val_loss in self.val_loss_history.values()])
            self.global_val_loss.append(val_loss)
            with self.summary_writer.as_default():
                tf.summary.scalar("global_val", val_loss, step=self.step)
            if val_loss <= tf.reduce_min(self.global_val_loss):
                print("Saving model to", self.params["model_path"])
                self.model.save_weights(self.params["model_path"])
            # run external validation
            if hasattr(self, "external_validation_datasets"):
                for validation_dataset, dataset_name in zip(
                    self.external_validation_datasets, self.external_dataset_names
                ):
                    validation_dataset = validation_dataset.map(
                        self.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
                    )
                    val_loss = self.model.evaluate(validation_dataset, verbose=1)
                    print("Validation loss:", val_loss)
                    if dataset_name not in self.val_loss_history.keys():
                        self.val_loss_history[dataset_name] = []
                    self.val_loss_history[dataset_name].append(val_loss)
                    with self.summary_writer.as_default():
                        tf.summary.scalar(dataset_name + "_val", val_loss, step=self.step)
            if "save_model_on_dataset_name" in self.params.keys():
                current = self.val_loss_history[self.params["save_model_on_dataset_name"]][-1]
                if current <= self.best_val_loss[self.params["save_model_on_dataset_name"]]:
                    print("Saving model to", self.params["model_path"])
                    self.model.save_weights(self.params["model_path"]+"_best.pkl")

    def prep_loss(self):
        """Prepares the loss function for the model
        Args:
            n_classes (int): Number of semantic classes in the dataset
        Returns:
            loss_fn (function): Loss function for the model
        """
        loss_fn = Loss(
            self.params["loss_fn"],
            self.params["loss_selective_masking"],
            **self.params["loss_kwargs"]
        )
        return loss_fn

    def gen_prep_batches_fn(self, keys=["mplex_img", "binary_mask"]):
        """Generates a function that preprocesses batches for training
        Args:
            keys (list): List of keys to concatenate into a single batch
        Returns:
            prep_batches (function): Function that preprocesses batches for training
        """

        def prep_batches(batch):
            """Preprocess batches for training
            Args:
                batch (dict):
                    Dictionary of tensors and strings containing data from a single batch
            Returns:
                inputs (tf.Tensor):
                    Batch of images
                targets (tf.Tensor):
                    Batch of labels
            """
            inputs = tf.concat(
                [tf.cast(batch[key], tf.float32) for key in keys], axis=-1
            )
            targets = batch["marker_activity_mask"]
            return inputs, targets

        return prep_batches

    def predict(self, image):
        """Runs inference on a single image or a batch of images
        Args:
            image np.ndarray or tf.Tensor:
                Image to run inference on shape (H, W, C) or (N, H, W, C)
        Returns:
            prediction (np.ndarray):
                Prediction from the model (N, H, W, 1)
        """
        if image.ndim != 4:
            image = tf.expand_dims(image, axis=0)
        prediction = self.model.predict(image)
        return prediction

    def load_model(self, path):
        """Loads a model from a path
        Args:
            path (str):
                Path to the model checkpoint file
        """
        if not hasattr(self, "model") or self.model is None:
            self.prep_model()
        self.model.load_weights(path)

    def validate(self, val_dset):
        """Runs inference on a validation dataset
        Args:
            val_dset (tf.data.Dataset):
                Dataset to run inference on
        Returns:
            loss (float):
                Loss on the validation dataset
        """
        val_dset = val_dset.map(self.prep_batches, num_parallel_calls=tf.data.AUTOTUNE)
        loss = self.model.evaluate(val_dset)
        return loss

    def predict_dataset(self, test_dset, save_predictions=False):
        """Runs inference on a test dataset
        Args:
            test_dset (tf.data.Dataset):
                Dataset to run inference on
            save_predictions (bool):
                Whether to save the predictions to a file
        Returns:
            predictions (np.ndarray):
                Predictions from the model
        """
        # prepare output folder
        if "eval_dir" not in self.params.keys() and save_predictions:
            self.params["eval_dir"] = os.path.join(self.params["model_dir"], "eval")
            os.makedirs(self.params["eval_dir"], exist_ok=True)

        single_example_list = []
        j = 0
        for sample in tqdm(test_dset):
            sample["prediction"] = self.predict(self.prep_batches(sample)[0])

            # split batches to single samples
            # split numpy arrays to list of arrays
            for key in sample.keys():
                sample[key] = np.split(sample[key], sample[key].shape[0])
            # iterate over samples in batch
            for i in range(len(sample["prediction"])):
                single_example = {}
                for key in sample.keys():
                    single_example[key] = np.squeeze(sample[key][i], axis=0)
                    if single_example[key].dtype == object:
                        single_example[key] = sample[key][i].item().decode("utf-8")
                # decode activity df
                if not isinstance(single_example["activity_df"], pd.DataFrame):
                    single_example["activity_df"] = pd.read_json(single_example["activity_df"])
                # calculate cell level predictions
                single_example["prediction_mean"], pred_df = process_to_cells(
                    single_example["instance_mask"], single_example["prediction"]
                )
                single_example["activity_df"] = merge_activity_df(
                    single_example["activity_df"], pred_df
                )
                # save single example to file
                if save_predictions:
                    fname = os.path.join(self.params["eval_dir"], str(j).zfill(4) + "_pred.hdf")
                    j += 1
                    with h5py.File(fname, "w") as f:
                        for key in [key for key in single_example.keys() if key != "activity_df"]:
                            f.create_dataset(key, data=single_example[key])
                        f.create_dataset(
                            "activity_df", data=single_example["activity_df"].to_json()
                        )
                single_example_list.append(single_example)
        # save params to toml file
        with open(os.path.join(self.params["model_dir"], "params.toml"), "w") as f:
            toml.dump(self.params, f)
        return single_example_list

    def add_weight_decay(self):
        if self.params["weight_decay"] in [False, None]:
            return None
        alpha = self.params["weight_decay"]
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(
                layer, tf.keras.layers.Dense
            ):
                layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.kernel))
            if hasattr(layer, "bias_regularizer") and layer.use_bias:
                layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.bias))

    def quantile_filter(self, dataset, record_path):
        """Filter out training examples that contain less than a certain quantile per marker of
        positive cells
        Args:
            dataset (tf.data.Dataset):
                Dataset to filter
            record_path (str):
                Path to the tfrecord file
        Returns:
            dataset (tf.data.Dataset):
                Filtered dataset
        """
        print("Filtering out sparse training examples...")
        self.num_pos_dict_path = record_path.split(".tfrecord")[0] + \
            "num_pos_dict.json"
        if os.path.exists(self.num_pos_dict_path):
            with open(self.num_pos_dict_path, "r") as f:
                num_pos_dict = json.load(f)
        else:
            num_pos_dict = {}
            for example in tqdm(dataset):
                marker = tf.get_static_value(example["marker"]).decode("utf-8")
                activity_df = pd.read_json(
                    tf.get_static_value(example["activity_df"]).decode("utf-8")
                )
                if marker not in num_pos_dict.keys():
                    num_pos_dict[marker] = []
                num_pos_dict[marker].append(int(np.sum(activity_df.activity == 1)))

            # save num_pos_dict to file
            with open(self.num_pos_dict_path, "w") as f:
                json.dump(num_pos_dict, f)

        quantile_dict = {}
        for marker, pos_list in num_pos_dict.items():
            quantile_dict[marker] = np.quantile(pos_list, self.params["filter_quantile"])

        def predicate(marker, activity_df):
            """Helper function that returns true if the number of positive cells is above the
            quantile threshold
            Args:
                marker (tf.Tensor):
                    Marker name of the example
                activity_df (tf.Tensor):
                    Activity dataframe of the example
            Returns:
                tf.Tensor:
                    True if the number of positive cells is above the quantile threshold
            """
            marker = tf.get_static_value(marker).decode("utf-8")
            activity_df = pd.read_json(tf.get_static_value(activity_df).decode("utf-8"))
            num_pos = tf.reduce_sum(tf.constant(activity_df.activity == 1, dtype=tf.float32))
            return tf.greater_equal(num_pos, quantile_dict[marker])

        dataset = dataset.filter(
            lambda example: tf.py_function(
                predicate, [example["marker"], example["activity_df"]], tf.bool
            )
        )
        return dataset


if __name__ == "__main__":
    print("CUDA_VISIBLE_DEVICES: " + str(os.getenv("CUDA_VISIBLE_DEVICES")))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        default="cell_classification/configs/params.toml",
    )
    args = parser.parse_args()
    params = toml.load(args.params)
    trainer = ModelBuilder(params)
    trainer.train()
