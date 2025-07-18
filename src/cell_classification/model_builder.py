import argparse
import json
import os
from time import time
from cell_classification.unet import build_model
from cell_classification.inference import segment_mean
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import toml
from io import StringIO
from deepcell.panopticnet import PanopticNet
from deepcell.utils import count_gpus
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from deepcell.semantic_head import create_semantic_head
from tqdm import tqdm
from cell_classification.metrics import calc_scores
from cell_classification.augmentation_pipeline import (
    get_augmentation_pipeline, prepare_tf_aug, py_aug)
from cell_classification.loss import Loss
from cell_classification.post_processing import (merge_activity_df,
                                                 process_to_cells)
from cell_classification.segmentation_data_prep import (feature_description,
                                                        parse_dict)
import wandb
from tensorflow.keras import mixed_precision


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
        # prepare folders
        self.params["model_dir"] = os.path.join(
            os.path.normpath(self.params["path"]), self.params["experiment"]
        )
        self.params["log_dir"] = os.path.join(self.params["model_dir"], "logs", str(int(time())))
        os.makedirs(self.params["model_dir"], exist_ok=True)
        os.makedirs(self.params["log_dir"], exist_ok=True)


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
        # select markers for training if put into config file
        if "exclude_dset_marker_dict" in self.params.keys():
            datasets = [
                self.dset_marker_filter(
                    dataset, self.params["exclude_dset_marker_dict"]) for dataset in datasets
            ]

        # split into train, validation and test
        if "data_splits" in self.params.keys():
            data_splits = []
            for fpath in self.params["data_splits"]:
                with open(fpath, "r") as f:
                    data_splits.append(json.load(f))
            self.validation_datasets = [
                self.fov_filter(dataset, data_split["validation"]) for dataset, data_split in zip(
                    datasets, data_splits
                )
            ]
            self.test_datasets = [
                self.fov_filter(dataset, data_split["test"]) for dataset, data_split in zip(
                    datasets, data_splits
                )
            ]
            self.train_datasets = [
                self.fov_filter(dataset, data_split["train"]) for dataset, data_split in zip(
                    datasets, data_splits
                )
            ]
        else:
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
            datasets=self.train_datasets, weights=self.params["dataset_sample_probs"],
            stop_on_empty_dataset=True
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
        if "model_path" not in self.params.keys() or self.params["model_path"] is None:
            self.params["model_path"] = os.path.join(
                self.params["model_dir"], "{}.h5".format(self.params["experiment"])
            )
            self.params["avg_model_path"] = os.path.join(
                self.params["model_dir"], "{}_avg.h5".format(self.params["experiment"])
            )
        self.params["loss_path"] = os.path.join(
            self.params["model_dir"], "{}.npz".format(self.params["experiment"])
        )

        # initialize optimizer and lr scheduler
        # replace with AdamW when available
        initial_lr = self.params["lr"]
        if self.num_gpus > 1:
            print(f"Scaling learning rate by {self.num_gpus} for distributed training.")
            initial_lr *= self.num_gpus # Linear scaling rule

        self.lr_sched = CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=self.params["num_steps"],
            alpha=self.params["terminal_lr"],
        )
        self.optimizer = Adam(learning_rate=self.lr_sched, clipnorm=self.params["gradient_clipping"])

        # initialize model
        if "test" in self.params.keys() and self.params["test"]:
            self.model = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(
                        1, (3, 3), input_shape=self.params["input_shape"], padding="same",
                        name="semantic_head", activation="sigmoid", data_format="channels_last",
                    )]
            )
        elif self.params["backbone"] == "VanillaUNet":
            self.model = build_model(
                nx=self.params["input_shape"][0], ny=self.params["input_shape"][1],
                channels=self.params["input_shape"][2], num_classes=1,
                data_format="channels_last", padding="REFLECT"
            )
        else:
            self.model = PanopticNet(
                backbone=self.params["backbone"], input_shape=self.params["input_shape"],
                norm_method="std", num_semantic_classes=self.params["classes"],
                create_semantic_head=create_semantic_head, location=self.params["location"],
            )
        
        if self.params.get("load_model", False):
            self.load_model(self.params["load_model"])
            print("Loaded model from", self.params["load_model"])

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
    def train_step(model, x, y, mixed_precision=False):
        """Trains the model for one step"""
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = model.compute_loss(x, y, y_pred)
        if mixed_precision: # Add a flag to your params
            loss = model.optimizer.get_scaled_loss(loss)
            scaled_gradients = tape.gradient(loss, model.trainable_variables)
            gradients = model.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def distributed_train_step(self, model, x, y, mixed_precision=False):
        """Trains the model for one step on multiple GPUs"""
        # Define the step function that computes the loss per replica.
        # Ensure the loss returned by model.compute_loss is the MEAN over the replica's batch.
        # Keras default losses usually do this.
        def step_fn(inputs):
            x_rep, y_rep = inputs
            with tf.GradientTape() as tape:
                y_pred = model(x_rep, training=True)
                # model.compute_loss (often via model.compiled_loss) typically returns
                # the SCALAR mean loss for the replica's batch.
                per_replica_loss = model.compute_loss(x=x_rep, y=y_rep, y_pred=y_pred)
                if mixed_precision:
                    per_replica_loss = model.optimizer.get_scaled_loss(per_replica_loss)
                # If using custom loss, ensure it's the mean, or scale here:
                # scaled_loss = per_replica_loss / self.strategy.num_replicas_in_sync
            
            # Gradients are computed based on the per_replica_loss
            scaled_gradients = tape.gradient(per_replica_loss, model.trainable_variables)
            gradients = model.optimizer.get_unscaled_gradients(scaled_gradients)
            # Apply gradients - MirroredStrategy handles averaging automatically
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return per_replica_loss # Return the per-replica mean loss

        # Run the step function on each replica
        per_replica_losses = self.strategy.run(step_fn, args=((x, y),))

        # Reduce (average) the per-replica losses to get the mean loss over the global batch
        mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return mean_loss

    def train(self):
        """Calls prep functions and starts training loops"""
        print("Training on", self.num_gpus, "GPUs.")

        if self.params.get("use_mixed_precision", False): # Add a flag to your params
            print("Using mixed precision 'mixed_float16'")
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
        self.stop_training = False
        self.early_stopping_patience = self.params.get("early_stopping_patience", 0)
        # initialize data and model
        self.prep_data()

        wandb.init(
                name=self.params["experiment"],
                project=self.params["project"],
                entity="kainmueller-lab",
                config=self.params,
                dir=self.params["log_dir"],
                mode=self.params["logging_mode"]
        )

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

        self.avg_model = tf.keras.models.clone_model(self.model)
        self.step = 0
        self.val_f1_history = []
        self.train_loss_tmp = []
        while True:
            for x, y in tqdm(self.train_dataset):
                train_loss = train_step(
                    self.model, x, y, self.params.get("use_mixed_precision", False)
                )
                self.train_loss_tmp.append(train_loss)
                self.step += 1
                self.tensorboard_callbacks(x, y)
                if self.early_stopping_patience > 0 and self.stop_training:
                    print("Early stopping triggered")
                    break
                if self.step > self.params["num_steps"]:
                    self.stop_training = True
                    break
                # set avg model weights as exponential moving average of model weights
                for avg_w, w in zip(self.avg_model.weights, self.model.weights):
                    avg_w.assign(0.99 * avg_w + 0.01 * w)
            if self.stop_training:
                break

        wandb.finish()

    def tensorboard_callbacks(self, x, y):
        """Logs training metrics to Tensorboard
        Args:
            x (tf.Tensor): input image
            y (tf.Tensor): ground truth labels
        """
        if self.step % 10 == 0:
            wandb.log({
                "train_loss": tf.reduce_mean(self.train_loss_tmp).numpy(),
                "lr": self.lr_sched(self.model.optimizer.iterations).numpy(),
                "step": self.step
            })
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
            wandb.log({
                "x_0": wandb.Image(x[:1, ..., :1]),
                "y": wandb.Image(x[:1, ..., 1:2] * 0.25 + tf.cast(y[:1, ..., :1], tf.float32)),
                "y_pred": wandb.Image(y_pred[:1, ..., :1]),
                "step": self.step
            })
        # run validation and write to tensorboard
        if self.step % self.params["val_steps"] == 0:
            print("Running validation...")
            metric_dict = {}
            for validation_dataset, dataset_name in zip(
                self.validation_datasets, self.dataset_names
            ):
                activity_df = self.predict_dataset_list(validation_dataset, save_predictions=False)
                for marker in activity_df.marker.unique():
                    tmp_df = activity_df[activity_df.marker == marker]
                    metrics = calc_scores(
                        gt=tmp_df["activity"].values, pred=tmp_df["prediction"].values, threshold=0.5
                    )
                    metric_dict[dataset_name + "/" + marker] = {
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "specificity": metrics["specificity"],
                    }
                # average over all markers
                metric_dict[dataset_name + "/avg"] = {
                    "precision": np.mean(
                        [v["precision"] for k, v in metric_dict.items() if dataset_name in k]
                    ),
                    "recall": np.mean(
                        [v["recall"] for k, v in metric_dict.items() if dataset_name in k]
                    ),
                    "f1_score": np.mean(
                        [v["f1_score"] for k, v in metric_dict.items() if dataset_name in k]
                    ),
                    "specificity": np.mean(
                        [v["specificity"] for k, v in metric_dict.items() if dataset_name in k]
                    ),
                }
            # average over all datasets
            metric_dict["avg"] = {
                "precision": np.mean(
                    [v["precision"] for k, v in metric_dict.items() if "avg" in k]
                ),
                "recall": np.mean(
                    [v["recall"] for k, v in metric_dict.items() if "avg" in k]
                ),
                "f1_score": np.mean(
                    [v["f1_score"] for k, v in metric_dict.items() if "avg" in k]
                ),
                "specificity": np.mean(
                    [v["specificity"] for k, v in metric_dict.items() if "avg" in k]
                ),
            }
            self.val_f1_history.append(metric_dict["avg"]["f1_score"])
            wandb.log(metric_dict)
            if metric_dict["avg"]["f1_score"] >= tf.reduce_max(self.val_f1_history):
                print("Saving model to", self.params["model_path"])
                self.model.save_weights(self.params["model_path"])
                self.avg_model.save_weights(self.params["avg_model_path"])
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
                    wandb.log({dataset_name + "_val": val_loss})
            if "save_model_on_dataset_name" in self.params.keys():
                current = self.val_loss_history[self.params["save_model_on_dataset_name"]][-1]
                if current <= self.best_val_loss[self.params["save_model_on_dataset_name"]]:
                    print("Saving model to", self.params["model_path"])
                    self.model.save_weights(self.params["model_path"]+"_best.pkl")
            # early stopping
            if self.early_stopping_patience > 0:
                if tf.reduce_max(self.val_f1_history[-self.early_stopping_patience:]) < tf.reduce_max(
                    self.val_f1_history):
                    print("Early stopping triggered")
                    self.stop_training = True

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

    def fov_filter(self, dataset, fov_list, fov_key="folder_name"):
        """Filter out training examples that are not in the fov_list and return a copy of the
        dataset
        Args:
            dataset (tf.data.Dataset):
                Dataset to filter
            fov_list (list):
                List of fovs to keep
            fov_key (str):
                Key of the fov in the dataset
        Returns:
            dataset (tf.data.Dataset):
                Filtered dataset
        """

        def predicate(example):
            """Helper function that returns true if the fov is in fov_list
            Args:
                example (dict):
                    Example dictionary
            Returns:
                tf.Tensor:
                    True if the fov is in fov_list
            """
            return tf.reduce_any(tf.equal(example[fov_key], fov_list))
        dataset = dataset.filter(predicate)
        return dataset

    def dset_marker_filter(self, dataset, exclude_dset_marker):
        """Filter out training examples that are in the exclude_dset_marker_dict and return a copy
        of the dataset
        Args:
            dataset (tf.data.Dataset):
                Dataset to filter
            exclude_dset_marker (list):
                List containing dataset and marker pairs to exclude [[d1, d2], [m1, m2]]
        Returns:
            dataset (tf.data.Dataset):
                Filtered dataset
        """
        
        def predicate(dataset, marker):
            """Helper function that returns true if the marker is in marker_list
            Args:
                dataset (tf.Tensor):
                    Dataset name of the example
                marker (tf.Tensor):
                    Marker name of the example
            Returns:
                bool:
                    True if the marker is in marker_list
            """
            dataset = tf.get_static_value(dataset).decode("utf-8")
            marker = tf.get_static_value(marker).decode("utf-8")
            return tf.logical_not(tf.logical_and(
                tf.reduce_any(tf.equal(dataset,
                                       exclude_dset_marker[0])
                ),
                tf.reduce_any(tf.equal(marker,
                                       exclude_dset_marker[1])
                )
            ))

        dataset = dataset.filter(
            lambda example: tf.py_function(
                predicate, [example["dataset"], example["marker"]], tf.bool
            )
        )
        return dataset
    
    def predict_dataset_list(
            self, datasets, save_predictions=True, fname="predictions", ckpt_path=None
        ):
        """Runs predictions on a list of datasets and returns results as a dataframe
        Args:
            datasets (list): List of tf.data.Datasets to run predictions on
            save_predictions (bool): Whether to save the predictions to a file
            fname (str): Name of the file to save the predictions to
            ckpt_path (str): Path to the model checkpoint file
        Returns:
            df (pd.DataFrame): Dataframe containing the predictions
        """
        if ckpt_path:
            self.load_model(ckpt_path)
            print("Loaded model from", ckpt_path)
        elif not hasattr(self, "model") or self.model is None:
            self.load_model(self.params["model_path"])
            print("Loaded model from", self.params["model_path"])
        if isinstance(datasets, tf.data.Dataset):
            datasets = [datasets]

        df_list = []
        for dataset in datasets:
            dataset = dataset.prefetch(8)
            for j, example in enumerate(dataset):
                x_batch, _ = self.prep_batches(example)
                prediction = self.predict(x_batch)
                for i, df in enumerate(example["activity_df"].numpy()):
                    df = pd.read_json(StringIO(df.decode()))
                    cell_ids, mean_per_cell = segment_mean(
                        example["instance_mask"][i:i+1], prediction[i:i+1]
                    )
                    pred_df = pd.DataFrame({"labels": cell_ids, "prediction": mean_per_cell})
                    df = df.merge(pred_df, on="labels", how="left")
                    df["fov"] = [example["folder_name"].numpy()[i].decode()] * len(df)
                    df["dataset"] = [example["dataset"].numpy()[i].decode()] * len(df)
                    df["marker"] = [example["marker"].numpy()[i].decode()] * len(df)
                    df_list.append(df)
        df = pd.concat(df_list)
        df = df[df["labels"] != 0]
        if save_predictions:
            df.to_csv(os.path.join(self.params["model_dir"], fname+".csv"))
        return df


if __name__ == "__main__":
    print("CUDA_VISIBLE_DEVICES: " + str(os.getenv("CUDA_VISIBLE_DEVICES")))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        default="configs/params.toml",
    )
    args = parser.parse_args()
    params = toml.load(args.params)
    trainer = ModelBuilder(params)
    trainer.train()