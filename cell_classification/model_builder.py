import os
import argparse
import tensorflow as tf
import toml
from augmentation_pipeline import prepare_tf_aug, py_aug, get_augmentation_pipeline
from post_processing import merge_activity_df, process_to_cells
from segmentation_data_prep import parse_dict, feature_description
from deepcell.model_zoo.panopticnet import PanopticNet
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import CosineDecay
from deepcell.utils.train_utils import rate_scheduler, get_callbacks, count_gpus
from deepcell import losses
from semantic_head import create_semantic_head
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd
from loss import Loss


class ModelBuilder:
    """Builds, trains and writes validation metrics for models"""

    def __init__(self, params):
        """Initialize the trainer with the parameters from the config file
        Args:
            params (dict): Dictionary of parameters from the config file
        """
        self.params = params

    def prep_data(self):
        """Prepares training and validation data"""
        # make datasets and splits
        dataset = tf.data.TFRecordDataset(self.params["record_path"])
        dataset = dataset.map(
            lambda x: tf.io.parse_single_example(x, feature_description),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(parse_dict, num_parallel_calls=tf.data.AUTOTUNE)

        # split into train and validation
        self.validation_dataset = dataset.take(self.params["num_validation"])
        self.train_dataset = dataset.skip(self.params["num_validation"])

        # shuffle, batch and augment the training data
        self.train_dataset = self.train_dataset.shuffle(self.params["shuffle_buffer_size"]).batch(
            self.params["batch_size"]
        )
        augmentation_pipeline = get_augmentation_pipeline(self.params)
        tf_aug = prepare_tf_aug(augmentation_pipeline)
        self.train_dataset = self.train_dataset.map(
            lambda x: py_aug(x, tf_aug),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.map(
            self.prep_batches,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.train_dataset = self.train_dataset.prefetch(tf.data.AUTOTUNE)
        self.validation_dataset = self.validation_dataset.batch(self.params["batch_size"])

    def prep_model(self):
        """Prepares the model for training"""
        # initialize optimizer and lr scheduler
        # replace with AdamW when available
        self.optimizer = Adam(learning_rate=self.params["lr"], clipnorm=0.001)
        self.lr_sched = LearningRateScheduler(CosineDecay(
            initial_learning_rate=self.params["lr"],
            decay_steps=self.params["num_epochs"]*self.params["steps_per_epoch"], alpha=1e-6
            )
        )
        # initialize model
        if "test" in self.params.keys() and self.params["test"]:
            self.model = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(
                    1, (3, 3), input_shape=self.params["input_shape"], name="semantic_head",
                    activation="sigmoid", padding="same", data_format="channels_last"
                )]
            )
        else:
            self.model = PanopticNet(
                backbone=self.params["backbone"], input_shape=self.params["input_shape"],
                norm_method="std", num_semantic_classes=self.params["classes"],
                create_semantic_head=create_semantic_head
            )

        loss = {}
        # Give losses for all of the semantic heads
        for layer in self.model.layers:
            if layer.name.startswith("semantic_"):
                loss[layer.name] = self.prep_loss()

        # prepare folders
        self.params['model_dir'] = os.path.join(
            os.path.normpath(self.params["path"]), self.params["experiment"]
        )
        self.params['log_dir'] = os.path.join(self.params['model_dir'], "logs")
        os.makedirs(self.params['model_dir'], exist_ok=True)
        os.makedirs(self.params['log_dir'], exist_ok=True)
        if "model_path" not in self.params.keys() or self.params["model_path"] is None:
            self.params['model_path'] = os.path.join(
                self.params['model_dir'], "{}.h5".format(self.params["experiment"])
            )
        self.params['loss_path'] = os.path.join(
            self.params['model_dir'], "{}.npz".format(self.params["experiment"])
        )
        self.num_gpus = count_gpus()
        print("Training on", self.num_gpus, "GPUs.")

        self.train_callbacks = get_callbacks(
            self.params['model_path'],
            tensorboard_log_dir=self.params['log_dir'],
            save_weights_only=self.num_gpus >= 2,
            monitor="val_loss",
            verbose=1,
        )
        self.train_callbacks.append(self.lr_sched)
        if "weight_decay" in self.params.keys():
            self.add_weight_decay()
        self.model.compile(loss=loss, optimizer=self.optimizer)

    def train(self):
        """Calls prep functions and starts training loops"""
        self.prep_data()
        self.prep_model()
        #
        validation_dataset = self.validation_dataset.map(
                self.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        )

        with open(os.path.join(self.params['model_dir'], "params.toml"), "w") as f:
            toml.dump(self.params, f)
        self.loss_history = self.model.fit(
            self.train_dataset,
            epochs=self.params["num_epochs"],
            validation_data=validation_dataset,
            callbacks=self.train_callbacks,
        )

    def prep_loss(self):
        """Prepares the loss function for the model
        Args:
            n_classes (int): Number of semantic classes in the dataset
        Returns:
            loss_fn (function): Loss function for the model
        """
        loss_fn = Loss(
            self.params["loss_fn"], self.params["loss_selective_masking"],
            **self.params["loss_kwargs"]
        )
        return loss_fn

    @staticmethod
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
            [batch["mplex_img"], tf.cast(batch["binary_mask"], tf.float32)], axis=-1
        )
        targets = batch["marker_activity_mask"]
        return inputs, targets

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
        val_dset = val_dset.map(
                self.prep_batches, num_parallel_calls=tf.data.AUTOTUNE
        )
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
            if isinstance(layer, tf.keras.layers.Conv2D) or \
                    isinstance(layer, tf.keras.layers.Dense):
                layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.bias))


if __name__ == "__main__":
    print('CUDA_VISIBLE_DEVICES: ' + str(os.getenv("CUDA_VISIBLE_DEVICES")))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params", type=str, default="cell_classification/configs/params.toml",
    )
    args = parser.parse_args()
    params = toml.load(args.params)
    trainer = ModelBuilder(params)
    trainer.train()
