from deepcell.utils.train_utils import rate_scheduler, get_callbacks, count_gpus
from segmentation_data_prep import parse_dict, feature_description
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import LearningRateScheduler
from deepcell.model_zoo.panopticnet import PanopticNet
from tensorflow.keras.optimizers import SGD, Adam
from semantic_head import create_semantic_head
from model_builder import ModelBuilder
import tensorflow as tf
from time import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import toml
import os


class PromixNaive(ModelBuilder):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.num_gpus = count_gpus()
        self.loss_fn = self.prep_loss()
        self.quantile = self.params["quantile"]
        self.class_wise_loss_quantiles = {}

    def matched_high_confidence_selection_thresholds(self):
        """Returns a dictionary with the thresholds for the high confidence selection"""
        neg_thresh, pos_thresh = self.params["confidence_thresholds"]
        targets = tf.constant(np.array([[0, 1]]).transpose())
        y_pred = tf.constant(np.array([[neg_thresh, pos_thresh]]).transpose())
        loss = self.loss_fn(targets, y_pred)
        return {"negative": loss.numpy()[0], "positive": loss.numpy()[1]}

    def prep_data(self):
        """Prepares training and validation data"""
        # make datasets and splits
        dataset = tf.data.TFRecordDataset(self.params["record_path"])
        dataset = dataset.map(
            lambda x: tf.io.parse_single_example(x, feature_description),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(parse_dict, num_parallel_calls=tf.data.AUTOTUNE)

        # split into train and validation
        self.validation_dataset = dataset.take(self.params["num_validation"])
        self.train_dataset = dataset.skip(self.params["num_validation"])

        # shuffle, batch and prefetch the training data
        self.train_dataset = self.train_dataset.shuffle(self.params["shuffle_buffer_size"]).batch(
            self.params["batch_size"]
        )
        self.train_dataset = self.train_dataset.prefetch(tf.data.AUTOTUNE)

        self.validation_dataset = self.validation_dataset.batch(
            self.params["batch_size"] * np.max([self.num_gpus, 1])
        )

    @staticmethod  # with @tf.function 0.4 s/batch, without 0.15 s/batch on notebook
    def train_step(model, optimizer, loss_fn, aug_fn, loss_mask, x, y_gt):
        """Performs a training step
        Args:
            model (tf.keras.Model): model to train
            optimizer (tf.keras.optimizers.Optimizer): optimizer to use
            loss_fn (tf.keras.losses.Loss): loss function to use
            loss_mask (tf.Tensor): mask to apply to loss
            x (tf.Tensor): input data
            y_gt (tf.Tensor): ground truth labels
        Returns:
            tf.Tensor: loss value
        """
        x_aug, loss_mask_aug, y_gt_aug = aug_fn(x, loss_mask, y_gt)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_img = loss_fn(y_gt, y_pred)
            loss_img *= loss_mask
            loss = tf.reduce_mean(loss_img)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train(self):
        """Calls prep functions and starts training loops"""
        print("Training on", self.num_gpus, "GPUs.")
        # initialize data and model
        self.prep_data()
        self.prep_model()
        self.confidence_loss_thresholds = self.matched_high_confidence_selection_thresholds()
        train_step = self.train_step

        with open(os.path.join(self.params["model_dir"], "params.toml"), "w") as f:
            toml.dump(self.params, f)
        self.summary_writer = tf.summary.create_file_writer(self.params["log_dir"])
        self.step = 0
        self.val_loss_history = []
        self.train_loss_tmp = []
        # train the model
        while self.step < self.params["num_steps"]:
            for batch in tqdm(self.train_dataset):
                # prepare loss mask with unaugmented batches
                x, y = self.prep_batches(batch)
                y_pred = self.model(x, training=False)
                loss_img = self.loss_fn(y, y_pred)
                uniques, loss_per_cell = tf.map_fn(
                    self.reduce_to_cells,
                    (loss_img, batch["instance_mask"]),
                    infer_shape=False,
                    fn_output_signature=[
                        tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0),
                        tf.RaggedTensorSpec(shape=[None], dtype=tf.float32, ragged_rank=0),
                    ],
                )
                batch["activity_df"] = [
                    pd.read_json(df.decode()) for df in batch["activity_df"].numpy()
                ]
                batch["activity_df"] = [
                    df.merge(
                        pd.DataFrame(
                            {"labels": uniques[i].numpy(), "loss": loss_per_cell[i].numpy()}
                        ),
                        on="labels",
                    )
                    for i, df in enumerate(batch["activity_df"])
                ]
                #
                loss_mask = self.batchwise_loss_selection(
                    batch["activity_df"], batch["instance_mask"], batch["marker"]
                )
                loss_mask = tf.cast(loss_mask, tf.float32)
                # augment batches and do train_step

                def aug_fn(x, y, z):
                    return (x, y, z)
                train_loss = train_step(
                    self.model, self.optimizer, self.loss_fn, aug_fn, loss_mask, x, y
                )
                self.train_loss_tmp.append(train_loss)
                self.step += 1
                self.tensorboard_callbacks(x, y)
                if self.step > self.params["num_steps"]:
                    break
                # save loss_mask
                if self.step % self.params["snap_steps"] == 0:
                    with self.summary_writer.as_default():
                        tf.summary.image(
                            "loss_mask", tf.expand_dims(loss_mask, axis=-1), step=self.step
                        )

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def reduce_to_cells(tuple_in):
        """Reduces the prediction to the cell level
        Args:
            tuple_in (tf.Tensor): tuple of (loss_img, instance_mask)
        Returns:
            tuple: tuple of (uniques, mean_per_cell)
                uniques (tf.Tensor): unique cell ids, batch x num_cells 0-padded to 200
                mean_per_cell (tf.Tensor): mean loss per cell, batch x num_cells 0-padded to 200
        """
        pred, instance_mask = tuple_in
        instance_mask_flat = tf.cast(tf.reshape(instance_mask, -1), tf.int32)  # b x (h*w)
        pred_flat = tf.cast(tf.reshape(pred, -1), tf.float32)
        uniques, _ = tf.unique(instance_mask_flat)
        sort_order = tf.argsort(instance_mask_flat)
        instance_mask_flat = tf.gather(instance_mask_flat, sort_order)
        pred_flat = tf.gather(pred_flat, sort_order)
        mean_per_cell = tf.math.segment_mean(pred_flat, instance_mask_flat)
        mean_per_cell = tf.gather(mean_per_cell, uniques)
        return [uniques, mean_per_cell]

    def batchwise_loss_selection(self, activity_df, instance_mask, marker):
        """Selects the cells with the lowest loss for each class and runs
            matched_high_confidence_selection internally
        Args:
            activity_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
            instance_mask (tf.Tensor): instance_masks
        Returns:
            tf.Tensor: loss_mask
        """
        loss_selection = []
        for df, mask, mark in zip(activity_df, instance_mask, marker):
            if df.shape[0] == 0:
                loss_selection.append(tf.squeeze(tf.zeros_like(mask, tf.float32)))
                continue
            mark = mark.numpy().decode()

            positive_df = df[df["activity"] == 1]
            negative_df = df[df["activity"] == 0]
            selected_subset = []
            # loss selection methods
            selected_subset += self.class_wise_loss_selection(positive_df, negative_df, mark)
            selected_subset += self.matched_high_confidence_selection(positive_df, negative_df)
            selected_subset = pd.concat(selected_subset)
            positive_mask = tf.reduce_any(
                tf.equal(mask, np.unique(selected_subset.labels.values)), axis=-1
            )
            loss_selection.append(tf.cast(positive_mask, tf.float32))
        return tf.stack(loss_selection)

    def class_wise_loss_selection(self, positive_df, negative_df, mark):
        """Selects the cells with the lowest loss for each class and runs
        Args:
            positive_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
            negative_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
            mark (str): marker name
        Returns:
            list: list of pd.DataFrame that contain the selected cells
        """
        ema = self.params["ema"]
        selected_subset = []
        # get the quantile for gt=0 / gt=1 separately and store cell labels in selected_subset
        if mark not in self.class_wise_loss_quantiles.keys():
            # add keys to dict if not present and set ema to 1 for initialization
            self.class_wise_loss_quantiles[mark] = {"positive": 1.0, "negative": 1.0}
            ema = 1.0
        if positive_df.shape[0] > 0:
            self.class_wise_loss_quantiles[mark]["positive"] = (
                self.class_wise_loss_quantiles[mark]["positive"] * (1 - ema)
                + np.quantile(positive_df.loss, self.quantile) * ema
            )
            selected_subset.append(
                positive_df[
                    positive_df["loss"] <= self.class_wise_loss_quantiles[mark]["positive"]
                ]
            )
        if negative_df.shape[0] > 0:
            self.class_wise_loss_quantiles[mark]["negative"] = (
                self.class_wise_loss_quantiles[mark]["negative"] * (1 - ema)
                + np.quantile(negative_df.loss, self.quantile) * ema
            )
            selected_subset.append(
                negative_df[
                    negative_df["loss"] <= self.class_wise_loss_quantiles[mark]["negative"]
                ]
            )
        return selected_subset

    def matched_high_confidence_selection(self, positive_df, negative_df):
        """Selects the cells with the highest confidence for each class (negative/positive)
        Args:
            positive_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
                containing only GT positive cells
            negative_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
                containing only GT negative cells
        Returns:
            list: list of pd.DataFrame that contains the selected cells
        """
        selected_subset = []
        if positive_df.shape[0] > 0:
            selected_subset.append(
                positive_df[positive_df.loss < self.confidence_loss_thresholds["positive"]]
            )
        if negative_df.shape[0] > 0:
            selected_subset.append(
                negative_df[negative_df.loss < self.confidence_loss_thresholds["negative"]]
            )
        return selected_subset
