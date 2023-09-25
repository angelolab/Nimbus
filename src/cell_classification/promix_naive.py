import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import toml
from deepcell.utils import count_gpus
from tqdm import tqdm

from cell_classification.augmentation_pipeline import MixUp, prepare_keras_aug
from cell_classification.model_builder import ModelBuilder


class PromixNaive(ModelBuilder):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.params["model"] = "PromixNaive"
        self.num_gpus = count_gpus()
        self.loss_fn = self.prep_loss()
        self.quantile = self.params["quantile"]
        self.quantile_start = self.params["quantile"]
        self.quantile_end = self.params["quantile_end"]
        self.quantile_warmup_steps = self.params["quantile_warmup_steps"]
        self.class_wise_loss_quantiles = {}
        self.aug_fn = prepare_keras_aug(params, dtype=(tf.float32, tf.uint8))
        self.mixup_fn = MixUp(prob=params["mixup_prob"], alpha=params["mixup_alpha"])
        if "batch_constituents" in list(self.params.keys()):
            self.prep_batches_promix = self.gen_prep_batches_promix_fn(
                self.params["batch_constituents"]
            )
        else:
            self.prep_batches_promix = self.gen_prep_batches_promix_fn()
        # make prep_batches_promix a callable static method
        self.prep_batches_promix = staticmethod(self.prep_batches_promix).__func__

    def quantile_scheduler(self, step):
        """Linear scheduler for quantile
        Args:
            step (int): current step
        """
        return np.min([(
                    self.quantile_start
                    + ((self.quantile_end - self.quantile_start) * step)
                    / self.quantile_warmup_steps
                ), self.quantile_end])

    def gen_prep_batches_promix_fn(self, keys=["mplex_img", "binary_mask"]):
        """Generates a function that preprocesses batches for training. This function needs to
        coexist with ModelBuilder.gen_prep_batches_fn and does not replace it.
        Args:
            keys (list): List of keys to concatenate into a single batch
        Returns:
            prep_batches (function): Function that preprocesses batches for training
        """
        # remove binary_mask from keys, because it'll be loaded anyways
        keys = [key for key in keys if key != "binary_mask"]

        def prep_batches_promix(batch):
            """Preprocess batches for training
            Args:
                batch (dict):
                    Dictionary of tensors and strings containing data from a single batch
            Returns:
                mplex_img (tf.Tensor):
                    Batch of mplex images
                binary_mask (tf.Tensor)
                    Batch of binary mask images
                targets (tf.Tensor):
                    Batch of labels
            """
            targets = batch["marker_activity_mask"]
            mplex_img = tf.concat([batch[key] for key in keys], axis=-1)
            binary_mask = tf.cast(batch["binary_mask"], tf.float32)
            return mplex_img, binary_mask, targets

        return prep_batches_promix

    def matched_high_confidence_selection_thresholds(self):
        """Returns a dictionary with the thresholds for the high confidence selection"""
        neg_thresh, pos_thresh = self.params["confidence_thresholds"]
        targets = tf.constant(np.array([[0, 1]]).transpose())
        y_pred = tf.constant(np.array([[neg_thresh, pos_thresh]]).transpose())
        loss = self.loss_fn(targets, y_pred)
        self.confidence_loss_thresholds = {"positive": loss[0].numpy(), "negative": loss[1].numpy()}

    @staticmethod  # with @tf.function 0.4 s/batch, without 0.15 s/batch on notebook
    def train_step(model, optimizer, loss_fn, aug_fn, mixup_fn, loss_mask, x_mplex, x_binary, y):
        """Performs a training step
        Args:
            model (tf.keras.Model): model to train
            optimizer (tf.keras.optimizers.Optimizer): optimizer to use
            loss_fn (tf.keras.losses.Loss): loss function to use
            loss_mask (tf.Tensor): mask to apply to loss
            x_mplex (tf.Tensor): input mplex image
            x_binary (tf.Tensor): input binary cell mask
            y (tf.Tensor): ground truth labels
        Returns:
            tf.Tensor: loss value
        """
        # run data through augmentation, floats [x_mplex] and ints [loss_mask, y, x_binary]
        # separately to use correct interpolation [bilinear, nearest]
        loss_mask = tf.expand_dims(tf.cast(loss_mask, tf.uint8), -1)
        x_binary = tf.cast(x_binary, tf.uint8)
        x_mplex_aug, cat = aug_fn(x_mplex, tf.concat([loss_mask, y, x_binary], axis=-1))
        loss_mask_aug, y_aug, x_binary_aug = tf.split(
            cat, [loss_mask.shape[-1], y.shape[-1], x_binary.shape[-1]], axis=-1
        )
        # add unspecific mask from y_aug and background from x_binary_aug to loss_mask and set
        # y_aug to be in range [0,1] before mixup
        loss_mask_aug = tf.where(y_aug == 2, 0, loss_mask_aug)
        y_aug = tf.clip_by_value(y_aug, 0, 1)
        # run mixup
        if mixup_fn.prob > 0.0:
            x_mplex_aug_mix, x_binary_aug_mix, y_aug_mix, loss_mask_aug_mix = mixup_fn(
                x_mplex_aug, x_binary_aug, y_aug, loss_mask_aug
            )
        else:
            x_mplex_aug_mix, x_binary_aug_mix, y_aug_mix, loss_mask_aug_mix = (
                x_mplex_aug, x_binary_aug, y_aug, loss_mask_aug,
            )
            loss_mask_aug_mix = tf.cast(loss_mask_aug_mix, tf.float32)
            x_binary_aug_mix = tf.cast(x_binary_aug_mix, tf.float32)
        background = tf.cast(tf.where(x_binary_aug_mix == 0, 1, 0), tf.float32)
        # add background pixels to loss mask and prepare model input x_aug_mix
        loss_mask_aug_mix += background
        x_aug_mix = tf.concat([x_mplex_aug_mix, x_binary_aug_mix], axis=-1)
        with tf.GradientTape() as tape:
            y_pred = model(x_aug_mix, training=True)
            loss_img = loss_fn(y_aug_mix, y_pred)
            loss_img *= tf.squeeze(loss_mask_aug_mix, axis=-1)
            loss = tf.reduce_mean(loss_img)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, x_aug_mix, y_aug_mix, y_pred, loss_img, loss_mask_aug_mix

    def train(self):
        """Calls prep functions and starts training loops"""
        print("Training on", self.num_gpus, "GPUs.")
        # initialize data and model
        self.prep_data()
        self.prep_model()
        self.matched_high_confidence_selection_thresholds()
        train_step = self.train_step

        # make transformations on the training dataset
        self.train_dataset = self.train_dataset.prefetch(tf.data.AUTOTUNE)

        with open(os.path.join(self.params["model_dir"], "params.toml"), "w") as f:
            toml.dump(self.params, f)
        self.summary_writer = tf.summary.create_file_writer(self.params["log_dir"])
        self.step = 0
        self.global_val_loss = []
        self.val_loss_history = {}
        self.train_loss_tmp = []
        # train the model
        while self.step < self.params["num_steps"]:
            for batch in tqdm(self.train_dataset):
                # prepare loss mask with unaugmented batches
                x_mplex, x_binary, y = self.prep_batches_promix(batch)
                x = tf.concat([x_mplex, x_binary], axis=-1)
                y_pred = self.model(x, training=False)
                loss_img = self.loss_fn(y, y_pred)
                uniques, loss_per_cell = tf.map_fn(
                    self.reduce_to_cells,
                    (loss_img, batch["instance_mask"]),
                    infer_shape=False,
                    parallel_iterations=4,
                    fn_output_signature=[
                        tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0),
                        tf.RaggedTensorSpec(shape=[None], dtype=tf.float32, ragged_rank=0),
                    ],
                )
                batch["activity_df"] = [
                    pd.read_json(df.decode()) for df in tf.get_static_value(batch["activity_df"])
                ]
                batch["activity_df"] = [
                    df.merge(
                        pd.DataFrame({
                            "labels": tf.get_static_value(uniques[i]),
                            "loss": tf.get_static_value(loss_per_cell[i])
                        }),
                        on="labels",
                    )
                    for i, df in enumerate(batch["activity_df"])
                ]
                #
                loss_mask = self.batchwise_loss_selection(
                    batch["activity_df"], batch["instance_mask"], batch["marker"], batch["dataset"]
                )
                loss_mask *= tf.cast(tf.squeeze(batch["binary_mask"], -1), tf.float32)
                # augment batches and do train_step
                train_loss, x_aug, y_gt_aug, _, _, loss_mask_aug = train_step(
                    self.model, self.optimizer, self.loss_fn, self.aug_fn, self.mixup_fn,
                    loss_mask, x_mplex, x_binary, y
                )
                self.train_loss_tmp.append(train_loss)
                self.step += 1
                self.quantile = self.quantile_scheduler(self.step)
                self.tensorboard_callbacks(x_aug, y_gt_aug)
                if self.step > self.params["num_steps"]:
                    break
                # custom tensorboard callbacks
                if self.step % self.params["snap_steps"] == 0:
                    with self.summary_writer.as_default():
                        tf.summary.text("marker", batch["marker"][0], step=self.step)
                        tf.summary.image(
                            "loss_mask",
                            tf.cast(loss_mask_aug[:1, ...], tf.float32)
                            - tf.math.abs(x_aug[:1, ..., 1:2] * -1) * 0.25,
                            step=self.step,
                        )
                        for key in list(self.class_wise_loss_quantiles.keys()):
                            for class_ in ["positive", "negative"]:
                                tf.summary.scalar(
                                    key + "_" + class_[:3],
                                    self.class_wise_loss_quantiles[key][class_],
                                    step=self.step,
                                )
                        tf.summary.scalar("quantile_thresh", self.quantile, step=self.step)
                    # save self.class_wise_loss_quantiles as toml
                    with open(
                        os.path.join(self.params["log_dir"], "loss_quantiles.toml"), "w"
                    ) as f:
                        toml.dump(self.class_wise_loss_quantiles, f)
                if self.step % self.params["val_steps"] == 0:
                    model_fname = os.path.join(
                        self.params["model_dir"], "checkpoint_{}.h5".format(self.step)
                    )
                    print("Saving model to", model_fname)
                    self.model.save_weights(model_fname)

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

    def batchwise_loss_selection(self, activity_df, instance_mask, marker, dataset):
        """Selects the cells with the lowest loss for each class and runs
            matched_high_confidence_selection internally
        Args:
            activity_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
            instance_mask (tf.Tensor): instance_masks
            marker list(tf.Tensor): list of markers
            dataset list(tf.Tensor): dataset name
        Returns:
            tf.Tensor: loss_mask that has ones for every pixel that is selected for loss
            calculation and zeros for the background and all not selected cells
        """
        loss_selection = []
        for df, mask, mark, dset in zip(activity_df, instance_mask, marker, dataset):
            if df.shape[0] == 0:
                loss_selection.append(tf.squeeze(tf.zeros_like(mask, tf.float32)))
                continue
            mark = tf.get_static_value(mark).decode()
            dset = tf.get_static_value(dset).decode()

            positive_df = df[df["activity"] == 1]
            negative_df = df[df["activity"] == 0]
            selected_subset = []
            # loss selection methods
            selected_subset += self.class_wise_loss_selection(positive_df, negative_df, mark, dset)
            selected_subset += self.matched_high_confidence_selection(positive_df, negative_df)
            if selected_subset:
                selected_subset = pd.concat(selected_subset)
            else:
                selected_subset = pd.DataFrame(columns=["labels"])
            positive_mask = tf.reduce_any(
                tf.equal(mask, np.unique(selected_subset.labels.values)), axis=-1
            )
            loss_selection.append(tf.cast(positive_mask, tf.float32))
        return tf.stack(loss_selection)

    def class_wise_loss_selection(self, positive_df, negative_df, marker, dataset):
        """Selects the cells with the lowest loss for each class
        Args:
            positive_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
            negative_df (pd.DataFrame): dataframe with columns "labels", "activity" and "loss"
            marker (str): marker name
            dataset (str): dataset name
        Returns:
            list: list of pd.DataFrame that contain the selected cells
        """
        ema = self.params["ema"]
        selected_subset = []
        dataset_marker = dataset + "_" + marker
        # get the quantile for gt=0 / gt=1 separately and store cell labels in selected_subset
        if dataset_marker not in self.class_wise_loss_quantiles.keys():
            # add keys to dict if not present and set ema to 1 for initialization
            self.class_wise_loss_quantiles[dataset_marker] = {"positive": 1.0, "negative": 1.0}
            ema = 1.0
        if positive_df.shape[0] > 0:
            self.class_wise_loss_quantiles[dataset_marker]["positive"] = (
                self.class_wise_loss_quantiles[dataset_marker]["positive"] * (1 - ema)
                + np.quantile(positive_df.loss, self.quantile) * ema
            )
            selected_subset.append(
                positive_df[
                    positive_df["loss"] <=
                    self.class_wise_loss_quantiles[dataset_marker]["positive"]
                ]
            )
        if negative_df.shape[0] > 0:
            self.class_wise_loss_quantiles[dataset_marker]["negative"] = (
                self.class_wise_loss_quantiles[dataset_marker]["negative"] * (1 - ema)
                + np.quantile(negative_df.loss, self.quantile) * ema
            )
            selected_subset.append(
                negative_df[
                    negative_df["loss"] <=
                    self.class_wise_loss_quantiles[dataset_marker]["negative"]
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
                positive_df[positive_df.loss.values < self.confidence_loss_thresholds["positive"]]
            )
        if negative_df.shape[0] > 0:
            selected_subset.append(
                negative_df[negative_df.loss.values < self.confidence_loss_thresholds["negative"]]
            )
        return selected_subset


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
    trainer = PromixNaive(params)
    if "load_model" in params.keys() and params["load_model"]:
        trainer.load_model(params["load_model"])
    trainer.train()
