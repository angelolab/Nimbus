import h5py
import toml
import argparse
import pickle
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
from model_builder import ModelBuilder
import numpy as np
import toml
from copy import deepcopy
from joblib import Parallel, delayed
import pandas as pd
from promix_naive import PromixNaive


def load_model_and_val_data(params):
    """Load model and validation data from params dict
    Args:
        params (dict):
            dictionary containing model and validation data paths
    Returns:
        model (ModelBuilder):
            trained model
        val_data (tf.data.Dataset):
            validation dataset
    """
    params["eval"] = True
    if params["model"] == "ModelBuilder":
        model = ModelBuilder(params)
    elif params["model"] == "PromixNaive":
        model = PromixNaive(params)
    model.prep_data()
    model.load_model(params["model_path"])
    val_dset = model.validation_dataset
    return model, val_dset


def calc_roc(pred_list, gt_key="marker_activity_mask", pred_key="prediction", cell_level=False):
    """Calculate ROC curve
    Args:
        pred_list (list):
            list of samples with predictions
        gt_key (str):
            key for ground truth labels
        pred_key (str):
            key for predictions
    Returns:
        roc (dict):
            dictionary containing ROC curve data
    """
    roc = {"fpr": [], "tpr": [], "thresholds": [], "auc": []}
    for sample in pred_list:
        if cell_level:
            df = sample["activity_df"]
            gt = df[gt_key].to_numpy()
            pred = df[pred_key].to_numpy()
        else:
            foreground = sample["binary_mask"] > 0
            gt = sample[gt_key][foreground].flatten()
            pred = sample[pred_key][foreground].flatten()
        if gt.size > 0 and gt.min() == 0 and gt.max() > 0:  # roc is only defined for this interval
            fpr, tpr, thresholds = roc_curve(gt, pred)
            roc["fpr"].append(fpr)
            roc["tpr"].append(tpr)
            roc["thresholds"].append(thresholds)
            roc["auc"].append(auc(fpr, tpr))
    return roc


def calc_scores(gt, pred, threshold):
    """Calculate scores for a given threshold
    Args:
        gt (np.array):
            ground truth labels
        pred (np.array):
            predictions
        threshold (float):
            threshold for predictions
    Returns:
        scores (dict):
            dictionary containing scores
    """
    # exclude masked out regions from metric calculation
    pred = pred[gt < 2]
    gt = gt[gt < 2]
    tn, fp, fn, tp = confusion_matrix(
        y_true=gt, y_pred=(pred >= threshold).astype(int), labels=[0, 1]
    ).ravel()
    metrics = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "f1_score": 2 * tp / (2 * tp + fp + fn + 1e-8),
    }
    return metrics


def calc_metrics(
    pred_list, gt_key="marker_activity_mask", pred_key="prediction", cell_level=False
):
    """Calculate metrics
    Args:
        pred_list (list):
            list of samples with predictions
        gt_key (str):
            key of ground truth in pred_list
        pred_key (str):
            key of prediction in pred_list
    Returns:
        avg_metrics (dict):
            dictionary containing metrics averaged over all samples
    """
    metrics_dict = {
        "accuracy": [], "precision": [], "recall": [], "specificity": [], "f1_score": [], "tp": [],
        "tn": [], "fp": [], "fn": [],
    }

    def _calc_metrics(threshold):
        """Helper function to calculate metrics for a given threshold in parallel"""
        metrics = deepcopy(metrics_dict)

        for sample in pred_list:
            if cell_level:
                df = sample["activity_df"]
                gt = np.array(df[gt_key])
                pred = np.array(df[pred_key])
            else:
                foreground = sample["binary_mask"] > 0
                gt = sample[gt_key][foreground].flatten()
                pred = sample[pred_key][foreground].flatten()
            if gt.size == 0:
                continue
            scores = calc_scores(gt, pred, threshold)

            # only add specificity for samples that have no positives
            if np.sum(gt) == 0:
                keys = ["specificity"]
            else:
                keys = scores.keys()
            for key in keys:
                metrics[key].append(scores[key])
            metrics["threshold"] = threshold
        for key in ["dataset", "imaging_platform", "marker"]:
            metrics[key] = sample[key]
        return metrics

    # calculate metrics for all thresholds in parallel
    thresholds = np.linspace(0.01, 1, 50)
    # metric_list = Parallel(n_jobs=8)(delayed(_calc_metrics)(i) for i in thresholds)
    metric_list = [_calc_metrics(i) for i in thresholds]
    # reduce metrics over all samples for each threshold
    avg_metrics = deepcopy(metrics_dict)
    for key in ["dataset", "imaging_platform", "marker", "threshold"]:
        avg_metrics[key] = []
    for metrics in metric_list:
        for key in ["accuracy", "precision", "recall", "specificity", "f1_score"]:
            avg_metrics[key].append(np.mean(metrics[key]))
        for key in ["tp", "tn", "fp", "fn"]:  # sum fn, fp, tn, tp
            avg_metrics[key].append(np.sum(metrics[key]))
        for key in ["dataset", "imaging_platform", "marker", "threshold"]:  # copy strings
            avg_metrics[key].append(metrics[key])
    return avg_metrics


def average_roc(roc_list):
    """Average ROC curves
    Args:
        roc_list (list):
            list of ROC curves
    Returns:
        tprs (np.array):
            standardized true positive rates for each sample
        mean_tprs (np.array):
            mean true positive rates over all samples
        std np.array:
            standard deviation of true positive rates over all samples
        base (np.array):
            fpr values for interpolation
        mean_thresh (np.array):
            mean of the threshold values over all samples
    """
    base = np.linspace(0, 1, 101)
    tpr_list = []
    thresh_list = []
    for i in range(len(roc_list["tpr"])):
        tpr_list.append(np.interp(base, roc_list["fpr"][i], roc_list["tpr"][i]))
        thresh_list.append(np.interp(base, roc_list["tpr"][i], roc_list["thresholds"][i]))

    tprs = np.array(tpr_list)
    thresh_list = np.array(thresh_list)
    mean_thresh = np.mean(thresh_list, axis=0)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    return tprs, mean_tprs, base, std, mean_thresh


class HDF5Loader(object):
    """HDF5 iterator for loading data from HDF5 files"""

    def __init__(self, folder):
        """Initialize HDF5 generator
        Args:
            folder (str):
                path to folder containing HDF5 files
        """
        self.folder = folder
        self.files = os.listdir(folder)
        # filter out hdf files
        self.files = [os.path.join(folder, f) for f in self.files if f.endswith(".hdf")]
        self.file_idx = 0

    def __len__(self):
        return len(self.files)

    def load_hdf(self, file):
        """Load HDF5 file
        Args:
            file (str):
                path to HDF5 file
        Returns:
            data (dict):
                dictionary containing data from HDF5 file
        """
        out_dict = {}
        with h5py.File(file, "r") as f:
            keys = [key for key in f.keys() if key != "activity_df"]
            for key in keys:
                if isinstance(f[key][()], bytes):
                    out_dict[key] = f[key][()].decode("utf-8")
                else:
                    out_dict[key] = f[key][()]
            out_dict["activity_df"] = pd.read_json(f["activity_df"][()].decode())
        return out_dict

    def __iter__(self):
        self.file_idx = 0
        return self

    def __next__(self):
        if self.file_idx >= len(self.files):
            raise StopIteration
        else:
            self.file_idx += 1
            return self.load_hdf(self.files[self.file_idx - 1])
