import h5py
import toml
import argparse
import pickle
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
from model_builder import ModelBuilder
import numpy as np
from tqdm import tqdm
from plot_utils import plot_average_roc, plot_metrics_against_threshold
import toml
from copy import deepcopy
from joblib import Parallel, delayed
import pandas as pd
import tables


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
    model = ModelBuilder(params)
    model.prep_data()
    model.load_model(params["model_path"])
    val_dset = iter(model.validation_dataset)
    return model, val_dset


def process_to_cells(sample):
    """Process predictions from pixel level to cell level
    Args:
        sample (dict):
            single sample with predictions
    Returns:
        sample (dict):
            single sample with predictions at cell level
    """
    if not isinstance(sample["activity_df"], pd.DataFrame):
        if isinstance(sample["activity_df"], str):
            df = pd.read_json(sample["activity_df"])
        else:
            df = pd.read_hdf(sample["activity_df"])
    df["pred_activity"] = 0.0
    unique_labels = np.unique(sample["instance_mask"])
    unique_labels = unique_labels[unique_labels != 0]
    mean_per_cell_mask = np.zeros_like(sample["instance_mask"], dtype=np.float32)
    for unique_label in unique_labels:
        mask = sample["instance_mask"] == unique_label
        mean_pred = sample["prediction"][mask].mean()
        df.at[df.index[df["labels"] == unique_label].to_list().pop(), "pred_activity"] = mean_pred
        mean_per_cell_mask[mask] = mean_pred
    sample["activity_df"] = df
    sample["prediction_mean"] = mean_per_cell_mask
    return sample


def predict(model, val_dset, params, return_pred=False):
    """Predict labels for validation data
    Args:
        model (ModelBuilder):
            trained model
        val_dset (tf.data.Dataset):
            validation data
        params (dict):
            dictionary containing model and validation data
        return_pred (bool):
            whether to return predictions
    Returns:
        predictions (list) optional:
            list of dictionaries containing the full example + prediction
        params (dict):
            dictionary containing model hyperparams and validation data paths
    """
    # prepare output folder
    params["eval_dir"] = os.path.join(model.params["model_dir"], "eval")
    os.makedirs(params["eval_dir"], exist_ok=True)

    single_example_list = []
    j = 0
    for sample in tqdm(val_dset):
        sample["prediction"] = model.predict(model.prep_batches(sample)[0])

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
            # save single example
            single_example = process_to_cells(single_example)
            fname = os.path.join(params["eval_dir"], str(j).zfill(4) + "_pred.hdf")
            j += 1
            with h5py.File(fname, "w") as f:
                for key in [key for key in single_example.keys() if key != "activity_df"]:
                    f.create_dataset(key, data=single_example[key])
            single_example["activity_df"].to_hdf(fname, key="activity_df", mode="a")
            if return_pred:
                single_example_list.append(single_example)
    # save params to toml file
    with open(os.path.join(params["model_dir"], "params.toml"), "w") as f:
        toml.dump(params, f)
    if return_pred:
        return single_example_list, params
    else:
        return params


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
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": [],
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
            tn, fp, fn, tp = confusion_matrix(
                y_true=gt.clip(0, 1), y_pred=(pred >= threshold).astype(int), labels=[0, 1]
            ).ravel()
            metrics["tp"].append(tp)
            metrics["tn"].append(tn)
            metrics["fp"].append(fp)
            metrics["fn"].append(fn)
            metrics["accuracy"].append((tp + tn) / (tp + tn + fp + fn + 1e-8))
            metrics["precision"].append(tp / (tp + fp + 1e-8))
            metrics["recall"].append(tp / (tp + fn + 1e-8))
            metrics["f1_score"].append(2 * tp / (2 * tp + fp + fn + 1e-8))
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
        for key in ["accuracy", "precision", "recall", "f1_score"]:  # averade metrics
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
        out_dict["activity_df"] = pd.read_hdf(file, key="activity_df", mode="r")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model weights",
        default="C:\\Users\\lorenz\\Desktop\\angelo_lab\\cell_classification\\checkpoints\\" +
        "ex_1.h5",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to model params",
        default="C:\\Users\\lorenz\\Desktop\\angelo_lab\\cell_classification\\"
        + "cell_classification\\configs\\params.toml",
    )
    args = parser.parse_args()
    with open(args.params_path, "r") as f:
        params = toml.load(f)
    params["model_path"] = args.model_path
    model, val_dset = load_model_and_val_data(params)
    pred_list, params = predict(model, val_dset, params, True)

    # pixel level evaluation
    print("Calculate ROC curve")
    roc = calc_roc(pred_list)
    with open(os.path.join(params["eval_dir"], "roc.pkl"), "wb") as f:
        pickle.dump(roc, f)
    tprs, mean_tprs, fpr, std, mean_thresh = average_roc(roc)
    plot_average_roc(mean_tprs, std, save_dir=params["eval_dir"], save_file="avg_roc.png")
    print("AUC: {}".format(np.mean(roc["auc"])))

    print("Calculate precision, recall, f1_score and accuracy")
    avg_metrics = calc_metrics(pred_list)
    pd.DataFrame(avg_metrics).to_csv(
        os.path.join(params["eval_dir"], "pixel_metrics.csv"), index=False
    )
    plot_metrics_against_threshold(
        avg_metrics,
        metric_keys=["precision", "recall", "f1_score"],
        threshold_key="threshold",
        save_dir=params["eval_dir"],
        save_file="precision_recall_f1.png",
    )

    # cell level evaluation
    roc = calc_roc(pred_list, gt_key="activity", pred_key="pred_activity", cell_level=True)
    with open(os.path.join(params["eval_dir"], "roc_cell_lvl.pkl"), "wb") as f:
        pickle.dump(roc, f)
    tprs, mean_tprs, fpr, std, mean_thresh = average_roc(roc)
    plot_average_roc(mean_tprs, std, save_dir=params["eval_dir"], save_file="avg_roc_cell_lvl.png")
    print("AUC: {}".format(np.mean(roc["auc"])))

    print("Calculate precision, recall, f1_score and accuracy on the cell level")
    avg_metrics = calc_metrics(
        pred_list, gt_key="activity", pred_key="pred_activity", cell_level=True
    )
    pd.DataFrame(avg_metrics).to_csv(
        os.path.join(params["eval_dir"], "cell_metrics.csv"), index=False
    )
    plot_metrics_against_threshold(
        avg_metrics,
        metric_keys=["precision", "recall", "f1_score"],
        threshold_key="threshold",
        save_dir=params["eval_dir"],
        save_file="precision_recall_f1_cell_lvl.png",
    )
