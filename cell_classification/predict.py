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


def predict(model, val_dset, params):
    """Predict labels for validation data
    Args:
        model (ModelBuilder):
            trained model
        val_dset (tf.data.Dataset):
            validation data
        params (dict):
            dictionary containing model and validation data paths
    Returns:
        predictions (list):
            list of dictionaries containing the full example + prediction
        params (dict):
            dictionary containing model hyperparams and validation data paths
    """
    pred_list = []
    for sample in tqdm(val_dset):
        sample["prediction"] = model.predict(model.prep_batches(sample)[0])
        pred_list.append(sample)

    # split batches to single samples
    single_example_list = []
    for sample in pred_list:
        for key in sample.keys():
            sample[key] = np.split(sample[key], sample[key].shape[0])
        for i in range(len(sample["prediction"])):
            single_example = {}
            for key in sample.keys():
                single_example[key] = np.squeeze(sample[key][i], axis=0)
                if single_example[key].dtype == object:
                    single_example[key] = sample[key][i].item().decode("utf-8")
            single_example_list.append(single_example)
    # save pred_list to pickle file
    params["eval_dir"] = os.path.join(model.params["model_dir"], "eval")
    os.makedirs(params["eval_dir"], exist_ok=True)
    with open(os.path.join(params["eval_dir"], "pred_list.pkl"), "wb") as f:
        pickle.dump(single_example_list, f)

    # save params to toml file
    with open(os.path.join(params["model_dir"], "params.toml"), "w") as f:
        toml.dump(params, f)
    return single_example_list, params


def calc_roc(pred_list, gt_key="marker_activity_mask", pred_key="prediction"):
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
        if sample[gt_key].max() == 0:
            continue
        fpr, tpr, thresholds = roc_curve(sample[gt_key].flatten(), sample[pred_key].flatten())
        roc["fpr"].append(fpr)
        roc["tpr"].append(tpr)
        roc["thresholds"].append(thresholds)
        roc["auc"].append(auc(fpr, tpr))
    return roc


def calc_metrics(pred_list, gt_key="marker_activity_mask", pred_key="prediction"):
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
        """Calculate metrics for a given threshold"""
        metrics = deepcopy(metrics_dict)
        for sample in pred_list:
            if sample[gt_key].max() == 0:
                continue
            tn, fp, fn, tp = confusion_matrix(
                y_true=sample[gt_key].flatten(),
                y_pred=(sample[pred_key].flatten() >= threshold).astype(int),
            ).ravel()
            metrics["tp"].append(tp)
            metrics["tn"].append(tn)
            metrics["fp"].append(fp)
            metrics["fn"].append(fn)
            metrics["accuracy"].append((tp + tn) / (tp + tn + fp + fn))
            metrics["precision"].append(tp / (tp + fp + 1e-8))
            metrics["recall"].append(tp / (tp + fn + 1e-8))
            metrics["f1_score"].append(2 * tp / (2 * tp + fp + fn))
        metrics["threshold"] = threshold
        for key in ["dataset", "imaging_platform", "marker"]:
            metrics[key] = sample[key]
        return metrics

    thresholds = np.linspace(0, 1, 101)
    metric_list = Parallel(n_jobs=8)(delayed(_calc_metrics)(i) for i in thresholds)
    avg_metrics = deepcopy(metrics_dict)
    for key in ["dataset", "imaging_platform", "marker", "threshold"]:
        avg_metrics[key] = []
    for metrics in metric_list:
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            avg_metrics[key].append(np.mean(metrics[key]))
        for key in ["tp", "tn", "fp", "fn"]:
            avg_metrics[key].append(np.sum(metrics[key]))
        for key in ["dataset", "imaging_platform", "marker", "threshold"]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model weights",
        default="C:\\Users\\lorenz\\Desktop\\angelo_lab\\cell_classification\\checkpoints\\ex_1.h5"
    )
    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to model params",
        default="C:\\Users\\lorenz\\Desktop\\angelo_lab\\cell_classification\\" +
        "cell_classification\\configs\\params.toml",
    )
    args = parser.parse_args()
    with open(args.params_path, "r") as f:
        params = toml.load(f)
    params["model_path"] = args.model_path
    model, val_dset = load_model_and_val_data(params)
    pred_list, params = predict(model, val_dset, params)

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
        avg_metrics, metric_keys=["precision", "recall", "f1_score"], threshold_key="threshold",
        save_dir=params["eval_dir"], ssave_file="precision_recall_f1.png",
    )
