from ast import arg
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from segmentation_data_prep import parse_dict, feature_description
import tensorflow as tf
from skimage.segmentation import find_boundaries
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from metrics import calc_scores
import seaborn as sns


def segmentation_to_boundaries(labels):
    """Convert a segmentation to a binary mask of the boundaries

    Args:
        segmentation (np.ndarray):
            A 2D array of integers representing a segmentation

    Returns:
        np.ndarray:
            A 2D array of booleans representing the boundaries of the segmentation
    """
    boundaries = find_boundaries(labels, mode="inner")
    boundaries = labels * boundaries.astype(labels.dtype)
    return boundaries


def plot_overlay(example, save_dir=None, save_file=None, dpi=160):
    """Plot the marker image and the marker activity segmentation overlayed

    Args:
        example (dict):
            Dictionary with keys "mplex_img", "marker_activity_mask"
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """
    marker_activity_mask = example["marker_activity_mask"].numpy()
    positives = marker_activity_mask == 1
    undetermined = marker_activity_mask == 2
    positive_boundaries = segmentation_to_boundaries(positives)
    undetermined_boundaries = segmentation_to_boundaries(undetermined)
    binary_mask = example["binary_mask"].numpy()
    negative_boundaries = segmentation_to_boundaries(binary_mask) > 0
    img = np.repeat(example["mplex_img"], 3, axis=-1)
    img[np.squeeze(negative_boundaries)] = (1, 0, 0)
    img[np.squeeze(positive_boundaries)] = [0, 1, 0]
    img[np.squeeze(undetermined_boundaries)] = [0, 0, 1]
    #
    ax = plt.subplot(111)
    pos = mpatches.Patch(color=(0, 1, 0), label="Positive")
    neg = mpatches.Patch(color=(1, 0, 0), label="Negative")
    und = mpatches.Patch(color=(0, 0, 1), label="Undetermined")
    ax.imshow(img.clip(0, 1), interpolation="nearest")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.title.set_text(example["marker"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[pos, neg, und],
        loc="lower right",
        fancybox=False,
    )
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
    else:
        plt.show()
    plt.close()


def plot_together(example, keys, save_dir=None, save_file=None, dpi=160):
    """
    Plot the marker image, the marker activity segmentation and the instance segmentation overlayed
    Args:
        example (dict):
            Dictionary with keys "mplex_img", "marker_activity_mask", "instance_mask"
        keys (list):
            List of keys to plot
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """
    colors = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("BRGB", colors, N=4)
    fig, ax = plt.subplots(1, len(keys))
    fig.suptitle(example["marker"])
    for i, key in enumerate(keys):
        img = example[key]
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        if key == "marker_activity_mask":
            ax[i].imshow(img, interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
            pos = mpatches.Patch(color=colors[1], label="Positive")
            neg = mpatches.Patch(color=colors[0], label="Negative")
            und = mpatches.Patch(color=colors[3], label="Undetermined")
            ax[i].legend(
                handles=[pos, neg, und],
                loc="upper center",
                bbox_to_anchor=(-0.1, -0.2),
                fancybox=True,
                ncol=1,
                fontsize=6,
            )
        else:
            ax[i].imshow(img, interpolation="nearest")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(key, fontsize=6)
    for axx in ax:
        box = axx.get_position()
        axx.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
    else:
        plt.show()
    plt.close()


def plot_average_roc(avg_tpr, stddev, save_dir=None, save_file=None, dpi=160):
    """Plot the average ROC curve with the standard deviation
    Args:
        avg_tpr (np.ndarray):
            The average true positive rate
        stddev (np.ndarray):
            The standard deviation of the true positive rate
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """
    base_fpr = np.linspace(0, 1, avg_tpr.shape[0])
    tprs_upper = np.minimum(avg_tpr + stddev, 1)
    tprs_lower = avg_tpr - stddev

    plt.plot(base_fpr, avg_tpr, "b")
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
    else:
        plt.show()
    plt.close()


def plot_metrics_against_threshold(
    metric_dict, metric_keys, threshold_key, save_dir=None, save_file=None, dpi=160
):
    """Plot the metrics against the threshold
    Args:
        metric_dict (dict):
            A dictionary storing metrics for different thresholds
        metric_keys (list):
            A list of keys in metric_dict that are metrics you want to plot
        threshold_key (str):
            The key in metric_dict that is the threshold
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """
    for key in metric_keys:
        plt.plot(metric_dict[threshold_key], metric_dict[key], label=key)
        plt.xlabel("Threshold")
        plt.ylabel("Metric")
        plt.legend()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
    else:
        plt.show()
    plt.close()


def collapse_activity_dfs(pred_list):
    """Collapse activity dataframes
    Args:
        pred_list (list):
            list of dictionaries containing activity dataframes
    Returns:
        collapsed_df (pd.DataFrame):
            collapsed activity dataframe
    """
    collapse_list = []
    for sample in pred_list:
        sample["activity_df"]["marker"] = [sample["marker"]] * len(sample["activity_df"])
        collapse_list.append(sample["activity_df"])
    collapsed_df = pd.concat(collapse_list)
    return collapsed_df


def subset_activity_df(activity_df, subset_dict):
    """Subset predictions by marker
    Args:
        activity_df (pd.DataFrame):
            activity dataframe
        subset_dict (dict):
            dictionary mapping activity_df colnames to values in this column to subset
    Returns:
        subset_df (pd.DataFrame):
            subsetted activity dataframe
    """
    subset_df = activity_df.copy()
    for key, value in subset_dict.items():
        subset_df = subset_df[subset_df[key] == value]
    return subset_df


def subset_plots(
    activity_df, subset_list, save_dir=None, save_file=None, dpi=160, gt_key="activity",
    pred_key="prediction",
):
    """
    Plot the performance metrics of each marker in the subset_list
    Args:
        activity_df (pd.DataFrame):
            A dataframe containing the activity of each marker
        subset_list (list):
            A list of activity_df colnames that will be used to subset your data
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
        dpi (int):
            The resolution of the image to save, ignored if save_dir is None
    """
    # prepare subset_uniques
    subset_uniques = {}
    for subset in subset_list:
        subset_uniques[subset] = np.unique(getattr(activity_df, subset)).tolist()

    # prepare plot grid
    ndim = len(subset_list)
    if ndim == 1:
        plot_dim = [np.ceil(np.sqrt(len(subset_uniques[subset])))] * 2
    elif ndim > 1:
        plot_dim = [len(value) for value in subset_uniques.values()]
    plot_dim = [int(item) for item in plot_dim]
    fig, ax = plt.subplots(plot_dim[0], plot_dim[1], figsize=(plot_dim[1] * 7, plot_dim[0] * 7))
    k = 0
    for i in range(plot_dim[0]):
        for j in range(plot_dim[1]):
            if ndim == 1 and k < len(subset_uniques[subset_list[0]]):
                subset_key = subset_list[0]
                subset_dict = {subset_key: subset_uniques[subset_key][k]}
                k += 1
            elif ndim > 1:
                subset_dict = {
                    subset_list[0]: subset_uniques[subset_list[0]][i],
                    subset_list[1]: subset_uniques[subset_list[1]][j],
                }
            else:
                continue
            df = subset_activity_df(activity_df, subset_dict)
            thresholds = np.linspace(0.01, 1, 100)
            metrics = [calc_scores(df[gt_key], df[pred_key], t) for t in thresholds]
            metric_dict = {
                "threshold": thresholds,
                "precision": [np.mean(m["precision"]) for m in metrics],
                "recall": [np.mean(m["recall"]) for m in metrics],
                "specificity": [np.mean(m["specificity"]) for m in metrics],
                "f1_score": [np.mean(m["f1_score"]) for m in metrics],
            }
            threshold_key = "threshold"
            metric_keys = ["precision", "recall", "f1_score", "specificity"]
            for key in metric_keys:
                ax[i, j].set_title(
                    "".join([str(s[0]) + ": " + str(s[1]) + " " for s in subset_dict.items()])
                    + "pos=" + str(df["activity"].sum())
                    + " neg=" + str(len(df["activity"]) - df["activity"].sum())
                )
                ax[i, j].plot(metric_dict[threshold_key], metric_dict[key], label=key)
                ax[i, j].set_xlabel("Threshold")
                ax[i, j].set_ylabel("Metric")
                ax[i, j].legend(loc="lower right")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
        plt.close()
    else:
        plt.show()
    plt.close()


def heatmap_plot(
    activity_df, subset_list, save_dir=None, save_file=None, dpi=160, gt_key="activity",
    pred_key="prediction",
):
    """Plot the activity of each marker in the subset_list
        Args:
            activity_df (pd.DataFrame):
                A dataframe containing the activity of each marker
            subset_list (list):
                A list of activity_df colnames that will be used to subset your data
            save_dir (str):
                If specified, a directory where we will save the plot
            save_file (str):
                If save_dir specified, specify a file name you wish to save to.
                Ignored if save_dir is None
            dpi (int):
                The resolution of the image to save, ignored if save_dir is None
            gt_key (str):
                The key in the activity_df that contains the ground truth
            pred_key (str):
                The key in the activity_df that contains the prediction
    """
    # prepare subset_uniques
    subset_uniques = {}
    for subset in subset_list:
        subset_uniques[subset] = np.unique(getattr(activity_df, subset)).tolist()
    #
    thresholds = np.linspace(0.01, 1, 100)
    for key in subset_uniques:
        results = {}
        for class_ in subset_uniques[key]:
            df = subset_activity_df(activity_df, {key: class_})
            metrics = [calc_scores(df[gt_key], df[pred_key], t) for t in thresholds]
            metrics = pd.DataFrame(metrics)
            idx = metrics["f1_score"].idxmax()
            results[class_] = {
                "threshold": thresholds[idx],
                "precision": metrics["precision"][idx],
                "recall": metrics["recall"][idx],
                "specificity": metrics["specificity"][idx],
                "f1_score": metrics["f1_score"][idx],
            }
        results = pd.DataFrame(results).T
        # save results as csv
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            results.to_csv(os.path.join(save_dir, save_file.split(".")[0] + "_" + key + ".csv"))
        # plot heatmap
        ax = sns.heatmap(results, annot=True, cbar=False, cmap="viridis", vmin=0, vmax=1)
        ax.set(xlabel="", ylabel="")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_title("Split by " + str(key))
        # save figure with plt
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
            plt.close()
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record_path",
        type=str,
        default="C:/Users/lorenz/Desktop/angelo_lab/TONIC/TONIC.tfrecord",
    )
    parser.add_argument(
        "--save_dir", type=str, default="C:/Users/lorenz/Desktop/angelo_lab/TONIC/plots"
    )
    parser.add_argument("--dpi", type=float, default=160)
    parser.add_argument("--plot_overlay", default=True)
    parser.add_argument("--shuffle", default=True)
    args = parser.parse_args()
    path = args.record_path
    save_dir = args.save_dir
    dpi = args.dpi
    train_ds = tf.data.TFRecordDataset(path)
    if args.shuffle:
        train_ds = train_ds.shuffle(1500)
    for i, record in tqdm(enumerate(train_ds)):
        example_encoded = tf.io.parse_single_example(record, feature_description)
        example = parse_dict(example_encoded)
        plot_overlay(
            example,
            dpi=dpi,
            save_dir=save_dir,
            save_file=f"{example['marker']}_{example['folder_name']}_overlay_{i}.png",
        )
        # plot_together(
        #     example,
        #     dpi=dpi,
        #     save_dir=save_dir,
        #     save_file=f"{example['folder_name']}_{example['marker']}_together_{i}.png",
        # )
