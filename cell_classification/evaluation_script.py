import argparse
import toml
import ast
import os
import pickle
import pandas as pd
import numpy as np
from metrics import load_model, calc_roc, average_roc, calc_metrics
from plot_utils import plot_average_roc, plot_metrics_against_threshold, subset_plots
from plot_utils import heatmap_plot, plot_together
import tensorflow as tf
from segmentation_data_prep import parse_dict, feature_description


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model weights",
        default=None,
    )
    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to model params",
        default="E:\\angelo_lab\\test\\params.toml",
    )
    parser.add_argument(
        "--worst_n",
        type=int,
        help="Number of worst predictions to plot",
        default=20,
    )
    parser.add_argument(
        "--best_n",
        type=int,
        help="Number of best predictions to plot",
        default=20,
    )
    parser.add_argument(
        "--split_by_marker",
        type=bool,
        help="Split best/worst predictions by marker",
        default=True,
    )
    parser.add_argument(
        "--external_datasets",
        type=str,
        help="List of paths to tfrecord datasets",
        nargs='+',
        default=[],
    )
    args = parser.parse_args()
    with open(args.params_path, "r") as f:
        params = toml.load(f)
    if args.model_path is not None:
        params["model_path"] = args.model_path

    model = load_model(params)
    datasets = {name: dataset for name, dataset in zip(model.dataset_names, model.test_datasets)}
    if hasattr(args, "external_datasets"):
        external_datasets = {
            os.path.split(external_dataset)[-1].split(".")[0]: external_dataset.replace(",", "")
            for external_dataset in args.external_datasets
        }
        external_datasets = {
            key: tf.data.TFRecordDataset(external_datasets[key]) for key in external_datasets.keys()
        }
        external_datasets = {
            name: dataset.map(
                lambda x: tf.io.parse_single_example(x, feature_description),
                num_parallel_calls=tf.data.AUTOTUNE,
            ) for name, dataset in external_datasets.items()
        }
        external_datasets = {
            name: dataset.map(
                parse_dict, num_parallel_calls=tf.data.AUTOTUNE
            ) for name, dataset in external_datasets.items()
        }
        external_datasets = {
            name: dataset.batch(params["batch_size"], drop_remainder=False
        ) for name, dataset in external_datasets.items()}

        datasets.update(external_datasets)
 
    for name, val_dset in datasets.items():
        params["eval_dir"] = os.path.join(*os.path.split(params["model_path"])[:-1], "eval", name)
        os.makedirs(params["eval_dir"], exist_ok=True)
        # iterate over datasets
        pred_list = model.predict_dataset(val_dset, False)

        # prepare cell_table
        activity_list = []
        for pred in pred_list:
            activity_df = pred["activity_df"].copy()
            for key in ["dataset", "marker", "folder_name"]:
                activity_df[key] = [pred[key]]*len(activity_df)
            activity_list.append(activity_df)
        activity_df = pd.concat(activity_list)
        activity_df.to_csv(os.path.join(params["eval_dir"], "pred_cell_table.csv"), index=False)

        # cell level evaluation
        roc = calc_roc(pred_list, gt_key="activity", pred_key="pred_activity", cell_level=True)
        with open(os.path.join(params["eval_dir"], "roc_cell_lvl.pkl"), "wb") as f:
            pickle.dump(roc, f)

        # find index of n worst predictions and save plots of them
        roc_df = pd.DataFrame(roc)
        if args.split_by_marker:
            worst_idx = []
            best_idx = []
            markers = np.unique(roc_df.marker)
            for marker in markers:
                marker_df = roc_df[roc_df.marker == marker]
                sort_idx = np.argsort(marker_df.auc).index
                worst_idx.extend(sort_idx[-args.worst_n:])
                best_idx.extend(sort_idx[:args.best_n])
        else:
            sort_idx = np.argsort(roc["auc"])
            worst_idx = sort_idx[-args.worst_n:]
            best_idx = sort_idx[:args.best_n]
        for idx_list, best_worst in [(best_idx, "best"), (worst_idx, "worst")]:
            for i, idx in enumerate(idx_list):
                pred = pred_list[idx]
                plot_together(
                    pred, keys=["mplex_img", "marker_activity_mask", "prediction"],
                    save_dir=os.path.join(params["eval_dir"], best_worst + "_predictions"),
                    save_file="worst_{}_{}_{}.png".format(
                        i, pred["marker"], pred["dataset"], pred["folder_name"]
                    )
                )

        pd.DataFrame(roc).auc
        tprs, mean_tprs, fpr, std, mean_thresh = average_roc(roc)
        plot_average_roc(
            mean_tprs, std, save_dir=params["eval_dir"], save_file="avg_roc_cell_lvl.png"
        )
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
            metric_keys=["precision", "recall", "f1_score", "specificity"],
            threshold_key="threshold",
            save_dir=params["eval_dir"],
            save_file="precision_recall_f1_cell_lvl.png",
        )

        print("Plot activity predictions split by markers and cell types")
        subset_plots(
            activity_df, subset_list=["marker"],
            save_dir=params["eval_dir"],
            save_file="split_by_marker.png",
            gt_key="activity",
            pred_key="pred_activity",
        )
        if "cell_type" in activity_df.columns:
            subset_plots(
                activity_df, subset_list=["cell_type"],
                save_dir=params["eval_dir"],
                save_file="split_by_cell_type.png",
                gt_key="activity",
                pred_key="pred_activity",
            )
        heatmap_plot(
            activity_df, subset_list=["marker"],
            save_dir=params["eval_dir"],
            save_file="heatmap_split_by_marker.png",
            gt_key="activity",
            pred_key="pred_activity",
        )
