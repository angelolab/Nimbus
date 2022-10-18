import argparse
import toml
import os
import pickle
import pandas as pd
import numpy as np
from metrics import load_model_and_val_data, calc_roc, average_roc, calc_metrics
from plot_utils import plot_average_roc, plot_metrics_against_threshold, subset_plots
from plot_utils import collapse_activity_dfs, subset_activity_df
import matplotlib.pyplot as plt
from metrics import calc_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model weights",
        default="C:\\Users\\lorenz\\Desktop\\angelo_lab\\cell_classification\\checkpoints\\" +
        "ex_2\\ex_2.h5",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to model params",
        default="C:\\Users\\lorenz\\Desktop\\angelo_lab\\cell_classification\\checkpoints\\"
        + "ex_2\\params.toml",
    )
    args = parser.parse_args()
    with open(args.params_path, "r") as f:
        params = toml.load(f)
    params["model_path"] = args.model_path
    model, val_dset = load_model_and_val_data(params)
    params["eval_dir"] = os.path.join(*os.path.split(params["model_path"])[:-1], "eval")
    os.makedirs(params["eval_dir"], exist_ok=True)
    pred_list = model.predict_dataset(val_dset, False)

    # pixel level evaluation
    print("Calculate ROC curve")
    roc = calc_roc(pred_list)
    with open(os.path.join(params["eval_dir"], "roc.pkl"), "wb") as f:
        pickle.dump(roc, f)
    tprs, mean_tprs, fpr, std, mean_thresh = average_roc(roc)
    plot_average_roc(mean_tprs, std, save_dir=params["eval_dir"], save_file="avg_roc.png")
    print("AUC: {}".format(np.mean(roc["auc"])))

    # print("Calculate precision, recall, f1_score and accuracy")
    # avg_metrics = calc_metrics(pred_list)
    # pd.DataFrame(avg_metrics).to_csv(
    #     os.path.join(params["eval_dir"], "pixel_metrics.csv"), index=False
    # )
    # plot_metrics_against_threshold(
    #     avg_metrics,
    #     metric_keys=["precision", "recall", "f1_score"],
    #     threshold_key="threshold",
    #     save_dir=params["eval_dir"],
    #     save_file="precision_recall_f1.png",
    # )

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
        metric_keys=["precision", "recall", "f1_score", "specificity"],
        threshold_key="threshold",
        save_dir=params["eval_dir"],
        save_file="precision_recall_f1_cell_lvl.png",
    )

    print("Plot activity predictions split by markers and cell types")
    activity_df = collapse_activity_dfs(pred_list)
    activity_df.to_csv(
        os.path.join(params["eval_dir"], "pred_activity_df.csv"), index=False
    )
    subset_plots(
        activity_df, subset_list=["marker"],
        save_dir=params["eval_dir"],
        save_file="split_by_marker.png",
        gt_key="activity",
        pred_key="pred_activity",
    )
    subset_plots(
        activity_df, subset_list=["cell_type"],
        save_dir=params["eval_dir"],
        save_file="split_by_cell_type.png",
        gt_key="activity",
        pred_key="pred_activity",
    )
    subset_plots(
        activity_df, subset_list=["cell_type", "marker"],
        save_dir=params["eval_dir"],
        save_file="split_by_marker_ct.png",
        gt_key="activity",
        pred_key="pred_activity",
    )
