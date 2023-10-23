from cell_classification.model_builder import ModelBuilder
from cell_classification.metrics import calc_scores
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import argparse
import toml
import os


def hyperparameter_search(params, n_jobs=10):
    """
    Hyperparameter search for the best pos/neg threshold per marker and dataset
    Args:
        params: configs comparable to Nimbus/configs/params.toml
        n_jobs: number of jobs for parallelization
    Returns:
    """
    optimal_thresh_dict = {}
    model = ModelBuilder(params)
    model.prep_data()
    df = model.predict_dataset_list(model.validation_datasets, save_predictions=False)
    print("Run hyperparameter search")
    thresholds = np.linspace(0, 1, 101)
    thresholds = [np.round(thresh, 2) for thresh in thresholds]
    for dataset in df.dataset.unique():
        df_subset = df[df.dataset == dataset]
        if dataset not in optimal_thresh_dict.keys():
            optimal_thresh_dict[dataset] = {}
        for marker in tqdm(df_subset.marker.unique()):
            df_subset_marker = df_subset[df_subset.marker == marker]
            metrics = Parallel(n_jobs=n_jobs)(
                delayed(calc_scores)(
                    df_subset_marker["activity"].astype(np.int32), df_subset_marker["prediction"],
                    threshold=thresh
                ) for thresh in thresholds
            )
            f1_scores = [metric["f1_score"] for metric in metrics]
            optimal_thresh_dict[dataset][marker] = thresholds[np.argmax(f1_scores)]
    # assign classes based on optimal thresholds
    df["pred_class"] = df.apply(
        lambda row: 1 if row["prediction"] >= optimal_thresh_dict[row["dataset"]][row["marker"]] else 0,
        axis=1
    )
    df.to_csv(os.path.join(params["path"], params["experiment"], "validation_predictions.csv"))
    # save as toml
    fpath = os.path.join(params["path"], params["experiment"], "optimal_thresholds.toml")
    with open(fpath, "w") as f:
        toml.dump(optimal_thresh_dict, f)
    return optimal_thresh_dict, df


def prepare_testset_predictions(params, optimal_thresh_dict):
    """
    Prepare testset predictions based on optimal thresholds
    Args:
        params: configs comparable to Nimbus/configs/params.toml
        optimal_thresh_dict: optimal thresholds per marker and dataset
    Returns:
    """
    model = ModelBuilder(params)
    model.prep_data()
    df = model.predict_dataset_list(model.test_datasets, save_predictions=False)
    # assign classes based on optimal thresholds
    df["pred_class"] = df.apply(
        lambda row: 1 if row["prediction"] >= optimal_thresh_dict[row["dataset"]][row["marker"]] else 0,
        axis=1
    )
    df.to_csv(os.path.join(params["path"], params["experiment"], "test_predictions.csv"))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="configs/params.toml")
    args = parser.parse_args()
    params = toml.load(args.params)
    if not os.path.exists(
        os.path.join(params["path"], params["experiment"], "optimal_thresholds.toml")
    ):
        optimal_thresh_dict,_ = hyperparameter_search(params)
    prepare_testset_predictions(params, optimal_thresh_dict)
