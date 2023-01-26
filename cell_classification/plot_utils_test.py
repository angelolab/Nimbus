from cgi import test
from segmentation_data_prep import parse_dict, feature_description
from segmentation_data_prep_test import prep_object_and_inputs
import pytest
import tempfile
from plot_utils import plot_overlay, plot_together, plot_average_roc, subset_plots, heatmap_plot
from plot_utils import plot_metrics_against_threshold, subset_activity_df, collapse_activity_dfs
from metrics_test import make_pred_list
from metrics import calc_roc, average_roc, calc_metrics
import os
import tensorflow as tf


def prepare_dataset(temp_dir):
    data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
    data_prep.tf_record_path = temp_dir
    data_prep.make_tf_record()
    tf_record_path = os.path.join(temp_dir, data_prep.dataset + ".tfrecord")
    dataset = tf.data.TFRecordDataset(tf_record_path)
    return dataset


def test_plot_overlay():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = iter(prepare_dataset(temp_dir))
        plot_path = os.path.join(temp_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        record = next(dataset)
        example_encoded = tf.io.parse_single_example(record, feature_description)
        example = parse_dict(example_encoded)
        plot_overlay(
            example, save_dir=plot_path, save_file=f"{example['folder_name']}_overlay.png"
        )

        # check if plot was saved
        assert os.path.exists(os.path.join(plot_path, f"{example['folder_name']}_overlay.png"))


def test_plot_together():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = iter(prepare_dataset(temp_dir))
        plot_path = os.path.join(temp_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        record = next(dataset)
        example_encoded = tf.io.parse_single_example(record, feature_description)
        example = parse_dict(example_encoded)
        plot_together(
            example, ["mplex_img", "nuclei_img", "marker_activity_mask"], save_dir=plot_path,
            save_file=f"{example['folder_name']}_together.png"
        )

        # check if plot was saved
        assert os.path.exists(os.path.join(plot_path, f"{example['folder_name']}_together.png"))


def test_plot_average_roc():
    with tempfile.TemporaryDirectory() as temp_dir:
        pred_list = make_pred_list()
        roc_list = calc_roc(pred_list)
        tprs, mean_tprs, base, std, mean_thresh = average_roc(roc_list)
        plot_path = os.path.join(temp_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        plot_average_roc(mean_tprs, std, save_dir=plot_path, save_file="average_roc.png")

        # check if plot was saved
        assert os.path.exists(os.path.join(plot_path, "average_roc.png"))


def test_plot_metrics_against_threshold():
    with tempfile.TemporaryDirectory() as temp_dir:
        pred_list = make_pred_list()
        avg_metrics = calc_metrics(pred_list)
        plot_path = os.path.join(temp_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        plot_metrics_against_threshold(
            avg_metrics, metric_keys=['precision', 'recall', 'f1_score'], save_dir=plot_path,
            threshold_key="threshold", save_file="metrics_against_threshold.png"
        )

        # check if plot was saved
        assert os.path.exists(os.path.join(plot_path, "metrics_against_threshold.png"))


def test_collapse_activity_dfs():
    pred_list = make_pred_list()
    df = collapse_activity_dfs(pred_list)

    # check if df has the right shape and keys
    assert df.shape == (len(pred_list)*6, 8)
    assert set(df.columns) == set(pred_list[0]["activity_df"].columns)


def test_subset_activity_df():
    pred_list = make_pred_list()
    df = collapse_activity_dfs(pred_list)
    cd4_subset = subset_activity_df(df, {"marker": "CD4"})
    cd4_tcells_subset = subset_activity_df(df, {"marker": "CD4", "cell_type": "T cell"})

    # check if cd4_subset and cd8_subset have the right shape
    assert set(cd4_subset["marker"]) == set(["CD4"])
    assert cd4_subset.shape == df[df.marker == "CD4"].shape

    # check if cd4_tcells_subset has the right shape
    assert set(cd4_tcells_subset["marker"]) == set(["CD4"])
    assert set(cd4_tcells_subset["cell_type"]) == set(["T cell"])
    assert cd4_tcells_subset.shape == df[(df.marker == "CD4") & (df.cell_type == "T cell")].shape


def test_subset_plots():
    pred_list = make_pred_list()
    activity_df = collapse_activity_dfs(pred_list)
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = os.path.join(temp_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        subset_plots(
            activity_df, subset_list=["marker"], save_dir=plot_path,
            save_file="split_by_marker.png"
        )
        subset_plots(
            activity_df, subset_list=["marker", "cell_type"], save_dir=plot_path,
            save_file="split_by_marker_ct.png"
        )

        # check if plots were saved
        assert os.path.exists(os.path.join(plot_path, "split_by_marker.png"))


def test_heatmap_plot():
    pred_list = make_pred_list()
    activity_df = collapse_activity_dfs(pred_list)
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = os.path.join(temp_dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        heatmap_plot(
            activity_df, ["marker"], save_dir=plot_path, save_file="heatmap.png"
        )

        # check if plot was saved
        assert os.path.exists(os.path.join(plot_path, "heatmap.png"))
