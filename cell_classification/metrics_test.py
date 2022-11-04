import h5py
from model_builder_test import prep_object_and_inputs
from model_builder import ModelBuilder
import tempfile
import toml
import os
from metrics import calc_roc, calc_metrics, average_roc, HDF5Loader
import numpy as np
import pandas as pd
import tensorflow as tf


def make_pred_list():
    pred_list = []
    for i in range(10):
        instance_mask = np.random.randint(0, 10, size=(256, 256, 1))
        binary_mask = (instance_mask > 0).astype(np.uint8)
        activity_df = pd.DataFrame(
            {
                "labels": np.array([1, 2, 5, 7, 9, 11], dtype=np.uint16),
                "activity": [1, 0, 0, 0, 0, 1],
                "cell_type": ["T cell", "B cell", "T cell", "B cell", "T cell", "B cell"],
                "sample": [str(i)]*6,
                "imaging_platform": ["test"]*6,
                "dataset": ["test"]*6,
                "marker": ["CD4"]*6 if i % 2 == 0 else "CD8",
                "prediction": np.random.rand(6),
            }
        )
        pred_list.append(
            {
                "marker_activity_mask": np.random.randint(0, 2, (256, 256, 1))*binary_mask,
                "prediction": np.random.rand(256, 256, 1),
                "instance_mask": instance_mask,
                "binary_mask": binary_mask,
                "dataset": "test", "imaging_platform": "test",
                "marker": "CD4" if i % 2 == 0 else "CD8",
                "activity_df": activity_df,
            }
        )
    pred_list[-1]["marker_activity_mask"] = np.zeros((256, 256, 1))
    return pred_list


def test_calc_roc():
    pred_list = make_pred_list()
    roc = calc_roc(pred_list)

    # check if roc has the right keys
    assert set(roc.keys()) == set(["fpr", "tpr", "auc", "thresholds"])

    # check if roc has the right number of items
    assert len(roc["fpr"]) == len(roc["tpr"]) == len(roc["thresholds"]) == len(roc["auc"]) == 9


def test_calc_metrics():
    pred_list = make_pred_list()
    avg_metrics = calc_metrics(pred_list)
    keys = [
        "accuracy", "precision", "recall", "specificity", "f1_score", "tp", "tn", "fp", "fn",
        "dataset", "imaging_platform", "marker", "threshold",
    ]

    # check if avg_metrics has the right keys
    assert set(avg_metrics.keys()) == set(keys)

    # check if avg_metrics has the right number of items
    assert (
        len(avg_metrics["accuracy"]) == len(avg_metrics["precision"])
        == len(avg_metrics["recall"]) == len(avg_metrics["f1_score"])
        == len(avg_metrics["tp"]) == len(avg_metrics["tn"])
        == len(avg_metrics["fp"]) == len(avg_metrics["fn"])
        == len(avg_metrics["dataset"]) == len(avg_metrics["imaging_platform"])
        == len(avg_metrics["marker"]) == len(avg_metrics["threshold"]) == 50
    )


def test_average_roc():
    pred_list = make_pred_list()
    roc_list = calc_roc(pred_list)
    tprs, mean_tprs, base, std, mean_thresh = average_roc(roc_list)

    # check if mean_tprs, base, std and mean_thresh have the same length
    assert len(mean_tprs) == len(base) == len(std) == len(mean_thresh) == tprs.shape[1]

    # check if mean and std give reasonable results
    assert np.array_equal(np.mean(tprs, axis=0), mean_tprs)
    assert np.array_equal(np.std(tprs, axis=0), std)


def test_HDF5Generator():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["num_steps"] = 2
        params["num_validation"] = 2
        params["snap_steps"] = 100
        params["val_steps"] = 100
        model = ModelBuilder(params)
        model.train()
        model.predict_dataset(model.validation_dataset, save_predictions=True)
        generator = HDF5Loader(model.params['eval_dir'])

        # check if generator has the right number of items
        assert len(generator) == params['num_validation']

        # check if generator returns the right items
        for sample in generator:
            assert isinstance(sample, dict)
            assert len(list(sample.keys())) == 11
