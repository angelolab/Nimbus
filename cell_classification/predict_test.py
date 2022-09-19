from model_builder_test import prep_object_and_inputs
from model_builder import ModelBuilder
import tempfile
import toml
import os
from predict import predict, calc_roc, calc_metrics, average_roc
import numpy as np


def test_predict():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["num_epochs"] = 2
        params["num_validation"] = 2
        model = ModelBuilder(params)
        model.train()
        model.params["eval"] = True
        model.prep_data()
        val_dset = iter(model.validation_dataset)
        single_example_list, params = predict(model, val_dset, params)

        # check if predict returns a list with the right number of items
        assert len(single_example_list) == params["num_validation"]

        # check if predictions were pickled
        assert "pred_list.pkl" in os.listdir(params["eval_dir"])

        # check if params were saved to file
        assert "params.toml" in os.listdir(params["model_dir"])


def make_pred_list():
    pred_list = []
    for _ in range(10):
        pred_list.append(
            {
                "marker_activity_mask": np.random.randint(0, 2, (256, 256, 1)),
                "prediction": np.random.rand(256, 256, 1),
                "dataset": "test", "imaging_platform": "test", "marker": "test",
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
        "accuracy", "precision", "recall", "f1_score", "tp", "tn", "fp", "fn", "dataset",
        "imaging_platform", "marker", "threshold",
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
        == len(avg_metrics["marker"]) == len(avg_metrics["threshold"]) == 101
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
