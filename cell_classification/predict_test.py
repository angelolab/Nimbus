import h5py
from model_builder_test import prep_object_and_inputs
from model_builder import ModelBuilder
import tempfile
import toml
import os
from predict import predict, calc_roc, calc_metrics, average_roc, HDF5Loader, process_to_cells
import numpy as np
import pandas as pd


def test_predict():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate a model and run predict
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["num_epochs"] = 0
        params["num_validation"] = 2
        model = ModelBuilder(params)
        model.train()
        model.params["eval"] = True
        model.prep_data()
        val_dset = iter(model.validation_dataset)
        single_example_list, params = predict(model, val_dset, params, True)

        # check if predict returns a list with the right number of items
        assert len(single_example_list) == params["num_validation"]

        # check if params were saved to file
        assert "params.toml" in os.listdir(params["model_dir"])

        # check if examples get serialized correctly
        for i in range(params["num_validation"]):
            assert str(i).zfill(4)+'_pred.hdf' in list(os.listdir(params['eval_dir']))

        with h5py.File(os.path.join(params['eval_dir'], str(0).zfill(4)+'_pred.hdf'), 'r') as f:
            assert f['prediction'].shape == (256, 256, 1)
            assert f['marker_activity_mask'].shape == (256, 256, 1)
            assert set(list(f.keys())) == set(list(single_example_list[0].keys()))

        params = predict(model, val_dset, params)
        assert isinstance(params, dict)


def make_pred_list():
    pred_list = []
    for _ in range(10):
        instance_mask = np.random.randint(0, 10, size=(256, 256, 1))
        binary_mask = (instance_mask > 0).astype(np.uint8)
        pred_list.append(
            {
                "marker_activity_mask": np.random.randint(0, 2, (256, 256, 1))*binary_mask,
                "prediction": np.random.rand(256, 256, 1),
                "instance_mask": instance_mask,
                "binary_mask": binary_mask,
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
        params["num_epochs"] = 0
        params["num_validation"] = 2
        model = ModelBuilder(params)
        model.train()
        model.params["eval"] = True
        model.prep_data()
        val_dset = iter(model.validation_dataset)
        single_example_list, params = predict(model, val_dset, params, True)
        generator = HDF5Loader(params['eval_dir'])

        # check if generator has the right number of items
        assert len(generator) == params['num_validation']

        # check if generator returns the right items
        for sample in generator:
            assert isinstance(sample, dict)
            assert set(list(sample.keys())) == set(list(single_example_list[0].keys()))


def test_process_to_cells():
    sample = {
        "instance_mask": np.random.randint(0, 10, size=(256, 256, 1)),
        "prediction": np.random.rand(256, 256, 1),
        "activity_df": pd.DataFrame({"labels": list(range(1, 10))}).to_json()
    }
    sample["prediction"][sample["instance_mask"] == 1] = 1.0
    sample["prediction"][sample["instance_mask"] == 2] = 0.0
    sample = process_to_cells(sample)

    # check if new keys are in sample dict
    assert "prediction_mean" in list(sample.keys())
    assert "pred_activity" in list(sample["activity_df"].columns)

    # check types and shape
    assert isinstance(sample["activity_df"], pd.DataFrame)
    assert isinstance(sample["prediction_mean"], np.ndarray)
    assert len(sample["activity_df"].pred_activity) == 9
    assert sample["prediction_mean"].shape == (256, 256, 1)

    # check values
    assert sample["prediction_mean"][sample["instance_mask"] == 1].mean() == 1.0
    assert sample["prediction_mean"][sample["instance_mask"] == 2].mean() == 0.0
    assert sample["activity_df"].pred_activity[0] == 1.0
    assert sample["activity_df"].pred_activity[1] == 0.0
