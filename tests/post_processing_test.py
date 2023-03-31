from cell_classification.post_processing import process_to_cells, merge_activity_df
import numpy as np
import pandas as pd


def test_process_to_cells():
    # prepare data
    instance_mask = np.random.randint(0, 10, size=(256, 256, 1))
    prediction = np.random.rand(256, 256, 1)
    prediction[instance_mask == 1] = 1.0
    prediction[instance_mask == 2] = 0.0
    prediction_mean, activity_df = process_to_cells(instance_mask, prediction)

    # check types and shape
    assert isinstance(activity_df, pd.DataFrame)
    assert isinstance(prediction_mean, np.ndarray)
    assert len(activity_df.pred_activity) == 9
    assert prediction_mean.shape == (256, 256, 1)

    # check values
    assert prediction_mean[instance_mask == 1].mean() == 1.0
    assert prediction_mean[instance_mask == 2].mean() == 0.0
    assert activity_df.pred_activity[0] == 1.0
    assert activity_df.pred_activity[1] == 0.0


def test_merge_activity_df():
    # prepare data
    pred_df = pd.DataFrame({"labels": list(range(1, 10)), "pred_activity": np.random.rand(9)})
    gt_df = pd.DataFrame(
        {"labels": list(range(1, 10)), "gt_activity": np.random.randint(0, 2, 9)}
    )
    merged_df = merge_activity_df(gt_df, pred_df)

    # check if columns are there and no labels got lost in the merge
    assert np.array_equal(merged_df.labels, gt_df.labels)
    assert set(['labels', 'gt_activity', "pred_activity"]) == set(merged_df.columns)
