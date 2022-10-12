
# def test_process_to_cells():
#     sample = {
#         "instance_mask": np.random.randint(0, 10, size=(256, 256, 1)),
#         "prediction": np.random.rand(256, 256, 1),
#         "activity_df": pd.DataFrame({"labels": list(range(1, 10))}).to_json()
#     }
#     sample["prediction"][sample["instance_mask"] == 1] = 1.0
#     sample["prediction"][sample["instance_mask"] == 2] = 0.0
#     sample = process_to_cells(sample)

#     # check if new keys are in sample dict
#     assert "prediction_mean" in list(sample.keys())
#     assert "pred_activity" in list(sample["activity_df"].columns)

#     # check types and shape
#     assert isinstance(sample["activity_df"], pd.DataFrame)
#     assert isinstance(sample["prediction_mean"], np.ndarray)
#     assert len(sample["activity_df"].pred_activity) == 9
#     assert sample["prediction_mean"].shape == (256, 256, 1)

#     # check values
#     assert sample["prediction_mean"][sample["instance_mask"] == 1].mean() == 1.0
#     assert sample["prediction_mean"][sample["instance_mask"] == 2].mean() == 0.0
#     assert sample["activity_df"].pred_activity[0] == 1.0
#     assert sample["activity_df"].pred_activity[1] == 0.0
