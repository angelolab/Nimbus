from segmentation_data_prep_test import prep_object_and_inputs
from cell_classification.promix_naive import PromixNaive
import toml
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import os


def test_reduce_to_cells():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    pred = np.random.rand(16, 256, 266)
    instance_mask = np.random.randint(0, 100, (16, 256, 266))
    instance_mask[-1, instance_mask[-1] == 1] = 0
    marker_activity_mask = np.zeros_like(instance_mask)
    marker_activity_mask[instance_mask > 90] = 1
    trainer = PromixNaive(params)
    uniques, mean_per_cell = tf.map_fn(
        trainer.reduce_to_cells,
        (pred, instance_mask),
        infer_shape=False,
        fn_output_signature=[
            tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0),
            tf.RaggedTensorSpec(shape=[None], dtype=tf.float32, ragged_rank=0),
        ],
    )

    # check that the output has the right dimension
    assert uniques.shape[0] == instance_mask.shape[0]
    assert mean_per_cell.shape[0] == instance_mask.shape[0]

    # check that the output is correct
    assert set(np.unique(instance_mask[0])) == set(uniques[0].numpy())
    for i in np.unique(instance_mask[0]):
        assert np.isclose(
            np.mean(pred[0][instance_mask[0] == i]),
            mean_per_cell[0][uniques[0] == i].numpy().max(),
        )


def test_matched_high_confidence_selection_thresholds():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    trainer = PromixNaive(params)
    trainer.matched_high_confidence_selection_thresholds()
    thresholds = trainer.confidence_loss_thresholds
    # check that the output has the right dimension
    assert len(thresholds) == 2
    assert thresholds["positive"] > 0.0
    assert thresholds["negative"] > 0.0


def test_train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = [tf_record_path]
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["dataset_names"] = ["test1"]
        params["dataset_sample_probs"] = [1.0]
        params["num_steps"] = 7
        params["num_validation"] = [2]
        params["num_test"] = [2]
        params["batch_size"] = 2
        params["test"] = True
        params["weight_decay"] = 1e-4
        params["snap_steps"] = 5
        params["val_steps"] = 5
        params["quantile"] = 0.3
        params["ema"] = 0.01
        params["confidence_thresholds"] = [0.1, 0.9]
        params["mixup_prob"] = 0.5
        trainer = PromixNaive(params)
        trainer.train()

        # check params.toml is dumped to file and contains the created paths
        assert "params.toml" in os.listdir(trainer.params["model_dir"])
        loaded_params = toml.load(os.path.join(trainer.params["model_dir"], "params.toml"))
        for key in ["model_dir", "log_dir", "model_path"]:
            assert key in list(loaded_params.keys())

        # check if model can be loaded from file
        trainer.model = None
        trainer.load_model(trainer.params["model_path"])
        assert isinstance(trainer.model, tf.keras.Model)


def test_prep_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = tf_record_path
        params["path"] = temp_dir
        params["experiment"] = "test"
        params["dataset_names"] = ["test1"]
        params["dataset_sample_probs"] = [1.0]
        params["num_steps"] = 3
        params["num_validation"] = [2]
        params["num_test"] = [2]
        params["batch_size"] = 2
        trainer = PromixNaive(params)
        trainer.prep_data()

        # check if train and validation datasets exists and are of the right type
        assert isinstance(trainer.validation_datasets[0], tf.data.Dataset)
        assert isinstance(trainer.train_dataset, tf.data.Dataset)


def prepare_activity_df():
    activity_df_list = []
    for i in range(4):
        activity_df = pd.DataFrame(
            {
                "labels": np.array([1, 2, 5, 7, 9, 11], dtype=np.uint16),
                "activity": [1, 0, 0, 0, 0, 1],
                "cell_type": ["T cell", "B cell", "T cell", "B cell", "T cell", "B cell"],
                "sample": [str(i)] * 6,
                "imaging_platform": ["test"] * 6,
                "dataset": ["test"] * 6,
                "marker": ["CD4"] * 6 if i % 2 == 0 else "CD8",
                "prediction": [0.9, 0.1, 0.1, 0.7, 0.7, 0.2],
            }
        )
        activity_df_list.append(activity_df)
    return activity_df_list


def test_class_wise_loss_selection():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    trainer = PromixNaive(params)
    activity_df_list = prepare_activity_df()
    df = activity_df_list[0]
    marker = df["marker"][0]
    dataset = df["dataset"][0]

    dataset_marker = dataset + "_" + marker
    trainer.class_wise_loss_quantiles[dataset_marker] = {"positive": 0.5, "negative": 0.5}
    df["loss"] = df.activity * df.prediction + (1 - df.activity) * (1 - df.prediction)
    positive_df = df[df["activity"] == 1]
    negative_df = df[df["activity"] == 0]
    selected_subset = trainer.class_wise_loss_selection(positive_df, negative_df, marker, dataset)

    # check that the output has the right dimension
    assert len(selected_subset) == 2
    assert len(selected_subset[0]) == 1

    # check that the output is correct and only those cells are selected that have a loss
    # smaller than the threshold
    assert selected_subset[0].equals(
        df[df["activity"] == 1].loc[
            df["loss"] <= trainer.class_wise_loss_quantiles[dataset_marker]["positive"]
        ]
    )
    assert selected_subset[1].equals(
        df[df["activity"] == 0].loc[
            df["loss"] <= trainer.class_wise_loss_quantiles[dataset_marker]["negative"]
        ]
    )

    # check if quantiles got updated
    assert trainer.class_wise_loss_quantiles[dataset_marker]["positive"] != 0.5
    assert trainer.class_wise_loss_quantiles[dataset_marker]["negative"] != 0.5


def test_matched_high_confidence_selection():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    trainer = PromixNaive(params)
    activity_df_list = prepare_activity_df()
    df = activity_df_list[0]
    df["loss"] = df.activity * df.prediction + (1 - df.activity) * (1 - df.prediction)
    positive_df = df[df["activity"] == 1]
    negative_df = df[df["activity"] == 0]
    mark = df["marker"][0]
    trainer.matched_high_confidence_selection_thresholds()
    selected_subset = trainer.matched_high_confidence_selection(positive_df, negative_df)
    df = pd.concat(selected_subset)

    # check that the output has the right dimension
    assert len(df) == 1

    # check that the output is correct and only those cells are selected that have a loss
    # smaller than the threshold
    gt_activity = "positive" if df["activity"].values[0] == 1 else "negative"
    assert (df.loss.values[0] <= trainer.confidence_loss_thresholds[gt_activity]).numpy()


def test_batchwise_loss_selection():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    trainer = PromixNaive(params)
    trainer.matched_high_confidence_selection_thresholds()
    activity_df_list = prepare_activity_df()
    instance_mask = np.zeros([256, 256], dtype=np.uint8)
    i = 1
    for h in range(0, 260, 20):
        for w in range(0, 260, 20):
            instance_mask[h: h + 10, w: w + 10] = i
            i += 1
    dfs = activity_df_list[:2]
    mark = [tf.constant(str(df["marker"][0]).encode()) for df in dfs]
    dset = [tf.constant(str(df["dataset"][0]).encode()) for df in dfs]

    for df in dfs:
        df["loss"] = df.activity * df.prediction + (1 - df.activity) * (1 - df.prediction)
    instance_mask = instance_mask[np.newaxis, ..., np.newaxis]
    instance_mask = np.concatenate([instance_mask, instance_mask], axis=0)
    loss_mask = trainer.batchwise_loss_selection(dfs, instance_mask, mark, dset)

    # check that the output has the right dimension
    assert list(loss_mask.shape) == [2, 256, 256]

    # check that they are equal
    assert np.array_equal(loss_mask[0], loss_mask[1])


def test_quantile_scheduler():
    params = toml.load("cell_classification/configs/params.toml")
    params["test"] = True
    trainer = PromixNaive(params)
    quantile_start = params["quantile"]
    quantile_end = params["quantile_end"]
    quantile_warmup_steps = params["quantile_warmup_steps"]
    step_0_quantile = trainer.quantile_scheduler(0)
    step_half_warmpup_quantile = trainer.quantile_scheduler(quantile_warmup_steps // 2)
    step_warump_quantile = trainer.quantile_scheduler(quantile_warmup_steps)
    step_n_quantile = trainer.quantile_scheduler(quantile_warmup_steps*2)

    # check that the output has the expected values
    assert step_0_quantile == quantile_start
    assert step_half_warmpup_quantile == (quantile_start + quantile_end) / 2
    assert step_warump_quantile == quantile_end
    assert step_n_quantile == quantile_end


def test_gen_prep_batches_promix_fn():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(data_prep.tf_record_path, data_prep.dataset + ".tfrecord")
        params = toml.load("cell_classification/configs/params.toml")
        params["record_path"] = [tf_record_path]
        params["path"] = temp_dir
        params["batch_constituents"] = ["mplex_img", "binary_mask", "nuclei_img", "membrane_img"]
        params["experiment"] = "test"
        params["dataset_names"] = ["test1"]
        params["num_steps"] = 20
        params["dataset_sample_probs"] = [1.0]
        params["batch_size"] = 1
        params["test"] = True
        params["num_validation"] = [2]
        params["num_test"] = [2]
        trainer = PromixNaive(params)
        trainer.prep_data()
        example = next(iter(trainer.train_dataset))
        prep_batches_promix_4 = trainer.prep_batches_promix
        prep_batches_promix_2 = trainer.gen_prep_batches_promix_fn(
            keys=["mplex_img", "binary_mask"]
        )

        # check if each batch contains the above specified constituents
        batch_2 = prep_batches_promix_2(example)
        assert batch_2[0].shape[-1] == 1
        assert np.array_equal(batch_2[0], example["mplex_img"])

        batch_4 = prep_batches_promix_4(example)
        assert batch_4[0].shape[-1] == 3
        assert np.array_equal(batch_4[0], tf.concat([
            example["mplex_img"], example["nuclei_img"], example["membrane_img"]
            ], axis=-1)
        )
