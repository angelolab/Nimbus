import copy
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tifffile import imread, imwrite

from cell_classification.segmentation_data_prep import (SegmentationTFRecords,
                                                        feature_description,
                                                        parse_dict)


def prep_object(
    data_dir="path", cell_table_path="path", conversion_matrix_path="path",
    normalization_dict_path="path", tf_record_path="path", tile_size=[256, 256], stride=[256, 256],
    normalization_quantile=0.999, selected_markers=None, segmentation_naming_convention=None,
    imaging_platform="imaging_platform", dataset="dataset", tissue_type="tissue_type",
):
    data_prep = SegmentationTFRecords(
        data_dir=data_dir, cell_table_path=cell_table_path,
        conversion_matrix_path=conversion_matrix_path, imaging_platform=imaging_platform,
        dataset=dataset, tile_size=tile_size, stride=stride, tf_record_path=tf_record_path,
        normalization_dict_path=normalization_dict_path, selected_markers=selected_markers,
        normalization_quantile=normalization_quantile, tissue_type=tissue_type,
        segmentation_naming_convention=segmentation_naming_convention
    )
    return data_prep


def prep_object_and_inputs(
        temp_dir, imaging_platform="imaging_platform", dataset="dataset", num_folders=5,
        selected_markers=["CD4"], scale=[0.5, 1.0, 1.5, 2.0, 5.0],
):
    # create temporary folders with data for the tests
    conversion_matrix = prepare_conversion_matrix()
    conversion_matrix_path = os.path.join(temp_dir, "conversion_matrix.csv")
    conversion_matrix.to_csv(conversion_matrix_path, index=True)
    norm_dict = {"CD11c": 1.0, "CD4": 1.0, "CD56": 1.0, "CD57": 1.0}
    with open(os.path.join(temp_dir, "norm_dict.json"), "w") as f:
        json.dump(norm_dict, f)
    data_folders = prepare_test_data_folders(
        num_folders, temp_dir, list(norm_dict.keys()) + ["XYZ"], random=True,
        scale=scale
    )
    cell_table_path = os.path.join(temp_dir, "cell_type_table.csv")
    cell_table = prepare_cell_type_table()
    cell_table.to_csv(cell_table_path, index=False)
    data_prep = prep_object(
        data_dir=temp_dir,
        conversion_matrix_path=conversion_matrix_path,
        tf_record_path=temp_dir,
        cell_table_path=cell_table_path,
        normalization_dict_path=None,
        selected_markers=selected_markers,
        imaging_platform=imaging_platform,
        dataset=dataset
    )
    data_prep.load_and_check_input()
    return data_prep, data_folders, conversion_matrix, cell_table


def prepare_conversion_matrix():
    conversion_matrix = pd.DataFrame(
        np.random.randint(0, 3, size=(6, 4)).clip(0, 1),
        # np.ones([6,4]),
        columns=["CD11c", "CD4", "CD56", "CD57"],
        index=["stromal", "FAP", "NK", "CD4", "CD14", "CD163"],
    )
    return conversion_matrix


def test_get_image():
    data_prep = prep_object()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_img_1 = np.random.rand(256, 256)
        test_img_2 = np.random.rand(256, 256, 1)
        imwrite(os.path.join(temp_dir, "CD8.tiff"), test_img_1)
        imwrite(os.path.join(temp_dir, "CD4.tiff"), test_img_2)
        CD8_img = data_prep.get_image(data_folder=temp_dir, marker="CD8")
        CD4_img = data_prep.get_image(data_folder=temp_dir, marker="CD4")

        # test if the images are the same and a single channel image is always returned
        assert np.array_equal(test_img_1, np.squeeze(CD8_img))
        assert np.array_equal(test_img_2, CD4_img)
        assert not np.array_equal(CD8_img, CD4_img)


def prepare_test_data_folders(num_folders, temp_dir, selected_markers, random=False, scale=[1.0]):
    data_folders = []
    np.random.seed(42)
    if len(scale) != num_folders:
        scale = [1.0] * num_folders
    for i in range(num_folders):
        folder = os.path.join(temp_dir, "fov_" + str(i))
        os.makedirs(folder, exist_ok=True)
        data_folders.append(folder)
        for marker, std in zip(selected_markers, scale):
            if random:
                img = np.random.rand(256, 256) * std
            else:
                img = np.ones([256, 256])
            imwrite(
                os.path.join(folder, marker + ".tiff"),
                img,
            )
        imwrite(
            os.path.join(
                folder, "cell_segmentation.tiff"), np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
                ).repeat(64, axis=1).repeat(64, axis=0)
        )
    return data_folders


def prepare_cell_type_table():

    # prepare cell_table
    cell_type_table = pd.DataFrame(
        {
            "SampleID": ["fov_0"] * 15 + ["fov_1"] * 15 + ["fov_2"] * 15 + ["fov_3"] * 15 +
            ["fov_4"] * 15 + ["fov_5"] * 15,
            "labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 6,
            "cluster_labels": ["stromal", "FAP", "NK", "NK", "NK"] * 3 * 3 +
            ["CD4", "CD14", "CD163", "CD163", "CD163"] * 3 * 3,
        }
    )
    return cell_type_table


def test_calculate_normalization_matrix():

    # instantiate data_prep, conversion_matrix and markers
    data_prep = prep_object()
    selected_markers = ["CD11c", "CD14", "CD56", "CD57"]
    scale = [1.0, 2.0, 8.512, 0.25]

    # create temporary folders with data and do tests
    with tempfile.TemporaryDirectory() as temp_dir:

        # check normalization_dict for different stochastic images
        data_folders = prepare_test_data_folders(
            4, temp_dir, selected_markers, random=True, scale=scale
        )
        data_prep = prep_object(
            normalization_dict_path=os.path.join(temp_dir, "norm_dict_test.json")
        )
        norm_dict = data_prep.calculate_normalization_matrix(
            data_folders=data_folders, selected_markers=selected_markers
        )

        # check if the normalization_dict has the correct values for stochastic images
        for marker, std in zip(selected_markers, scale):
            assert np.isclose(norm_dict[marker], std * 0.999, rtol=1e-3)

        # check if the normalization_dict is correctly written to the json file
        norm_dict_loaded = json.load(open(os.path.join(temp_dir, "norm_dict_test.json")))
        assert norm_dict_loaded == norm_dict

        # check if the normalization_dict has the correct keys
        assert set(selected_markers) == set(norm_dict.keys())


def test_load_and_check_input():

    with tempfile.TemporaryDirectory() as temp_dir:

        # create temporary folders with data for the tests
        conversion_matrix = prepare_conversion_matrix()
        conversion_matrix_path = os.path.join(temp_dir, "conversion_matrix.csv")
        conversion_matrix.to_csv(conversion_matrix_path, index=True)
        norm_dict = {"CD11c": 1.0, "CD4": 1.0, "CD56": 1.0, "CD57": 1.0}
        with open(os.path.join(temp_dir, "norm_dict.json"), "w") as f:
            json.dump(norm_dict, f)
        data_folders = prepare_test_data_folders(5, temp_dir, list(norm_dict.keys()) + ["XYZ"])
        cell_table_path = os.path.join(temp_dir, "cell_type_table.csv")
        cell_table = prepare_cell_type_table()
        cell_table.to_csv(cell_table_path, index=False)

        # CONVERSION MATRIX
        # check if conversion_matrix is loaded correctly in check_input
        data_prep = prep_object(
            data_dir=temp_dir,
            conversion_matrix_path=conversion_matrix_path,
            tf_record_path=temp_dir,
            cell_table_path=cell_table_path,
            normalization_dict_path=os.path.join(temp_dir, "norm_dict.json"),
            selected_markers="CD4",
        )
        data_prep.load_and_check_input()
        assert np.array_equal(data_prep.conversion_matrix, conversion_matrix)
        data_prep_working = copy.deepcopy(data_prep)

        # check if ValueError is raised when selected_markers not in conversion_matrix
        data_prep.selected_markers = ["XYZ"]
        with pytest.raises(ValueError, match="selected markers were found in list conversion"):
            data_prep.load_and_check_input()

        # NORMALIZATION DICT
        # check if the normalization_dict is loaded correctly in check_input
        # when normalization_dict_path is given to init
        data_prep = prep_object(
            data_dir=temp_dir,
            conversion_matrix_path=conversion_matrix_path,
            tf_record_path=temp_dir,
            normalization_dict_path=os.path.join(temp_dir, "norm_dict.json"),
            cell_table_path=cell_table_path,
        )
        data_prep.load_and_check_input()
        assert norm_dict == data_prep.normalization_dict

        # check if the normalization_dict is calculated in check_input when
        # data_dir but no normalization_dict_path is given to init
        # data_prep.data_dir = temp_dir
        data_prep.normalization_dict_path = None
        data_prep.load_and_check_input()
        assert norm_dict == data_prep.normalization_dict

        # check if ValueError is raised if selected_markers in conversion_matrix
        # but not in loaded normalization_dict
        conversion_matrix = pd.DataFrame(
            np.random.randint(0, 2, size=(6, 5)),
            columns=["CD11c", "CD14", "CD56", "CD57", "XYZ"],
            index=["stromal", "FAP", "NK", "CD4", "CD14", "CD163"],
        )
        conversion_matrix_path = os.path.join(temp_dir, "conversion_matrix.csv")
        conversion_matrix.to_csv(conversion_matrix_path, index=True)
        data_prep = copy.deepcopy(data_prep_working)
        data_prep.conversion_matrix_path = conversion_matrix_path
        data_prep.normalization_dict_path = os.path.join(temp_dir, "norm_dict.json")
        data_prep.selected_markers = ["XYZ"]
        with pytest.raises(ValueError, match="selected markers were found in list normalization"):
            data_prep.load_and_check_input()

        # check if FileNotFoundError is raised if data_folders and conversion_matrix_path are given
        # together with selected_markers were images are missing for in data_folders
        conversion_matrix = pd.DataFrame(
            np.random.randint(0, 2, size=(6, 6)),
            columns=["CD11c", "CD4", "CD56", "CD57", "XYZ", "ZYX"],
            index=["stromal", "FAP", "NK", "CD4", "CD14", "CD163"],
        )
        conversion_matrix_path = os.path.join(temp_dir, "conversion_matrix.csv")
        conversion_matrix.to_csv(conversion_matrix_path, index=True)
        data_prep = copy.deepcopy(data_prep_working)
        data_prep.selected_markers = ["ZYX"]
        data_prep.conversion_matrix_path = conversion_matrix_path
        data_prep.normalization_dict_path = None
        data_prep.data_folders = data_folders
        with pytest.raises(FileNotFoundError, match="Marker ZYX not found in data folders"):
            data_prep.load_and_check_input()

        # check if ValueError is raised when normalization quantile is not in [0,1]
        data_prep = copy.deepcopy(data_prep_working)
        data_prep.normalization_quantile = 1.1
        with pytest.raises(ValueError, match="normalization_quantile is not in"):
            data_prep.load_and_check_input()

        # CELL TYPE TABLE
        # check if cell_type_table is loaded correctly in check_input
        # when cell_type_table_path is given to init
        conversion_matrix.to_csv(conversion_matrix_path, index=True)
        data_prep = copy.deepcopy(data_prep_working)
        data_prep.cell_type_table_path = cell_table_path
        data_prep.load_and_check_input()
        assert np.array_equal(cell_table, data_prep.cell_type_table)

        # check if ValueError is raised when cell_type_key not in cell_type_table
        data_prep.cell_type_key = "wrong_key"
        with pytest.raises(ValueError, match="The cell_type_key is not in the cell_type_table"):
            data_prep.load_and_check_input()

        # check if ValueError is raised when segment_label_key not in cell_type_table
        data_prep = copy.deepcopy(data_prep_working)
        data_prep.segment_label_key = "wrong_key"
        with pytest.raises(
            ValueError, match="The segment_label_key is not in the cell_type_table"
        ):
            data_prep.load_and_check_input()

        # check if ValueError is raised when sample_key not in cell_type_table
        data_prep = copy.deepcopy(data_prep_working)
        data_prep.sample_key = "wrong_key"
        with pytest.raises(ValueError, match="The sample_key is not in the cell_type_table"):
            data_prep.load_and_check_input()

        # check if ValueError is raised when sample_names in cell_type_table do not match
        # sample_names in data_folders
        data_prep = copy.deepcopy(data_prep_working)
        cell_table.SampleID[0] = "wrong_sample"
        cell_table_path_tmp = os.path.join(temp_dir, "cell_type_table_wrong_sample.csv")
        cell_table.to_csv(cell_table_path_tmp, index=False)
        data_prep.cell_table_path = cell_table_path_tmp
        with pytest.warns(UserWarning):
            data_prep.load_and_check_input()


def test_get_inst_binary_masks():

    instance_mask = np.zeros([256, 256], dtype=np.uint16)
    instance_mask[0:32, 0:32] = 1
    instance_mask[0:32, 32:64] = 2
    instance_mask[0:32, 64:96] = 3
    instance_mask[32:64, 0:32] = 4
    instance_mask[64:96, 64:96] = 5

    instance_mask_eroded = np.zeros([256, 256], dtype=np.uint8)
    instance_mask_eroded[0:31, 0:31] = 1
    instance_mask_eroded[0:31, 33:63] = 1
    instance_mask_eroded[0:31, 65:95] = 1
    instance_mask_eroded[33:63, 0:31] = 1
    instance_mask_eroded[65:95, 65:95] = 1
    with tempfile.TemporaryDirectory() as temp_dir:

        # check if the instance_mask is correctly loaded
        imwrite(os.path.join(temp_dir, "cell_segmentation.tiff"), instance_mask)
        data_prep = prep_object()
        loaded_binary_img, loaded_img = data_prep.get_inst_binary_masks(data_folder=temp_dir)
        assert np.array_equal(np.squeeze(loaded_img), instance_mask)

        # check if binary mask is binarized correctly
        assert np.array_equal(np.unique(loaded_binary_img), np.array([0, 1]))

        # check if binary mask is eroded correctly
        assert np.array_equal(np.squeeze(loaded_binary_img), instance_mask_eroded)

    # check if it works with naming convention function
    with tempfile.TemporaryDirectory() as temp_dir:
        segmentation_path = os.path.join(temp_dir, "segmentations")
        samples_path = os.path.join(temp_dir, "samples", "sample_1")
        os.mkdir(segmentation_path)
        imwrite(os.path.join(segmentation_path, "sample_1.tiff"), instance_mask)

        def naming_convention(sample_name):
            return os.path.join(segmentation_path, sample_name + ".tiff")

        data_prep = prep_object(segmentation_naming_convention=naming_convention)
        loaded_binary_img, loaded_img = data_prep.get_inst_binary_masks(data_folder=samples_path)

        # check if the instance_mask is correctly loaded
        assert np.array_equal(np.squeeze(loaded_img), instance_mask)


def test_get_composite_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, data_folders, _, _ = prep_object_and_inputs(temp_dir)

        # add channels to data_prep to ensure that the normalization_dict is constructed for them
        data_prep.nuclei_channels = ["CD56", "CD57"]
        data_prep.membrane_channels = ["CD11c", "CD4"]
        data_prep.normalization_dict_path = None
        data_prep.load_and_check_input()
        nuclei_img = data_prep.get_composite_image(
            data_folders[0], channels=data_prep.nuclei_channels
        )
        membrane_img = data_prep.get_composite_image(
            data_folders[0], channels=data_prep.membrane_channels
        )

        # load the images and check against the composite image
        nuclei_img_loaded = []
        for chan in data_prep.nuclei_channels:
            img = imread(os.path.join(data_folders[0], chan + ".tiff")).astype(np.float32)
            img /= data_prep.normalization_dict[chan]
            img = img.clip(0, 1)
            nuclei_img_loaded.append(img)
        nuclei_img_loaded = np.mean(np.stack(nuclei_img_loaded, axis=-1), axis=-1)[..., np.newaxis]
        assert np.array_equal(nuclei_img, nuclei_img_loaded)

        membrane_img_loaded = []
        for chan in data_prep.membrane_channels:
            img = imread(os.path.join(data_folders[0], chan + ".tiff")).astype(np.float32)
            img /= data_prep.normalization_dict[chan]
            img = img.clip(0, 1)
            membrane_img_loaded.append(img)
        membrane_img_loaded = np.mean(
            np.stack(membrane_img_loaded, axis=-1), axis=-1
        )[..., np.newaxis]
        assert np.array_equal(membrane_img, membrane_img_loaded)

        # test if the function works for a single channel
        single_channel_img = data_prep.get_composite_image(data_folders[0], channels=["CD4"])
        single_channel_img_loaded = imread(
            os.path.join(data_folders[0], "CD4.tiff")
        ).astype(np.float32)
        single_channel_img_loaded /= data_prep.normalization_dict["CD4"]
        single_channel_img_loaded = single_channel_img_loaded.clip(0, 1)[..., np.newaxis]
        assert np.array_equal(single_channel_img, single_channel_img_loaded)


def test_get_marker_activity():

    data_prep = prep_object()
    cell_table = prepare_cell_type_table()
    conversion_matrix = prepare_conversion_matrix()
    data_prep.cell_type_table = cell_table
    marker = "CD4"
    sample_name = "fov_1"
    fov_1_subset = cell_table[cell_table.SampleID == sample_name]
    data_prep.sample_subset = fov_1_subset
    conversion_matrix.index = conversion_matrix.index.str.lower()
    data_prep.conversion_matrix = conversion_matrix
    fov_1_subset["cluster_labels"] = fov_1_subset["cluster_labels"].str.lower()
    marker_activity, _ = data_prep.get_marker_activity(sample_name, marker)
    # check if the we get marker_acitivity for all labels in the fov_1 subset
    assert np.array_equal(marker_activity.labels, fov_1_subset.labels)

    # check if the df has the right marker activity values for a given cell
    for i in range(len(fov_1_subset.labels)):
        assert (
            marker_activity.activity.values[i]
            == conversion_matrix.loc[fov_1_subset.cluster_labels.values[i], "CD4"]
        )


def test_get_marker_activity_mask():

    data_prep = prep_object()
    marker_activity = pd.DataFrame(
        {
            "labels": [1, 2, 5, 7, 9, 11],
            "activity": [1, 0, 0, 0, 0, 1],
        }
    )
    instance_mask = np.zeros([256, 256], dtype=np.uint16)
    instance_mask[0:32, 0:32] = 1
    instance_mask[0:32, 32:64] = 2
    instance_mask[0:32, 64:96] = 5
    instance_mask[32:64, 0:32] = 7
    instance_mask[64:96, 64:96] = 9
    instance_mask[128:160, 128:160] = 11
    binary_mask = (instance_mask > 0).astype(np.uint8)
    marker_activity_mask = data_prep.get_marker_activity_mask(
        instance_mask, binary_mask, marker_activity
    )

    # check if returned spatial dimensions are correct
    assert marker_activity_mask.shape == instance_mask.shape

    # check if returned marker activity values are correct
    for i in np.unique(instance_mask):
        if i == 0:
            continue
        assert (
            marker_activity_mask[instance_mask == i]
            == int(marker_activity.activity[marker_activity.labels == i])
        ).all()


@pytest.mark.parametrize("tile_size", [[256, 256], [128, 256], [256, 128], [128, 128]])
def test_tile_example(tile_size):
    marker_activity = pd.DataFrame(
        {
            "labels": np.array([1, 2, 5, 7, 9, 11], dtype=np.uint16),
            "activity": [1, 0, 0, 0, 0, 1],
            "cell_type": ["T cell", "B cell", "T cell", "B cell", "T cell", "B cell"],
        }
    )
    instance_mask = np.zeros([512, 512, 1], dtype=np.uint16)
    instance_mask[0:32, 0:32] = 1
    instance_mask[0:32, 32:64] = 2
    instance_mask[0:32, 64:96] = 5
    instance_mask[476:, 476:] = 7
    instance_mask[444:476, 476:] = 9
    instance_mask[476:, 444:476] = 11

    example = {
        "mplex_img": np.random.rand(512, 512, 3).astype(np.float32),
        "binary_mask": np.random.randint(0, 2, [512, 512, 1]).astype(np.uint8),
        "nuclei_img": np.random.rand(512, 512, 1).astype(np.float32),
        "membrane_img": np.random.rand(512, 512, 1).astype(np.float32),
        "instance_mask": instance_mask,
        "marker_activity_mask": np.random.randint(0, 2, [512, 512, 21]).astype(np.uint8),
        "dataset": "test_dataset",
        "platform": "mibi",
        "activity_df": marker_activity,
        "marker": "CD11c",
    }
    data_prep = prep_object(tile_size=tile_size, stride=tile_size)
    tiled_examples = data_prep.tile_example(example)

    # check if the correct number of tiles got returned
    assert len(tiled_examples) == int(np.floor(512 / tile_size[0]) * np.floor(512 / tile_size[1]))

    # check if the correct spatial dimensions got returned and dtype is correct
    for key in ["mplex_img", "binary_mask", "instance_mask", "marker_activity_mask"]:
        assert tiled_examples[0][key].dtype == example[key].dtype
        assert list(tiled_examples[0][key].shape[:2]) == tile_size
        assert list(tiled_examples[-1][key].shape[:2]) == tile_size

    # check if the correct values for non spatial keys got returned
    for key in ["dataset", "platform", "marker"]:
        assert tiled_examples[0][key] == example[key]
        assert tiled_examples[-1][key] == example[key]

    # check if the correct tiles are returned
    for key in ["mplex_img", "binary_mask", "instance_mask", "marker_activity_mask"]:
        assert np.array_equal(
            tiled_examples[0][key], example[key][:tile_size[0], :tile_size[1], ...]
        )
        max_h = example[key].shape[0] % tile_size[0] or tile_size[0]
        max_w = example[key].shape[1] % tile_size[1] or tile_size[1]
        assert np.array_equal(
            tiled_examples[-1][key][:max_h, :max_w, ...], example[key][-max_h:, -max_w:, ...]
        )

    # check if marker_activity contains the correct subset of labels
    assert np.array_equal(
        np.unique(tiled_examples[0]["activity_df"].labels),
        np.array([1, 2, 5], dtype=np.uint16)
    )

    # check if channel gets added to images without channels
    example["instance_mask"] = np.squeeze(example["instance_mask"], axis=-1)
    tiled_examples = data_prep.tile_example(example)
    assert tiled_examples[0]["instance_mask"].ndim == 3
    assert tiled_examples[1]["instance_mask"].ndim == 3

    # check if tiles are excluded if they don't contain any cells when setting
    # exclude_background_tiles=True
    data_prep.exclude_background_tiles = True
    tiled_examples = data_prep.tile_example(example)
    assert len(tiled_examples) < int(np.ceil(512 / tile_size[0]) * np.ceil(512 / tile_size[1]))
    for example in tiled_examples:
        assert example["activity_df"].activity.sum() > 0


def test_prepare_example():
    data_prep = prep_object()
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, data_folders, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.sample_subset = data_prep.cell_type_table[
            data_prep.cell_type_table.SampleID == os.path.basename(data_folders[0])
        ]
        data_prep.binary_mask = np.random.randint(0, 2, [256, 256, 1]).astype(np.uint8)
        data_prep.instance_mask = np.zeros([256, 256, 1], dtype=np.uint16)
        data_prep.nuclei_img = np.zeros([256, 256, 1], dtype=np.uint16)
        data_prep.membrane_img = np.zeros([256, 256, 1], dtype=np.uint16)
        example = data_prep.prepare_example(data_folders[0], marker="CD4")
        # check keys in example
        assert set(example.keys()) == set(
            [
                "mplex_img", "binary_mask", "instance_mask", "imaging_platform", "nuclei_img",
                "membrane_img", "marker_activity_mask", "dataset", "marker", "folder_name",
                "activity_df", "tissue_type"
            ]
        )

        # check correct normalization of mplex_img
        assert np.isclose(np.quantile(example["mplex_img"], 0.999), 1.0, rtol=1e-2)
        assert example["mplex_img"].min() >= 0.0

        # check if all images are 3 dimensional
        for key in ["mplex_img", "binary_mask", "instance_mask", "marker_activity_mask"]:
            assert example[key].ndim == 3


def test_serialize_example():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.sample_subset = data_prep.cell_type_table[
            data_prep.cell_type_table.SampleID == os.path.basename("fov_1")
        ]
        data_prep.binary_mask = np.random.randint(0, 2, [256, 256, 1]).astype(np.uint8)
        data_prep.instance_mask = np.zeros([256, 256, 1], dtype=np.uint16)
        data_prep.nuclei_img = np.zeros([256, 256, 1], dtype=np.uint16)
        data_prep.membrane_img = np.zeros([256, 256, 1], dtype=np.uint16)
        example = data_prep.prepare_example(os.path.join(temp_dir, "fov_1"), marker="CD4")
        serialized_example = data_prep.serialize_example(copy.deepcopy(example))
        deserialized_dict = tf.io.parse_single_example(serialized_example, feature_description)
        parsed_example = parse_dict(deserialized_dict)

        # compare parsed example to original example
        # check if parsed example has the correct keys
        assert set(parsed_example.keys()) == set(example.keys())

        # check string features
        for key in ["dataset", "marker", "imaging_platform", "folder_name", 'tissue_type']:
            assert example[key] == parsed_example[key]
        # check df features
        for key in ["activity_df"]:
            assert example[key].equals(parsed_example[key])
        # check image features
        for key in ["binary_mask", "marker_activity_mask", "instance_mask"]:
            assert np.array_equal(example[key], parsed_example[key].numpy())
        # check if mplex_img (float32) is correctly reconstructed from uint16 png
        assert np.allclose(example["mplex_img"], parsed_example["mplex_img"].numpy(), atol=1e-3)


def test_make_tf_record():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_prep, _, _, _ = prep_object_and_inputs(temp_dir)
        data_prep.tf_record_path = temp_dir
        data_prep.make_tf_record()
        tf_record_path = os.path.join(temp_dir, data_prep.dataset + ".tfrecord")
        # check if tf record was created
        assert os.path.exists(tf_record_path)

        # check if tf record has the right number of examples
        dataset = tf.data.TFRecordDataset(tf_record_path)
        num_examples = 0
        for string_record in dataset:
            num_examples += 1
        assert num_examples == 5

        # parse samples and compare to original example
        deserialized_dict = tf.io.parse_single_example(string_record, feature_description)
        parsed_dict = parse_dict(deserialized_dict)
        example = data_prep.prepare_example(
            os.path.join(temp_dir, parsed_dict["folder_name"]), marker="CD4"
        )

        # check if serialized example has the right keys
        assert set(parsed_dict.keys()) == set(example.keys())
        # check string features
        for key in ["dataset", "marker", "imaging_platform", "folder_name", "tissue_type"]:
            assert example[key] == parsed_dict[key]
        # check df features, empty df is also okay
        for key in ["activity_df"]:
            assert example[key].equals(parsed_dict[key]) or example[key].empty
        # check image features
        for key in ["binary_mask", "marker_activity_mask", "instance_mask"]:
            assert np.array_equal(example[key], parsed_dict[key].numpy())
        # check if mplex_img (float32) is correctly reconstructed from uint16 png
        assert np.allclose(example["mplex_img"], parsed_dict["mplex_img"].numpy(), atol=1e-3)

        # remove tiled-tfrecord and check if everything works with tile_size = None
        os.remove(tf_record_path)
        data_prep.tile_size = None
        data_prep.make_tf_record()
        # check if tf record was created
        assert os.path.exists(tf_record_path)

        # check if tf record has the right number of examples
        dataset = tf.data.TFRecordDataset(tf_record_path)
        num_examples = 0
        for string_record in dataset:
            num_examples += 1
        assert num_examples == 5
