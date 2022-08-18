from calendar import c
from genericpath import exists
import os
import pytest
import tempfile
import numpy as np
import pandas as pd
import json
from tifffile import imwrite
from segmentation_data_prep import SegmentationTFRecords


def prep_object():
    with tempfile.TemporaryDirectory() as temp_dir:
        norm_dict = {"CD8": 1.0, "CD4": 1.0}
        norm_dict_path = os.path.join(temp_dir, "norm_dict.json")
        json.dump(norm_dict, open(norm_dict_path, "w"))
        conversion_matrix = prepare_conversion_matrix()
        conversion_matrix_path = os.path.join(temp_dir, "conversion_matrix.csv")
        conversion_matrix.to_csv(conversion_matrix_path, index=False)
        data_prep = SegmentationTFRecords(
            data_folders=["list", "of", "data", "folders"],
            cell_table_path="list_to_cell_table_path",
            conversion_matrix_path=conversion_matrix_path,
            imaging_platform="imaging_platform",
            dataset="dataset",
            tile_size=[256, 256],
            tf_record_path=os.path.join(temp_dir, "tf_record_path"),
            normalization_dict_path=norm_dict_path,
        )
    return data_prep


def prepare_conversion_matrix():
    col_names = [
        "cluster_labels",
        "CD11c",
        "CD14",
        "CD56",
        "CD57",
        "CD163",
        "CD20",
        "CD3",
        "CD31",
        "CD38",
        "CD4",
        "CD45",
        "CD45RB",
        "CD45RO",
        "CD86",
    ]
    row_names = [
        "stromal",
        "FAP",
        "NK",
        "CD4T",
        "CD14",
        "CD163",
        "CD8T",
        "CD3_DN",
        "CD20",
        "CD20",
        "SMA",
        "HLADR",
        "CD45",
    ]
    conversion_matrix = pd.DataFrame(
        np.random.randint(0, 2, size=(len(row_names), len(col_names))),
        columns=col_names,
    )
    conversion_matrix.cluster_labels = row_names
    return conversion_matrix


def test_get_image():
    data_prep = prep_object()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_img_1 = np.random.rand(256, 256)
        test_img_2 = np.random.rand(256, 256)
        imwrite(os.path.join(temp_dir, "CD8.tiff"), test_img_1)
        imwrite(os.path.join(temp_dir, "CD4.tiff"), test_img_2)
        CD8_img = data_prep.get_image(data_folder=temp_dir, marker="CD8")
        CD4_img = data_prep.get_image(data_folder=temp_dir, marker="CD4")
        assert np.array_equal(test_img_1, CD8_img)
        assert np.array_equal(test_img_2, CD4_img)
        assert not np.array_equal(CD8_img, CD4_img)


def prepare_test_data_folders(
    num_folders, temp_dir, selected_markers, random=False, scale=1.0
):
    data_folders = []
    for i in range(num_folders):
        folder = os.path.join(temp_dir, "fov_1" + str(i))
        os.mkdir(folder)
        data_folders.append(folder)
        for marker in selected_markers:
            if random:
                img = np.random.rand(256, 256) * scale
            else:
                img = np.ones([256, 256])
            imwrite(
                os.path.join(temp_dir, "fov_1" + str(i), marker + ".tiff"),
                img,
            )
    return data_folders


def test_calculate_normalization_matrix():
    # instantiate data_prep and conversion_matrix
    data_prep = prep_object()
    conversion_matrix = prepare_conversion_matrix()
    # get markers from there
    selected_markers = list(conversion_matrix.columns)
    selected_markers.remove("cluster_labels")
    with tempfile.TemporaryDirectory() as temp_dir:
        # create temporary folders with data
        data_folders = prepare_test_data_folders(5, temp_dir, selected_markers)
        norm_dict = data_prep.calculate_normalization_matrix(
            data_folders=data_folders,
            normalization_dict_path=os.path.join(temp_dir, "norm_dict_test.json"),
            normalization_quantile=0.99,
            selected_markers=selected_markers,
        )
        norm_dict_loaded = json.load(
            open(os.path.join(temp_dir, "norm_dict_test.json"))
        )
        # check if the normalization_dict is correctly written to the json file
        assert norm_dict_loaded == norm_dict
        # check if the normalization_dict has the correct values
        for marker in norm_dict.keys():
            assert norm_dict[marker] == 1.0
        # check if the normalization_dict has the correct keys
        for marker in selected_markers:
            assert marker in norm_dict.keys()
        # check if the normalization_dict works correctly when
        # normalization_dict_path is given to init
        conversion_matrix = prepare_conversion_matrix()
        conversion_matrix_path = os.path.join(temp_dir, "conversion_matrix.csv")
        conversion_matrix.to_csv(conversion_matrix_path, index=False)
        data_prep = SegmentationTFRecords(
            data_folders=["list", "of", "data", "folders"],
            cell_table_path="list_to_cell_table_path",
            conversion_matrix_path=conversion_matrix_path,
            imaging_platform="imaging_platform",
            dataset="dataset",
            tile_size=[256, 256],
            tf_record_path=os.path.join(temp_dir, "tf_record_path"),
            normalization_dict_path=os.path.join(temp_dir, "norm_dict_test.json"),
        )
        assert norm_dict == data_prep.normalization_dict
        # check if the normalization_dict works correctly when
        # data_folders but no normalization_dict_path is given to init
        data_prep = SegmentationTFRecords(
            data_folders=data_folders,
            cell_table_path="list_to_cell_table_path",
            conversion_matrix_path=conversion_matrix_path,
            imaging_platform="imaging_platform",
            dataset="dataset",
            tile_size=[256, 256],
            tf_record_path=os.path.join(temp_dir, "tf_record_path"),
        )
        assert norm_dict == data_prep.normalization_dict
        norm_dict_loaded = json.load(
            open(os.path.join(temp_dir, "tf_record_path", "normalization_dict.json"))
        )
        # check if the normalization_dict is correctly written to the json file
        assert norm_dict_loaded == norm_dict

    # check if the normalization_dict has the correct values for stochastic images
    for scale in [0.5, 9.132]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # create temporary folders with data
            data_folders = prepare_test_data_folders(
                5, temp_dir, selected_markers, random=True, scale=scale
            )
            norm_dict = data_prep.calculate_normalization_matrix(
                data_folders=data_folders,
                normalization_dict_path=os.path.join(temp_dir, "norm_dict_test.json"),
                normalization_quantile=0.99,
                selected_markers=selected_markers,
            )
            for marker in norm_dict.keys():
                assert norm_dict[marker] - 1 / (0.5 * scale) < 0.001


def test_instance_mask():
    instance_mask = np.zeros([256, 256], dtype=np.uint16)
    instance_mask[0:32, 0:32] = 1
    instance_mask[0:32, 32:64] = 2
    instance_mask[0:32, 64:96] = 3
    instance_mask[32:64, 0:32] = 4
    instance_mask[64:96, 64:96] = 5

    instance_mask_full = np.zeros([256, 256], dtype=np.uint16)
    instance_mask_full[32:64, 32:64] = 9

    instance_mask_eroded = np.zeros([256, 256], dtype=np.uint16)
    instance_mask_eroded[33:63, 33:63] = 1
    with tempfile.TemporaryDirectory() as temp_dir:
        # check if the instance_mask is correctly loaded
        imwrite(os.path.join(temp_dir, "cell_segmentation.tiff"), instance_mask)
        imwrite(os.path.join(temp_dir, "weird_name_segmentation.tiff"), instance_mask)
        imwrite(
            os.path.join(temp_dir, "cell_segmentation_full.tiff"), instance_mask_full
        )
        data_prep = prep_object()
        loaded_binary_img, loaded_img = data_prep.get_instance_mask(
            data_folder=temp_dir, cell_mask_key="cell_segmentation"
        )
        assert np.array_equal(loaded_img, instance_mask)
        # check if binary mask is binarized correctly
        assert np.min(loaded_binary_img) == 0
        assert np.max(loaded_binary_img) == 1
        # check if binary mask is eroded correctly
        loaded_eroded_img, loaded_img = data_prep.get_instance_mask(
            data_folder=temp_dir, cell_mask_key="cell_segmentation_full"
        )
        assert np.array_equal(loaded_eroded_img, instance_mask_eroded)


def test_prepare_example():
    data_prep = prep_object()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_img = np.random.rand(256, 256)
        imwrite(os.path.join(temp_dir, "CD8.tiff"), test_img)
        imwrite(os.path.join(temp_dir, "cell_segmentation.tiff"), np.ones([256, 256]))
        data_prep.prepare_example(temp_dir, "CD8")
