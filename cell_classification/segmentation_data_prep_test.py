import os
import pytest
import tempfile
import numpy as np
from tifffile import imwrite
from segmentation_data_prep import SegmentationTFRecords


def prep_object():
    data_prep = SegmentationTFRecords(
        data_folders="path_to_folders",
        cell_table_path="path_to_cell_table",
        conversion_matrix_path="path_to_conversion_matrix",
        imaging_platform="ImagingPlatform",
        dataset="Dataset",
        tile_size=[256, 256],
        tf_record_path="path_to_tf_record",
    )
    return data_prep


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


def test_prepare_example():
    data_prep = prep_object()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_img = np.random.rand(256, 256)
        imwrite(os.path.join(temp_dir, "CD8.tiff"), test_img)
        data_prep.prepare_example(temp_dir, "CD8")
