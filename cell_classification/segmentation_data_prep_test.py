import pytest
from segmentation_data_prep import SegmentationTFRecords

data_prep = SegmentationTFRecords(
    data_folders="path_to_folders",
    cell_table_path="path_to_cell_table",
    conversion_matrix_path="path_to_conversion_matrix",
    imaging_platform="ImagingPlatform",
    dataset="Dataset",
    tile_size=[256, 256],
    tf_record_path="path_to_tf_record",
)
