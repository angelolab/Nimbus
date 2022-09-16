from segmentation_data_prep import SegmentationTFRecords
import os

os.listdir()
data_prep = SegmentationTFRecords(
    data_dir=os.path.normpath("C:/Users/lorenz/Downloads/Lorenz_example_data/mantis_directory"),
    cell_table_path=os.path.normpath(
        "C:/Users/lorenz/Downloads/Lorenz_example_data/cell_table.csv"
    ),
    conversion_matrix_path=os.path.normpath(
        "C:/Users/lorenz/Downloads/Lorenz_example_data/conversion_matrix.csv"
    ),
    imaging_platform="MIBI",
    dataset="TNBC",
    tile_size=[256, 256],
    stride=[240, 240],
    tf_record_path=os.path.normpath("C:/Users/lorenz/Downloads/Lorenz_example_data"),
    selected_markers=["CD8"],
    normalization_dict_path=None,
    normalization_quantile=0.99,
    cell_type_key="cluster_labels",
    sample_key="SampleID",
    segmentation_fname="cell_segmentation",
    segment_label_key="label",
)

data_prep.make_tf_record()
