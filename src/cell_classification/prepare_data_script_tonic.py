import os

from cell_classification.segmentation_data_prep import SegmentationTFRecords

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def naming_convention(fname):
    return os.path.join(
        "E:/angelo_lab/data/TONIC/raw/segmentation_data/deepcell_output",
        fname + "_feature_0.tif"
    )


data_prep = SegmentationTFRecords(
    data_dir=os.path.normpath(
        "E:/angelo_lab/data/TONIC/raw/image_data/samples"
    ),
    cell_table_path=os.path.normpath(
        "E:/angelo_lab/data/TONIC/raw/" +
        "combined_cell_table_normalized_cell_labels_updated.csv"
    ),
    conversion_matrix_path=os.path.normpath(
        "E:/angelo_lab/data/TONIC/raw/TONIC_conversion_matrix.csv"
    ),
    imaging_platform="MIBI",
    dataset="TONIC",
    tissue_type="TNBC",
    nuclei_channels=["H3K27me3", "H3K9ac"],
    membrane_channels=["CD45", "ECAD", "CD14", "CD38", "CK17"],
    tile_size=[256, 256],
    stride=[240, 240],
    tf_record_path=os.path.normpath("E:/angelo_lab/data/TONIC"),
    # normalization_dict_path=os.path.normpath(
    #   "E:/angelo_lab/data/TONIC/normalization_dict.json"
    # ),
    normalization_quantile=0.999,
    cell_type_key="cell_meta_cluster",
    sample_key="fov",
    segmentation_fname="cell_segmentation",
    segmentation_naming_convention=naming_convention,
    segment_label_key="label",
    exclude_background_tiles=True,
)

data_prep.make_tf_record()
