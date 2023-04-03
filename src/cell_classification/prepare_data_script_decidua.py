import os

from segmentation_data_prep import SegmentationTFRecords

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def naming_convention(fname):
    return os.path.join(
        "E:/angelo_lab/data/decidua/segmentation_data/",
        fname + "_segmentation_labels.tiff",
    )


data_prep = SegmentationTFRecords(
    data_dir=os.path.normpath("E:/angelo_lab/data/decidua/image_data"),
    cell_table_path=os.path.normpath(
        "E:/angelo_lab/data/decidua/"
        "Supplementary_table_3_single_cells_updated.csv"
    ),
    conversion_matrix_path=os.path.normpath(
        "E:/angelo_lab/data/decidua/conversion_matrix.csv"
    ),
    imaging_platform="MIBI",
    dataset="decidua_erin",
    tissue_type="decidua",
    nuclei_channels=["H3"],
    membrane_channels=["VIM", "HLAG", "CD3", "CD14", "CD56"],
    tile_size=[256, 256],
    stride=[240, 240],
    tf_record_path=os.path.normpath("E:/angelo_lab/data/decidua"),
    normalization_dict_path=os.path.normpath(
        "E:/angelo_lab/data/decidua/normalization_dict.json"
    ),
    selected_markers=[
        "CD45", "CD14", "HLADR", "CD11c", "DCSIGN", "CD68", "CD206", "CD163", "CD3", "Ki67", "IDO",
        "CD8", "CD4", "CD16", "CD56", "CD57", "SMA", "VIM", "CD31", "CK7", "HLAG", "FoxP3", "PDL1",
    ],
    normalization_quantile=0.999,
    cell_type_key="lineage",
    sample_key="Point",
    segmentation_naming_convention=naming_convention,
    segment_label_key="cell_ID_in_Point",
    exclude_background_tiles=True,
    img_suffix=".tif",
)

data_prep.make_tf_record()
