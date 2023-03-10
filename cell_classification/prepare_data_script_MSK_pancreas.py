import os
from simple_data_prep import SimpleTFRecords
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def naming_convention(fname):
    return os.path.join(
        "E:/angelo_lab/data/MSKCC_pancreas/segmentation",
        fname + "feature_0.ome.tif"
    )


data_prep = SimpleTFRecords(
    data_dir=os.path.normpath(
        "E:/angelo_lab/data/MSKCC_pancreas/raw_structured"
    ),
    cell_table_path=os.path.normpath(
        "E:/angelo_lab/data/MSKCC_pancreas/cell_table.csv"
    ),
    imaging_platform="Vectra",
    dataset="MSK_pancreas",
    tissue_type="pancreas",
    nuclei_channels=["DAPI"],
    membrane_channels=["CD8", "CD40", "CD40-L", "panCK"],
    tile_size=[256, 256],
    stride=[240, 240],
    tf_record_path=os.path.normpath("E:/angelo_lab/data/MSKCC_pancreas"),
    normalization_quantile=0.999,
    selected_markers=["CD8", "CD40", "CD40-L", "panCK", "PD-1", "PD-L1"],
    sample_key="fov",
    segment_label_key="labels",
    segmentation_naming_convention=naming_convention,
    exclude_background_tiles=True,
    img_suffix=".ome.tif",
    # normalization_dict_path=os.path.normpath(
    #     "E:/angelo_lab/data/MSKCC_pancreas/normalization_dict.json"
    # ),
)

data_prep.make_tf_record()
