import os

from simple_data_prep import SimpleTFRecords

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def naming_convention(fname):
    return os.path.join(
        "E:/angelo_lab/data/MSKCC_colon/segmentation",
        fname + "feature_0.ome.tif"
    )


data_prep = SimpleTFRecords(
    data_dir=os.path.normpath(
        "E:/angelo_lab/data/MSKCC_colon/raw_structured"
    ),
    cell_table_path=os.path.normpath(
        "E:/angelo_lab/data/MSKCC_colon/cell_table.csv"
    ),
    imaging_platform="Vectra",
    dataset="MSK_colon",
    tissue_type="colon",
    nuclei_channels=["DAPI"],
    membrane_channels=["CD3", "CD8", "ICOS", "panCK+CK7+CAM5.2"],
    tile_size=[256, 256],
    stride=[240, 240],
    tf_record_path=os.path.normpath("E:/angelo_lab/data/MSKCC_colon"),
    normalization_quantile=0.999,
    selected_markers=["CD3", "CD8", "Foxp3", "ICOS", "panCK+CK7+CAM5.2", "PD-L1"],
    sample_key="fov",
    segment_label_key="labels",
    segmentation_naming_convention=naming_convention,
    exclude_background_tiles=True,
    img_suffix=".ome.tif",
    # normalization_dict_path=os.path.normpath(
    #     "E:/angelo_lab/data/MSKCC_colon/normalization_dict.json"
    # ),
)

data_prep.make_tf_record()
