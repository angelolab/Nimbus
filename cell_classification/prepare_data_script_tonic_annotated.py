import os
from simple_data_prep import SimpleTFRecords
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def naming_convention(fname):
    return os.path.join(
        "E:/angelo_lab/data/TONIC/raw/segmentation_data/deepcell_output",
        fname + "_feature_0.tif"
    )


data_prep = SimpleTFRecords(
    data_dir=os.path.normpath(
        "C:/Users/Lorenz/OneDrive - Charité - Universitätsmedizin Berlin/cell_classification/" +
        "data_annotation/tonic/raw"
    ),
    cell_table_path=os.path.normpath(
        "C:/Users/Lorenz/OneDrive - Charité - Universitätsmedizin Berlin/cell_classification/" +
        "data_annotation/tonic/ground_truth.csv"
    ),
    imaging_platform="MIBI",
    dataset="TONIC",
    tissue_type="TNBC",
    nuclei_channels=["H3K27me3", "H3K9ac"],
    membrane_channels=["CD45", "ECAD", "CD14", "CD38", "CK17"],
    selected_markers=[
        "Calprotectin", "CD14", "CD163", "CD20", "CD3", "CD31", "CD4", "CD45", "CD56", "CD68",
        "CD8", "ChyTr", "CK17", "Collagen1", "ECAD", "FAP", "Fibronectin", "FOXP3", "HLADR", "SMA",
        "VIM"
    ],
    tile_size=[256, 256],
    stride=[240, 240],
    tf_record_path=os.path.normpath("E:/angelo_lab/data/TONIC/annotated"),
    normalization_dict_path=os.path.normpath(
      "E:/angelo_lab/data/TONIC/normalization_dict.json"
    ),
    normalization_quantile=0.999,
    sample_key="fov",
    segmentation_fname="cell_segmentation",
    segmentation_naming_convention=naming_convention,
    segment_label_key="labels",
    exclude_background_tiles=True,
    gt_suffix="",
)

data_prep.make_tf_record()
