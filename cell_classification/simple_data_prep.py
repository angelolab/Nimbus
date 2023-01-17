import pandas as pd
from segmentation_data_prep import SegmentationTFRecords


class SimpleTFRecords(SegmentationTFRecords):
    """Prepares the data for the segmentation model"""
    def __init__(
        self, data_dir, cell_table_path, imaging_platform, dataset, tissue_type, tile_size, stride,
        tf_record_path, nuclei_channels, membrane_channels, selected_markers=None,
        normalization_dict_path=None, normalization_quantile=0.999,
        segmentation_naming_convention=None, segmentation_fname="cell_segmentation",
        exclude_background_tiles=False, resize=None, img_suffix=".tiff", sample_key="SampleID",
        segment_label_key="labels", gt_suffix="_gt",

    ):
        """Initializes SegmentationTFRecords and loads everything except the images

        Args:
            data_dir str:
                Path where the data is stored
            cell_table_path (str):
                Path to the cell table
            imaging_platform (str):
                The imaging platform used to generate the multiplexed imaging data
            dataset (str):
                The dataset where the imaging data comes from
            tissue_type (str):
                The tissue type of the imaging data
            tile_size list [int,int]:
                The size of the tiles to use for the segmentation model
            stride list [int,int]:
                The stride to tile the data
            tf_record_path (str):
                The path to the tf record to make
            selected_markers (list):
                The markers of interest for generating the tf record. If None, all markers
                mentioned in the conversion_matrix are used
            normalization_dict_path (str):
                Path to the normalization dict json
            normalization_quantile (float):
                The quantile to use for normalization of multiplexed data
            segmentation_naming_convention (Function):
                Function that takes in the sample name and returns the path to the segmentation
                .tiff file. Default is None, then it is assumed that the segmentation file is in
                the sample folder and is named $segmentation_fname.tiff
            exclude_background_tiles (bool):
                Whether to exclude the all tiles that only contain background
            resize (float):
                The resize factor to use for the images
            img_suffix (str):
                The suffix of the image files
            sample_key (str):
                The key in the cell table that contains the sample name
            segment_label_key (str):
                The key in the cell table that contains the segmentation labels
            gt_suffix (str):
                The suffix of the ground truth column in the cell_table
        """
        super().__init__(
            data_dir=data_dir, cell_table_path=cell_table_path, imaging_platform=imaging_platform,
            dataset=dataset, tile_size=tile_size, stride=stride, tf_record_path=tf_record_path,
            nuclei_channels=nuclei_channels, membrane_channels=membrane_channels,
            selected_markers=selected_markers, normalization_dict_path=normalization_dict_path,
            normalization_quantile=normalization_quantile, segmentation_fname=segmentation_fname,
            segmentation_naming_convention=segmentation_naming_convention, resize=resize,
            exclude_background_tiles=exclude_background_tiles, img_suffix=img_suffix,
            sample_key=sample_key, segment_label_key=segment_label_key, tissue_type=tissue_type,
            conversion_matrix_path=None,
        )
        self.selected_markers = selected_markers
        self.data_dir = data_dir
        self.normalization_dict_path = normalization_dict_path
        self.normalization_quantile = normalization_quantile
        self.segmentation_fname = segmentation_fname
        self.segment_label_key = segment_label_key
        self.sample_key = sample_key
        self.dataset = dataset
        self.tissue_type = tissue_type
        self.imaging_platform = imaging_platform
        self.tf_record_path = tf_record_path
        self.cell_table_path = cell_table_path
        self.tile_size = tile_size
        self.stride = stride
        self.segmentation_naming_convention = segmentation_naming_convention
        self.exclude_background_tiles = exclude_background_tiles
        self.resize = resize
        self.img_suffix = img_suffix
        self.gt_suffix = gt_suffix
        self.nuclei_channels = nuclei_channels
        self.membrane_channels = membrane_channels

    def get_marker_activity(self, sample_name, marker):
        """Gets the marker activity for the given labels
        Args:
            sample_name (str):
                The name of the sample
            conversion_matrix (pd.DataFrame):
                The conversion matrix to use for the lookup
            marker (str, list):
                The markers to get the activity for
        Returns:
            np.array:
                The marker activity for the given labels, 1 if the marker is active, 0
                otherwise and -1 if the marker is not specific enough to be considered active
        """

        df = pd.DataFrame(
            {
                "labels": self.sample_subset[self.segment_label_key],
                "activity": self.sample_subset[marker + self.gt_suffix],
            }
        )
        return df, None

    def check_additional_inputs(self):
        """Checks additional inputs for correctness"""
        pass
