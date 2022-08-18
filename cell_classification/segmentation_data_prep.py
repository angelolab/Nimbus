from tifffile import imread
from skimage.morphology import erosion
import tensorflow as tf
import numpy as np
import os
import xarray
from tqdm import tqdm
import json


class SegmentationTFRecords:
    """Prepares the data for the segmentation model"""

    def __init__(
        self,
        data_folders,
        cell_table_path,
        conversion_matrix_path,
        imaging_platform,
        dataset,
        tile_size,
        tf_record_path,
        selected_markers=None,
        normalization_dict_path=None,
        normalization_quantile=0.99,
        cell_type_key="cluster_labels",
        sample_key="SampleID",
        cell_mask_key="cell_segmentation",
    ):
        """Initializes SegmentationTFRecords and loads everything except the images

        Args:
            data_folders (list):
                List of folders containing the multiplexed imaging data
            cell_table_path (str):
                Path to the cell table
            conversion_matrix_path (str):
                Path to the conversion matrix
            imaging_platform (str):
                The imaging platform used to generate the multiplexed imaging data
            dataset (str):
                The dataset where the imaging data comes from
            tile_size list [int,int]:
                The size of the tiles to use for the segmentation model
            tf_record_path (str):
                The path to the tf record to make
            selected_markers (list):
                The markers of interest for generating the tf record. If None, all markers
                mentioned in the conversion_matrix are used
            normalization_dict_path (str):
                Path to the normalization dict json
            normalization_quantile (float):
                The quantile to use for normalization of multiplexed data
            cell_type_key (str):
                The key in the cell table that contains the cell type labels
            sample_key (str):
                The key in the cell table that contains the sample name
            cell_mask_key (str):
                The key in the data_folder that contains the cell mask labels
        """
        pass
        if normalization_dict_path is not None:
            self.normalization_dict = json.load(open(normalization_dict_path, "r"))
        else:
            self.normalization_dict = self.calculate_normalization_matrix(
                normalization_dict_path, normalization_quantile
            )
            self.cell_mask_key = cell_mask_key
            self.sample_key = sample_key
            self.dataset = dataset
            self.imaging_platform = imaging_platform

    def get_image(self, data_folder, marker):
        """Loads the images from a single data_folder

        Args:
            data_folder (str):
                The path to the data_folder
            marker (str):
                The marker shown in the image, e.g. "CD8" corresponds
                to file name "CD8.tiff" in the data_folder
        Returns:
            np.array:
                The multiplexed image
        """
        return np.zeros([500, 500])

    def get_instance_masks(self, data_folder, cell_mask_key):
        """Makes a binary mask from an instance mask by eroding it

        Args:
            data_folder (str):
                The path to the data_folder
            cell_mask_key (str):
                The key in the data_folder that contains the cell mask labels
        Returns:
            np.array:
                The binary mask
            np.array:
                The instance mask
        """
        return np.zeros([500, 500]), np.zeros([500, 500])

    def get_cell_types(self, sample_name):
        """Gets the cell types from the cell table for the given labels
        Args:
            sample_name (str):
                The name of the sample we use for look up in column sample_key
                in cell_type.csv
        Returns:
            df:
                The labels and corresponding cell types for the given sample
        """
        return None

    def get_marker_activity(self, cell_types, marker):
        """Gets the marker activity for the given labels
        Args:
            cell_types list:
                The cell types to get the marker activity for
            marker (str, list):
                The markers to get the activity for
        Returns:
            list:
                The marker activity for the given labels, 1 if the marker is active, 0
                otherwise and -1 if the marker is not specific enough to be considered active
        """
        return None

    def get_marker_activity_mask(self, instance_mask, cell_types, marker_activity):
        """Makes a mask from the marker activity

        Args:
            instance_mask (np.array):
                The instance mask to make the marker activity mask for
            cell_types list:
                The cell types of the cells in the instance_mask
            marker_activity list:
                The marker activity that maps to the cell types
        Returns:
            np.array:
                The marker activity mask
        """
        return np.zeros([500, 500])

    def prepare_example(self, data_folder, marker):
        """Prepares a tfrecord example for the given data_folder and marker
        Args:
            data_folder (str):
                The path to the data_folder
            marker (str):
                The marker shown in the image, e.g. "CD8" corresponds
                to file name "CD8.tiff" in the data_folder
            normalization_dict (dict):
                The normalization dict
        Returns:
            dict:
                Example dict
        """
        # load and normalize the multiplexed image and masks
        mplex_img = self.get_image(data_folder, marker)
        mplex_img /= self.normalization_dict[marker]
        binary_mask, instance_mask = self.get_instance_masks(
            data_folder, self.cell_mask_key
        )
        # get the cell types and marker activity mask
        cell_types = self.get_cell_types(data_folder)
        marker_activity = self.get_marker_activity(cell_types, marker)
        marker_activity_mask = self.get_marker_activity_mask(
            instance_mask, cell_types, marker_activity
        )
        return {
            "mplex_img": mplex_img.astype(np.float32),
            "binary_mask": binary_mask.astype(np.uint8),
            "instance_mask": instance_mask.astype(np.uint16),
            "cell_types": cell_types,
            "marker_activity_mask": marker_activity_mask.astype(np.uint8),
            "imaging_platform": self.imaging_platform,
            "dataset": self.dataset,
            "marker": marker,
        }

    def tile_example(example, tile_size):
        """Tiles the example into a grid of tiles
        Args:
            example (dict):
                The example to tile
        Returns:
            list:
                List of example dicts, one for each tile
        """
        return None

    def make_tf_record(self, data_folders, tf_record_path):
        """Iterates through the data_folders and loads, transforms and
        serializes a tfrecord example for each data_folder

        Args:
            tf_record_path (str):
                The path to the tf record to make
        """
        return None

    def calculate_normalization_matrix(
        self, normalization_dict_path, normalization_quantile
    ):
        """Calculates the normalization matrix for the given data if it does not exist
        Args:
            normalization_dict_path (str):
                The path to the normalization dict json if it exists
            normalization_quantile (float):
                The quantile to use for normalization of multiplexed data
        Returns:
            dict:
                The normalization dict
        """
        return {"CD8": 0.0}
