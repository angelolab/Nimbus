from tifffile import imread
from skimage.morphology import erosion
import tensorflow as tf
import numpy as np
import os
import xarray
import sys

class SegmentationTFRecords:
    """Prepares the data for the segmentation model
    """
    def __init__(self,
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
                The markers of interest for generating the tf record. If None, all markers mentioned in
                the conversion_matrix are used
            normalization_dict_path (str):
                Path to the normalization dict json
            normalization_quantile (float):
                The quantile to use for normalization of multiplexed data
            cell_type_key (str):
                The key in the cell table that contains the cell type labels
            cell_mask_key (str):
                The key in the data_folder that contains the cell mask labels
        """
        pass
    
    def get_image(self, data_folder, marker):
        """Loads the images from a single data_folder

        Args:
            data_folder (str):
                The path to the data_folder
            marker (str):
                The marker shown in the image, e.g. "CD8" corresponds to file name "CD8.tiff" in the data_folder
        """
        pass

    def make_binary_mask(self, instance_mask):
        """Makes a binary mask from an instance mask by eroding it
        
        Args:
            instance_mask (np.array):
                The instance mask to make binary
        Returns:
            np.array:
                The binary mask
        """
        pass

    def get_cell_types(self, labels):
        """Gets the cell types from the cell table for the given labels
        Args:
            labels list:
                The labels to get the cell types for
        Returns:
            list:
                The cell types for the given labels
        """
        pass

    def get_marker_activity(self, cell_types, marker):
        """Gets the marker activity for the given labels
        Args:
            cell_types list:
                The cell types to get the marker activity for
            marker (str, list):
                The markers to get the activity for
        Returns:
            list:
                The marker activity for the given labels, 1 if the marker is active, 0 otherwise and 
                -1 if the marker is not specific enough to be considered active
        """
        pass

    def marker_activity_mask(self, instance_mask, cell_types, marker_activity):
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
        pass

    def make_tf_record(self, tf_record_path):
        """ Iterates through the data_folders and loads, transforms and serializes a
            tfrecord example for each data_folder
        
        Args:
            tf_record_path (str):
                The path to the tf record to make
        """
        pass

    def calculate_normalization_matrix(self, normalization_dict_path, normalization_quantile):
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
        pass