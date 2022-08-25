from tkinter import E
from tifffile import imread
from skimage.segmentation import find_boundaries
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import xarray
from tqdm import tqdm
import json
from ark.utils.misc_utils import verify_in_list
from ark.utils.io_utils import list_folders, validate_paths
import re


class SegmentationTFRecords:
    """Prepares the data for the segmentation model"""

    def __init__(
        self, data_dir, cell_table_path, conversion_matrix_path, imaging_platform, dataset,
        tile_size, tf_record_path, selected_markers=None, normalization_dict_path=None,
        normalization_quantile=0.99, cell_type_key="cluster_labels", sample_key="SampleID",
        segmentation_fname="cell_segmentation", segment_label_key="labels",
    ):
        """Initializes SegmentationTFRecords and loads everything except the images

        Args:
            data_dir str:
                Path where the data is stored
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
            segmentation_fname (str):
                The filename in the data_folder that contains the cell instance segmentation
            segment_label_key (str):
                The key in the cell_table.csv that contains the cell segment labels
        """
        self.selected_markers = selected_markers
        self.data_dir = data_dir
        self.normalization_dict_path = normalization_dict_path
        self.conversion_matrix_path = conversion_matrix_path
        self.normalization_quantile = normalization_quantile
        self.segmentation_fname = segmentation_fname
        self.segment_label_key = segment_label_key
        self.sample_key = sample_key
        self.dataset = dataset
        self.imaging_platform = imaging_platform
        self.tf_record_path = tf_record_path
        self.cell_type_key = cell_type_key
        self.cell_table_path = cell_table_path

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
        img = imread(os.path.join(data_folder, marker + ".tiff"))
        return img

    def get_inst_binary_masks(self, data_folder):
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
        instance_mask = imread(os.path.join(data_folder, self.segmentation_fname + ".tiff"))
        edge = find_boundaries(instance_mask, mode="inner").astype(np.uint8)
        interior = np.logical_and(edge == 0, instance_mask > 0).astype(np.uint8)
        return interior, instance_mask

    def get_marker_activity(self, sample_name, conversion_matrix, marker):
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
        sample_subset = self.cell_type_table[self.cell_type_table.SampleID == sample_name]
        cell_types = sample_subset[self.cell_type_key].values

        df = pd.DataFrame(
            {
                "labels": sample_subset[self.segment_label_key],
                "activity": conversion_matrix.loc[cell_types, marker].values,
            }
        )
        return df

    def get_marker_activity_mask(self, instance_mask, binary_mask, marker_activity):
        """Makes a mask from the marker activity

        Args:
            instance_mask (np.array):
                The instance mask to make the marker activity mask for
            binary_mask (np.array):
                The binary mask to make the marker activity mask for
            marker_activity list:
                The marker activity that maps to the cell types
        Returns:
            np.array:
                The marker activity mask
        """
        out_mask = np.zeros_like(instance_mask)
        for label, activity in zip(marker_activity.labels, marker_activity.activity):
            out_mask[instance_mask == label] = activity
        out_mask[binary_mask == 0] = 0
        return out_mask

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
        binary_mask, instance_mask = self.get_inst_binary_masks(data_folder)

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
            "imaging_platform": self.imaging_platform,
            "marker_activity_mask": marker_activity_mask.astype(np.uint8),
            "dataset": self.dataset,
            "marker": marker,
            "cell_types": cell_types,
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

    def load_and_check_input(self):
        """Checks the input for correctness"""
        # make tfrecord path
        os.makedirs(self.tf_record_path, exist_ok=True)

        # DATA DIR
        validate_paths(self.data_dir)
        self.data_folders = [
            os.path.join(self.data_dir, folder) for folder in list_folders(self.data_dir)
        ]

        # CONVERSION MATRIX
        # read the file
        validate_paths(self.conversion_matrix_path)
        self.conversion_matrix = pd.read_csv(self.conversion_matrix_path)

        # check if markers were selected or take all markers from conversion matrix
        if self.selected_markers is None:
            self.selected_markers = list(self.conversion_matrix.columns)

        # check if selected markers are in conversion matrix
        verify_in_list(
            selected_markers=self.selected_markers,
            conversion_matrix_columns=self.conversion_matrix.columns,
        )

        # check if selected markers are in data folders
        for marker in self.selected_markers:
            exists = False
            for folder in self.data_folders:
                if os.path.exists(os.path.join(folder, marker + ".tiff")):
                    exists = True
                    break
            if not exists:
                raise FileNotFoundError("Marker {} not found in data folders".format(marker))

        # NORMALIZATION DICT
        # load or construct normalization dict
        if self.normalization_dict_path:
            validate_paths(self.normalization_dict_path)
            self.normalization_dict = json.load(open(self.normalization_dict_path, "r"))

            # check if selected markers are in normalization dict
            verify_in_list(
                selected_markers=self.selected_markers,
                normalization_dict_keys=self.normalization_dict.keys(),
            )
        else:
            # function raises a generic FileNotFoundError if selected_marker file
            # or segmentation_fname not in data_folders
            self.normalization_dict = self.calculate_normalization_matrix(
                self.data_folders,
                self.selected_markers,
            )
        # check if normalization_quantile is in [0, 1]
        if self.normalization_quantile < 0 or self.normalization_quantile > 1:
            raise ValueError("The normalization_quantile is not in [0, 1]")

        # CELL TYPE TABLE
        # load cell_types.csv
        self.cell_type_table = pd.read_csv(self.cell_table_path)

        # check if cell_type_key is in cell_type_table
        if self.cell_type_key not in self.cell_type_table.columns:
            raise ValueError("The cell_type_key is not in the cell_type_table")

        # check if segment_label_key is in cell_type_table
        if self.segment_label_key not in self.cell_type_table.columns:
            raise ValueError("The segment_label_key is not in the cell_type_table")

        # check if sample_key is in cell_type_table
        if self.sample_key not in self.cell_type_table.columns:
            raise ValueError("The sample_key is not in the cell_type_table")

        # check if sample_names in cell_type_table match sample_names in data_folder
        verify_in_list(
            sample_names=self.cell_type_table[self.sample_key].values,
            data_folders=list_folders(self.data_dir)
        )

    def make_tf_record(self, data_folders):
        """Iterates through the data_folders and loads, transforms and
        serializes a tfrecord example for each data_folder

        Args:
            tf_record_path (str):
                The path to the tf record to make
        """
        return None

    def calculate_normalization_matrix(self, data_folders, selected_markers):
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
        # iterate through the data_folders and calculate the quantiles
        quantiles = {}
        for data_folder in data_folders:
            for marker in selected_markers:
                img = self.get_image(data_folder, marker)
                if marker not in quantiles:
                    quantiles[marker] = []
                quantiles[marker].append(np.quantile(img, self.normalization_quantile))

        # calculate the normalization matrix
        normalization_matrix = {}
        for marker in selected_markers:
            normalization_matrix[marker] = 1.0 / np.mean(quantiles[marker])

        # check path and save the normalization matrix
        if not str(self.normalization_dict_path).endswith(".json"):
            self.normalization_dict_path = os.path.join(
                self.tf_record_path, "normalization_dict.json"
            )
        with open(self.normalization_dict_path, "w") as f:
            json.dump(normalization_matrix, f)
        self.normalization_dict = normalization_matrix
        return normalization_matrix
