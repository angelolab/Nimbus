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
import copy


class SegmentationTFRecords:
    """Prepares the data for the segmentation model"""

    def __init__(
        self, data_dir, cell_table_path, conversion_matrix_path, imaging_platform, dataset,
        tile_size, stride, tf_record_path, selected_markers=None, normalization_dict_path=None,
        normalization_quantile=0.999, cell_type_key="cluster_labels", sample_key="SampleID",
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
        self.tile_size = tile_size
        self.stride = stride

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
        img = np.squeeze(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
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
        instance_mask = np.squeeze(instance_mask)
        if instance_mask.ndim == 2:
            instance_mask = np.expand_dims(instance_mask, axis=-1)
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
        sample_subset = self.cell_type_table[self.cell_type_table[self.sample_key] == sample_name]
        cell_types = sample_subset[self.cell_type_key].values

        df = pd.DataFrame(
            {
                "labels": sample_subset[self.segment_label_key],
                "activity": conversion_matrix.loc[cell_types, marker].values,
                "cell_type": cell_types,
            }
        )
        return df, cell_types

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
        mplex_img = self.get_image(data_folder, marker).astype(np.float32)
        mplex_img /= self.normalization_dict[marker]
        mplex_img = mplex_img.clip(0, 1)
        binary_mask, instance_mask = self.get_inst_binary_masks(data_folder)
        fov = os.path.split(data_folder)[-1]
        # get the cell types and marker activity mask
        marker_activity, cell_types = self.get_marker_activity(fov, self.conversion_matrix, marker)
        marker_activity_mask = self.get_marker_activity_mask(
            instance_mask, binary_mask, marker_activity
        )
        return {
            "mplex_img": mplex_img.astype(np.float32),
            "binary_mask": binary_mask.astype(np.uint8),
            "instance_mask": instance_mask.astype(np.uint16),
            "imaging_platform": self.imaging_platform,
            "marker_activity_mask": marker_activity_mask.astype(np.uint8),
            "dataset": self.dataset,
            "marker": marker,
            "activity_df": marker_activity,
            "folder_name": fov,
        }

    def tile_example(
        self, example, spatial_keys=[
            "mplex_img", "binary_mask", "instance_mask", "marker_activity_mask",
        ],
    ):
        """Tiles the example into a grid of tiles
        Args:
            example (dict):
                The example to tile
            spatial_keys (list):
                The keys in the example to tile
        Returns:
            list:
                List of example dicts, one for each tile
        """
        # tile the example
        example = copy.deepcopy(example)
        tiled_examples = {}
        for key in spatial_keys:
            if example[key].ndim == 2:
                example[key] = np.expand_dims(example[key], axis=-1)

            # pad images if they are not divisible by the tile size
            if not example[key].shape[0] % self.tile_size[0] == 0 or \
                    not example[key].shape[1] % self.tile_size[1] == 0:
                example[key] = np.pad(
                    example[key],
                    ((0, example[key].shape[0] % self.tile_size[0]),
                     (0, example[key].shape[1] % self.tile_size[1]),
                     (0, 0)),
                    mode="constant",
                )
            res = np.lib.stride_tricks.sliding_window_view(
                example[key], window_shape=self.tile_size + list(example[key].shape[2:])
            )[:: self.stride[0], :: self.stride[1]]
            sh = list(res.shape)
            tiled_examples[key] = res.reshape([sh[0] * sh[1]] + sh[3:])  # tiles x H x W x C

        # store individual tiled examples in a list of dicts
        non_spatial_keys = [
            key for key in example if key not in spatial_keys + ["activity_df"]
        ]
        num_tiles = tiled_examples[spatial_keys[0]].shape[0]
        example_list = []
        for tile in range(num_tiles):
            example_out = {}
            for key in spatial_keys:
                example_out[key] = tiled_examples[key][tile]
            for non_spatial_key in non_spatial_keys:
                example_out[non_spatial_key] = example[non_spatial_key]

            # subset marker_activity to the labels that are present in the tile
            label_subset = np.unique(example_out["instance_mask"]).astype(np.uint16).tolist()
            example_out["activity_df"] = example["activity_df"].loc[
                [True if i in label_subset else False for i in example["activity_df"].labels]
            ]
            example_list.append(example_out)

        return example_list

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
        self.conversion_matrix = pd.read_csv(self.conversion_matrix_path, index_col=0)

        # check if markers were selected or take all markers from conversion matrix
        if self.selected_markers is None:
            self.selected_markers = list(self.conversion_matrix.columns)

        # check if selected markers are a list
        if not isinstance(self.selected_markers, list):
            self.selected_markers = [self.selected_markers]

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
            data_folders=list_folders(self.data_dir),
            warn=True
        )

        # make cell_types lowercase to make matching easier
        self.conversion_matrix.index = self.conversion_matrix.index.str.lower()
        self.cell_type_table[self.cell_type_key] = self.cell_type_table[self.cell_type_key] \
            .str.lower()

    def make_tf_record(self):
        """Iterates through the data_folders and loads, transforms and
        serializes a tfrecord example for each data_folder

        Args:
            tf_record_path (str):
                The path to the tf record to make
        """
        # load, prepare and check data
        self.load_and_check_input()

        # initialize tfrecord writer
        if not hasattr(self, "writer"):
            self.writer = tf.io.TFRecordWriter(
                os.path.join(self.tf_record_path, self.dataset + ".tfrecord")
            )

        # iterate through data_folders and markers to prepare and tile examples
        print("Preparing examples...")
        for data_folder in tqdm(self.data_folders):
            for marker in self.selected_markers:
                example = self.prepare_example(data_folder, marker)
                if self.tile_size:
                    example_list = self.tile_example(example)
                else:
                    example_list = [example]

                # serialize and write examples to tfrecord
                for example in example_list:
                    example_serialized = self.serialize_example(example)
                    self.writer.write(example_serialized)
        self.writer.close()
        delattr(self, "writer")

    def serialize_example(self, example):
        """Serializes an example dict to a tfrecord example

        Args:
            example (dict):
                The example dict to serialize
        Returns:
            tf.train.Example:
                The serialized example
        """
        string_example = {}

        for key in example.keys():
            # if key in spatial_keys:
            if type(example[key]) in [np.ndarray, tf.Tensor]:
                # convert float32 into uint16 for compression and storage
                if example[key].dtype not in [np.uint8, np.uint16]:
                    example[key] = example[key] * (np.iinfo(np.uint16).max)
                    example[key] = example[key].astype(np.uint16)
                # convert to bytes
                string_example[key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.encode_png(example[key]).numpy()]
                    )
                )
            elif type(example[key]) in [pd.DataFrame, pd.Series]:
                string_example[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=tf.strings.unicode_decode(example[key].to_json(), "UTF-8")
                    )
                )
            elif type(example[key]) == str:
                string_example[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=tf.strings.unicode_decode(example[key], "UTF-8")
                    )
                )
        #
        train_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    key: string_example[key]
                    for key in string_example.keys()
                }
            )
        )
        return train_example.SerializeToString()

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
        print("Calculating normalization quantiles...")
        for data_folder in tqdm(data_folders):
            for marker in selected_markers:
                img = self.get_image(data_folder, marker)
                if marker not in quantiles:
                    quantiles[marker] = []
                quantiles[marker].append(np.quantile(img, self.normalization_quantile))

        # calculate the normalization matrix
        normalization_matrix = {}
        for marker in selected_markers:
            normalization_matrix[marker] = np.mean(quantiles[marker])

        # check path and save the normalization matrix
        if not str(self.normalization_dict_path).endswith(".json"):
            self.normalization_dict_path = os.path.join(
                self.tf_record_path, "normalization_dict.json"
            )
        with open(self.normalization_dict_path, "w") as f:
            json.dump(normalization_matrix, f)
        self.normalization_dict = normalization_matrix
        return normalization_matrix


feature_description = {
    "mplex_img": tf.io.RaggedFeature(tf.string),
    "binary_mask": tf.io.RaggedFeature(tf.string),
    "instance_mask": tf.io.RaggedFeature(tf.string),
    "imaging_platform": tf.io.RaggedFeature(tf.int64),
    "marker_activity_mask": tf.io.RaggedFeature(tf.string),
    "dataset": tf.io.RaggedFeature(tf.int64),
    "marker": tf.io.RaggedFeature(tf.int64),
    "activity_df": tf.io.RaggedFeature(tf.int64),
    "folder_name": tf.io.RaggedFeature(tf.int64),
}


def parse_dict(deserialized_dict):
    """Parse an example into a dictionary of tensors

    Args:
        deserialized_dict: a deserialized dictionary
    Returns:
        a dictionary of tensors and metadata strings
    """
    example = {}
    for key in ["dataset", "marker", "imaging_platform", "folder_name", "activity_df"]:
        example[key] = tf.strings.unicode_encode(
            tf.cast(deserialized_dict[key], tf.int32), "UTF-8"
        )
        if hasattr(example[key], "numpy"):
            example[key] = example[key].numpy().decode()
    for key in ["binary_mask", "marker_activity_mask"]:
        example[key] = tf.io.decode_png(deserialized_dict[key][0], dtype=tf.uint8)
    for key in ["mplex_img", "instance_mask"]:
        example[key] = tf.io.decode_png(deserialized_dict[key][0], dtype=tf.uint16)
    example["mplex_img"] = tf.cast(example["mplex_img"], tf.float32) / tf.constant(
        (np.iinfo(np.uint16).max), dtype=tf.float32
    )
    if type(example["activity_df"]) == str:
        example["activity_df"] = pd.read_json(example["activity_df"])
    return example
