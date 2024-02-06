from deepcell.panopticnet import PanopticNet
from deepcell.semantic_head import create_semantic_head
from deepcell.application import Application
from alpineer import io_utils
from cell_classification.inference import prepare_normalization_dict, predict_fovs
import cell_classification
from pathlib import Path
from glob import glob
import tensorflow as tf
import numpy as np
import json
import os


def nimbus_preprocess(image, **kwargs):
    """Preprocess input data for Nimbus model.
    Args:
        image: array to be processed
    Returns:
        np.array: processed image array
    """
    output = np.copy(image)
    if len(image.shape) != 4:
        raise ValueError("Image data must be 4D, got image of shape {}".format(image.shape))

    normalize = kwargs.get('normalize', True)
    if normalize:
        marker = kwargs.get('marker', None)
        normalization_dict = kwargs.get('normalization_dict', {})
        if marker in normalization_dict.keys():
            norm_factor = normalization_dict[marker]
        else:
            print("Norm_factor not found for marker {}, calculating directly from the image. \
            ".format(marker))
            norm_factor = np.quantile(output[..., 0], 0.999)
        # normalize only marker channel in chan 0 not binary mask in chan 1
        output[..., 0] /= norm_factor
        output = output.clip(0, 1)
    return output


def nimbus_postprocess(model_output):
    return model_output


def format_output(model_output):
    return model_output[0]


def prep_deepcell_naming_convention(deepcell_output_dir):
    """Prepares the naming convention for the segmentation data
    Args:
        deepcell_output_dir (str): path to directory where segmentation data is saved
    Returns:
        segmentation_naming_convention (function): function that returns the path to the
            segmentation data for a given fov
    """
    def segmentation_naming_convention(fov_path):
        """Prepares the path to the segmentation data for a given fov
        Args:
            fov_path (str): path to fov
        Returns:
            seg_path (str): paths to segmentation fovs
        """
        fov_name = os.path.basename(fov_path)
        return os.path.join(
            deepcell_output_dir, fov_name + "_whole_cell.tiff"
        )
    return segmentation_naming_convention


class Nimbus(Application):
    """Nimbus application class for predicting marker activity for cells in multiplexed images.
    """
    def __init__(
              self, fov_paths, segmentation_naming_convention, output_dir,
                save_predictions=True, exclude_channels=[], half_resolution=True,
                batch_size=4, test_time_aug=True, input_shape=[1024,1024]
        ):
        """Initializes a Nimbus Application.
        Args:
            fov_paths (list): List of paths to fovs to be analyzed.
            exclude_channels (list): List of channels to exclude from analysis.
            segmentation_naming_convention (function): Function that returns the path to the
                segmentation mask for a given fov path.
            output_dir (str): Path to directory to save output.
            save_predictions (bool): Whether to save predictions.
            half_resolution (bool): Whether to run model on half resolution images.
            batch_size (int): Batch size for model inference.
            test_time_aug (bool): Whether to use test time augmentation.
            input_shape (list): Shape of input images.
        """
        self.fov_paths = fov_paths
        self.exclude_channels = exclude_channels
        self.segmentation_naming_convention = segmentation_naming_convention
        self.output_dir = output_dir
        self.half_resolution = half_resolution
        self.save_predictions = save_predictions
        self._batch_size = batch_size
        self.checked_inputs = False
        self.test_time_aug = test_time_aug
        self.input_shape = input_shape
        # exclude segmentation channel from analysis
        seg_name = os.path.basename(self.segmentation_naming_convention(self.fov_paths[0]))
        self.exclude_channels.append(seg_name.split(".")[0])
        if self.output_dir != '':
            os.makedirs(self.output_dir, exist_ok=True)
        
        # initialize model and parent class
        self.initialize_model()
        
        super(Nimbus, self).__init__(
            model=self.model, 
            model_image_shape=self.model.input_shape[1:],
            preprocessing_fn=nimbus_preprocess,
            postprocessing_fn=nimbus_postprocess,
            format_model_output_fn=format_output,
        )

    def check_inputs(self):
        """ check inputs for Nimbus model
        """
        # check if all paths in fov_paths exists
        io_utils.validate_paths(self.fov_paths)

        # check if segmentation_naming_convention returns valid paths
        path_to_segmentation = self.segmentation_naming_convention(self.fov_paths[0])
        if not os.path.exists(path_to_segmentation):
            raise FileNotFoundError("Function segmentation_naming_convention does not return valid\
                                    path. Segmentation path {} does not exist."\
                                    .format(path_to_segmentation))
        # check if output_dir exists
        io_utils.validate_paths([self.output_dir])

        if isinstance(self.exclude_channels, str):
            self.exclude_channels = [self.exclude_channels]
        self.checked_inputs = True
        print("All inputs are valid.")

    def initialize_model(self):
        """Initializes the model and load weights.
        """
        backbone = "efficientnetv2bs"
        input_shape = self.input_shape + [2]
        model = PanopticNet(
            backbone=backbone, input_shape=input_shape,
            norm_method="std", num_semantic_classes=[1],
            create_semantic_head=create_semantic_head, location=False,
        )
        # make sure path can be resolved on any OS and when importing  from anywhere
        self.checkpoint_path = os.path.normpath(
            "../cell_classification/checkpoints/halfres_512_checkpoint_160000.h5"
        )
        if not os.path.exists(self.checkpoint_path):
            path = os.path.abspath(cell_classification.__file__)
            path = Path(path).resolve()
            self.checkpoint_path = os.path.join(
                *path.parts[:-3], 'checkpoints', 'halfres_512_checkpoint_160000.h5'
            )
        if not os.path.exists(self.checkpoint_path):
            self.checkpoint_path = os.path.abspath(*glob('**/halfres_512_checkpoint_160000.h5'))

        if not os.path.exists(self.checkpoint_path):
            self.checkpoint_path = os.path.join(
                os.getcwd(), 'checkpoints', 'halfres_512_checkpoint_160000.h5'
            )

        if os.path.exists(self.checkpoint_path):
            model.load_weights(self.checkpoint_path)
            print("Loaded weights from {}".format(self.checkpoint_path))
        else:
            raise FileNotFoundError("Could not find Nimbus weights at {ckpt_path}. \
                                    Current path is {current_path} and directory contains {dir_c},\
                                    path to cell_clasification i{p}".format(
                                        ckpt_path=self.checkpoint_path,
                                        current_path=os.getcwd(),
                                        dir_c=os.listdir(os.getcwd()),
                                        p=os.path.abspath(cell_classification.__file__)
                                    )
            )
        self.model = model

    def prepare_normalization_dict(
            self, quantile=0.999, n_subset=10, multiprocessing=False, overwrite=False,
        ):
        """Load or prepare and save normalization dictionary for Nimbus model.
        Args:
            quantile (float): Quantile to use for normalization.
            n_subset (int): Number of fovs to use for normalization.
            multiprocessing (bool): Whether to use multiprocessing.
            overwrite (bool): Whether to overwrite existing normalization dict.
        Returns:
            dict: Dictionary of normalization factors.
        """
        self.normalization_dict_path = os.path.join(self.output_dir, "normalization_dict.json")
        if os.path.exists(self.normalization_dict_path) and not overwrite:
            self.normalization_dict = json.load(open(self.normalization_dict_path))
        else:

            n_jobs = os.cpu_count() if multiprocessing else 1
            self.normalization_dict = prepare_normalization_dict(
                self.fov_paths, self.output_dir, quantile, self.exclude_channels, n_subset, n_jobs
            )

    def predict_fovs(self):
        """Predicts cell classification for input data.
        Returns:
            np.array: Predicted cell classification.
        """
        if self.checked_inputs == False:
            self.check_inputs()
        if not hasattr(self, "normalization_dict"):
            self.prepare_normalization_dict()
        # check if GPU is available
        print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
        print("Predictions will be saved in {}".format(self.output_dir))
        print("Iterating through fovs will take a while...")
        self.cell_table = predict_fovs(
            self.fov_paths, self.output_dir, self, self.normalization_dict,
            self.segmentation_naming_convention, self.exclude_channels, self.save_predictions,
            self.half_resolution, batch_size=self._batch_size,
            test_time_augmentation=self.test_time_aug,
        )
        self.cell_table.to_csv(
            os.path.join(self.output_dir,"nimbus_cell_table.csv"), index=False
        )
        return self.cell_table

Nimbus(["none_path"], lambda x: x, "")