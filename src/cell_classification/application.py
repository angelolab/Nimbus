from deepcell.model_zoo.panopticnet import PanopticNet
from cell_classification.semantic_head import create_semantic_head
from deepcell.applications import Application
from alpineer import io_utils
from cell_classification.inference import prepare_normalization_dict, predict
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


class Nimbus(Application):
    """Nimbus application class for predicting marker activity for cells in multiplexed images.
    """
    def __init__(
              self, fov_paths, exclude_channels, segmentation_naming_convention, output_dir,
                save_predictions, half_resolution=True
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
        """
        self.fov_paths = fov_paths
        self.exclude_channels = exclude_channels
        self.segmentation_naming_convention = segmentation_naming_convention
        self.output_dir = output_dir
        self.half_resolution = half_resolution
        self.save_predictions = save_predictions
        
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

    def initialize_model(self):
        """Initializes the model and load weights.
        """
        backbone = "efficientnetv2bs"
        input_shape = [1024,1024,2]
        model = PanopticNet(
            backbone=backbone, input_shape=input_shape,
            norm_method="std", num_semantic_classes=[1],
            create_semantic_head=create_semantic_head, location=False,
        )
        self.checkpoint_path = os.path.normpath("../cell_classification/checkpoints/" +
                                "halfres_512_checkpoint_160000.h5"
        )
        model.load_weights(self.checkpoint_path)
        print("Loaded weights from {}".format(self.checkpoint_path))
        self.model = model

    def prepare_normalization_dict(
            self, quantile=0.999, n_subset=10, multiprocessing=False
        ):
        """Load or prepare and save normalization dictionary for Nimbus model.
        Args:
            quantile (float): Quantile to use for normalization.
            n_subset (int): Number of fovs to use for normalization.
            multiprocessing (bool): Whether to use multiprocessing.
        Returns:
            dict: Dictionary of normalization factors.
        """
        self.normalization_dict_path = os.path.join(self.output_dir, "normalization_dict.json")
        if os.path.exists(self.normalization_dict_path):
            self.normalization_dict = json.loads(self.normalization_dict_path)
        else:
            n_jobs = os.cpu_count() if multiprocessing else 1
            self.normalization_dict = prepare_normalization_dict(
                self.fov_paths, self.output_dir, quantile, self.exclude_channels, n_subset,
                n_jobs
            )

    def predict_fovs(self, input_data, normalize=True, marker=None, normalization_dict=None):
        """Predicts cell classification for input data.
        Args:
            input_data (np.array): Input data to predict on.
            normalize (bool): Whether to normalize input data.
            marker (str): Name of marker to normalize.
            normalization_dict (dict): Dictionary of normalization factors.
        Returns:
            np.array: Predicted cell classification.
        """
        self.cell_table = predict(
            self.fov_paths, self.nimbus_output_dir, self, self.normalization_dict,
            self.segmentation_naming_convention, self.exclude_channels, self.save_predictions,
            self.half_resolution,
        )
        self.cell_table.to_csv(
            os.path.join(self.nimbus_output_dir,"nimbus_cell_table.csv"), index=False
        )

# if plot_predictions:
#     fig, ax = plt.subplots(1,3, figsize=(16,16))
#     # plot stuff
#     ax[0].imshow(np.squeeze(input_data[...,0]), vmin=0, vmax=np.quantile(input_data[...,0], 0.999))
#     ax[0].set_title(channel)
#     ax[1].imshow(np.squeeze(input_data[...,1]), cmap="Grays")
#     ax[1].set_title("Segmentation")
#     ax[2].imshow(np.squeeze(prediction), vmin=0, vmax=1)
#     ax[2].set_title(channel+"_pred")
#     for a in ax:
#         a.set_xticks([])
#         a.set_yticks([])
#     plt.tight_layout()
#     plt.show()
