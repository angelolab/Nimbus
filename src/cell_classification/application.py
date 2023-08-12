from deepcell.model_zoo.panopticnet import PanopticNet
from cell_classification.semantic_head import create_semantic_head
from deepcell.applications import Application
import numpy as np


def cell_preprocess(image, **kwargs):
    """Preprocess input data for CellClassification model.
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


def cell_postprocess(model_output):
    return model_output


def format_output(model_output):
    return model_output[0]


class CellClassification(Application):
    """Cell Classification Application class for predicting marker activity for cells in multi-
    plexed images.
    """
    def __init__(self):
        """Initializes a CellClassification Application.
        Args:
            model (tensorflow.keras.Model): Model to load weights into.
        """
        backbone = "efficientnetv2bs"
        input_shape = [1024,1024,2]
        model = PanopticNet(
            backbone=backbone, input_shape=input_shape,
            norm_method="std", num_semantic_classes=[1],
            create_semantic_head=create_semantic_head, location=False,
        )
        super(CellClassification, self).__init__(
            model=model, 
            model_image_shape=model.input_shape[1:],
            preprocessing_fn=cell_preprocess,
            postprocessing_fn=cell_postprocess,
            format_model_output_fn=format_output,
        )

    def load_weights(self, checkpoint_path):
        """Loads weights from a checkpoint file.
        Args:
            checkpoint_path (str): Path to checkpoint file.
        """
        self.model.load_weights(checkpoint_path)
        print("Loaded weights from {}".format(checkpoint_path))

    def predict(self, input_data, normalize=True, marker=None, normalization_dict=None):
        """Predicts cell classification for input data.
        Args:
            input_data (np.array): Input data to predict on.
            normalize (bool): Whether to normalize input data.
            marker (str): Name of marker to normalize.
            normalization_dict (dict): Dictionary of normalization factors.
        Returns:
            np.array: Predicted cell classification.
        """
        return self._predict_segmentation(input_data, preprocess_kwargs={
            'normalize': normalize, 'marker': marker, 'normalization_dict': normalization_dict
        })
