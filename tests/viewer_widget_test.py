from cell_classification.viewer_widget import NimbusViewer
from segmentation_data_prep_test import prep_object_and_inputs
import numpy as np
import tempfile
import os


def test_NimbusViewer():
    with tempfile.TemporaryDirectory() as temp_dir:
        _, _, _, _ = prep_object_and_inputs(temp_dir)
        viewer_widget = NimbusViewer(temp_dir, temp_dir)
        assert isinstance(viewer_widget, NimbusViewer)


def test_composite_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        _, _, _, _ = prep_object_and_inputs(temp_dir)
        viewer_widget = NimbusViewer(temp_dir, temp_dir)
        path_dict = {
            "red": os.path.join(temp_dir, "fov_0", "CD4.tiff"),
            "green": os.path.join(temp_dir, "fov_0", "CD11c.tiff"),
        }
        composite_image = viewer_widget.create_composite_image(path_dict)
        assert isinstance(composite_image, np.ndarray)
        assert composite_image.shape == (256, 256, 2)

        path_dict["blue"] = os.path.join(temp_dir, "fov_0", "CD56.tiff")
        composite_image = viewer_widget.create_composite_image(path_dict)
        assert composite_image.shape == (256, 256, 3)
