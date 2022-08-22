import tempfile
import pytest
import numpy as np
from augmentation_pipeline import augment_images, get_augmentation_pipeline
import imgaug.augmenters as iaa


def get_params():
    return {
        # flip
        "flip_prob": 0.5,
        # affine
        "affine_prob": 0.5, "scale_min": 0.5,
        "scale_max": 1.5, "shear_angle": 10,
        # elastic
        "elastic_prob": 0.5, "elastic_alpha": [0, 5.0], "elastic_sigma": 0.5,
        # rotate
        "rotate_count": [0, 3],
        # gaussian noise
        "gaussian_noise_prob": 0.5, "gaussian_noise_min": 0.1, "gaussian_noise_max": 0.5,
        # gaussian blur
        "gaussian_blur_prob": 0.5, "gaussian_blur_min": 0.1, "gaussian_blur_max": 0.5,
        # contrast aug
        "contrast_prob": 0.5, "contrast_min": 0.1, "contrast_max": 2.0,
    }


def test_get_augmentation_pipeline():
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    assert type(augmentation_pipeline) == iaa.Sequential


def test_augment_images():
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    images = np.random.rand(10, 100, 100, 1).astype(np.float32)
    masks = np.random.randint(0, 2, [10, 100, 100], dtype=np.int32)
    augmented_images, augmented_masks = augment_images(images, masks, augmentation_pipeline)

    # check if right types and shapes are returned
    assert type(augmented_images) == np.ndarray
    assert type(augmented_masks) == np.ndarray
    assert augmented_images.dtype == np.float32
    assert augmented_masks.dtype == np.int32
    assert augmented_images.shape == images.shape
    assert augmented_masks.shape == masks.shape

    # check if binary masks are returned
    assert list(np.unique(augmented_masks)) == [0, 1]
