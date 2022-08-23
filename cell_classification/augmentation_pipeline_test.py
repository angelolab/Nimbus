import pytest
import numpy as np
from augmentation_pipeline import augment_images, get_augmentation_pipeline
import imgaug.augmenters as iaa


def get_params():
    return {
        # flip
        "flip_prob": 1.0,
        # affine
        "affine_prob": 1.0,
        "scale_min": 0.5,
        "scale_max": 1.5,
        "shear_angle": 10,
        # elastic
        "elastic_prob": 1.0,
        "elastic_alpha": [0, 5.0],
        "elastic_sigma": 0.5,
        # rotate
        "rotate_count": [0, 3],
        # gaussian noise
        "gaussian_noise_prob": 1.0,
        "gaussian_noise_min": 0.1,
        "gaussian_noise_max": 0.5,
        # gaussian blur
        "gaussian_blur_prob": 0.0,
        "gaussian_blur_min": 0.1,
        "gaussian_blur_max": 0.5,
        # contrast aug
        "contrast_prob": 0.0,
        "contrast_min": 0.1,
        "contrast_max": 2.0,
    }


def test_get_augmentation_pipeline():
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    assert type(augmentation_pipeline) == iaa.Sequential


def test_augment_images():
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    images = np.zeros([3, 100, 100, 2], dtype=np.float32)
    masks = np.zeros([3, 100, 100, 2], dtype=np.int32)
    images[0, :50, :50, :] = 10.1
    images[1, 50:, 50:, :] = 201.12
    masks[0, :50, :50] = 1
    masks[1, 50:, 50:] = 2
    augmented_images, augmented_masks = augment_images(images, masks, augmentation_pipeline)

    # check if right types and shapes are returned
    assert type(augmented_images) == np.ndarray
    assert type(augmented_masks) == np.ndarray
    assert augmented_images.dtype == np.float32
    assert augmented_masks.dtype == np.int32
    assert augmented_images.shape == images.shape
    assert augmented_masks.shape == masks.shape

    # check if images are augmented
    assert not np.array_equal(augmented_images, images)
    assert not np.array_equal(augmented_masks, masks)

    # check if masks are still binary with the right labels
    assert list(np.unique(augmented_masks)) == [0, 1, 2]

    # check if images and masks where augmented with the same spatial augmentations approx.
    assert np.abs(augmented_images[augmented_masks == 0].mean() - images[masks == 0].mean()) < 1
    assert np.abs(augmented_images[augmented_masks == 1].mean() - images[masks == 1].mean()) < 1
    assert np.abs(augmented_images[augmented_masks == 2].mean() - images[masks == 2].mean()) < 5

    # check control flow for different number of samples per batch and channels
    for b in [1, 2, 3]:
        for c in [1, 2, 3]:
            images = np.zeros([b, 100, 100, c], dtype=np.float32)
            masks = np.zeros([b, 100, 100, c], dtype=np.int32)
            # if c == 1:
            #     masks = np.squeeze(masks, axis=-1)
            _, augmented_masks = augment_images(images, masks, augmentation_pipeline)
            assert augmented_masks.shape == masks.shape

    # check control flow for no channel dimensions for the masks
    masks = np.zeros([b, 100, 100], dtype=np.int32)
    _, augmented_masks = augment_images(images, masks, augmentation_pipeline)
    assert augmented_masks.shape == masks.shape
