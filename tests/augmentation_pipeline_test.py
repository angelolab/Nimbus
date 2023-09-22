from copy import deepcopy

import imgaug.augmenters as iaa
import numpy as np
import pytest
import tensorflow as tf

from cell_classification.augmentation_pipeline import (
    Flip, GaussianBlur, GaussianNoise, LinearContrast, MixUp, Rot90,
    augment_images, get_augmentation_pipeline, prepare_keras_aug,
    prepare_tf_aug, py_aug)

parametrize = pytest.mark.parametrize


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
        "rotate_prob": 1.0,
        "rotate_count": 3,
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
    assert isinstance(augmentation_pipeline, iaa.Sequential)


@parametrize("batch_num", [1, 2, 3])
@parametrize("chan_num", [1, 2, 3])
def test_augment_images(batch_num, chan_num):
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    images = np.zeros([batch_num, 100, 100, chan_num], dtype=np.float32)
    masks = np.zeros([batch_num, 100, 100, chan_num], dtype=np.int32)
    images[0, :50, :50, :] = 10.1
    images[0, 50:, 50:, :] = 201.12
    masks[0, :50, :50] = 1
    masks[0, 50:, 50:] = 2
    augmented_images, augmented_masks = augment_images(images, masks, augmentation_pipeline)

    # check if right types and shapes are returned
    assert isinstance(augmented_images, np.ndarray)
    assert isinstance(augmented_masks, np.ndarray)
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
    assert np.abs(augmented_images[augmented_masks == 0].mean() - images[masks == 0].mean()) < 1.5
    assert np.abs(augmented_images[augmented_masks == 1].mean() - images[masks == 1].mean()) < 1.5
    assert np.abs(augmented_images[augmented_masks == 2].mean() - images[masks == 2].mean()) < 5

    # check control flow for no channel dimensions for the masks
    masks = np.zeros([batch_num, 100, 100], dtype=np.int32)
    _, augmented_masks = augment_images(images, masks, augmentation_pipeline)
    assert augmented_masks.shape == masks.shape


def prepare_data(batch_num, return_tensor=False):
    mplex_img = np.zeros([batch_num, 100, 100, 2], dtype=np.float32)
    binary_mask = np.zeros([batch_num, 100, 100, 1], dtype=np.int32)
    marker_activity_mask = np.zeros([batch_num, 100, 100, 1], dtype=np.int32)
    mplex_img[0, :30, :50, :] = 10.1
    mplex_img[0, 50:, 50:, :] = 21.12
    mplex_img[-1, 30:60, :50, :] = 14.11
    mplex_img[-1, :50, 50:, :] = 18.12
    binary_mask[0, :30, :50] = 1
    binary_mask[0, 50:, 50:] = 1
    binary_mask[-1, 30:60, :50, :] = 1
    binary_mask[-1, :50, 50:, :] = 1
    marker_activity_mask[0, :30, :50] = 1
    marker_activity_mask[0, 50:, 50:] = 2
    marker_activity_mask[-1, 30:60, :50, :] = 1
    marker_activity_mask[-1, :50, 50:, :] = 2

    if return_tensor:
        mplex_img = tf.constant(mplex_img, tf.float32)
        binary_mask = tf.constant(binary_mask, tf.int32)
        marker_activity_mask = tf.constant(marker_activity_mask, tf.int32)
    return mplex_img, binary_mask, marker_activity_mask


@parametrize("batch_num", [1, 2, 3])
def test_prepare_tf_aug(batch_num):
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    tf_aug = prepare_tf_aug(augmentation_pipeline)
    mplex_img, binary_mask, marker_activity_mask = prepare_data(batch_num)
    mplex_aug, mask_out, marker_activity_aug = tf_aug(
        tf.constant(mplex_img, tf.float32), binary_mask, marker_activity_mask
    )
    mplex_img, binary_mask, marker_activity_mask = prepare_data(batch_num)
    # check if right types and shapes are returned
    assert isinstance(mplex_aug, np.ndarray)
    assert isinstance(mask_out, np.ndarray)
    assert isinstance(marker_activity_aug, np.ndarray)
    assert mplex_aug.dtype == np.float32
    assert mask_out.dtype == np.int32
    assert marker_activity_aug.dtype == np.int32
    assert mplex_aug.shape == mplex_img.shape
    assert mask_out.shape == binary_mask.shape
    assert marker_activity_aug.shape == marker_activity_mask.shape


@parametrize("batch_num", [1, 2, 3])
def test_py_aug(batch_num):
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    tf_aug = prepare_tf_aug(augmentation_pipeline)
    mplex_img, binary_mask, marker_activity_mask = prepare_data(batch_num)
    batch = {
        "mplex_img": tf.constant(mplex_img, tf.float32),
        "binary_mask": tf.constant(binary_mask, tf.int32),
        "marker_activity_mask": tf.constant(marker_activity_mask, tf.int32),
        "dataset": "test_dataset",
        "marker": "test_marker",
        "imaging_platform": "test_platform",
    }
    batch_aug = py_aug(deepcopy(batch), tf_aug)

    # check if right types and shapes are returned
    for key in batch.keys():
        assert isinstance(batch_aug[key], type(batch[key]))

    for key in ["mplex_img", "binary_mask", "marker_activity_mask"]:
        assert batch_aug[key].shape == batch[key].shape
        assert not np.array_equal(batch_aug[key], batch[key])


@parametrize("batch_num", [2, 4, 8])
def test_prepare_keras_aug(batch_num):
    params = get_params()
    augmentation_pipeline = prepare_keras_aug(params)
    images, _, masks = prepare_data(batch_num, True)
    augmented_images, augmented_masks = augmentation_pipeline(images, masks)

    # check if right types and shapes are returned
    assert augmented_images.dtype == tf.float32
    assert augmented_masks.dtype == tf.int32
    assert augmented_images.shape == images.shape
    assert augmented_masks.shape == masks.shape


@parametrize("batch_num", [2, 4, 8])
def test_flip(batch_num):
    images, _, masks = prepare_data(batch_num, True)
    flip = Flip(prob=1.0)
    aug_img, aug_mask = flip(images, masks)

    # check if right types and shapes are returned
    assert aug_img.dtype == images.dtype
    assert aug_mask.dtype == masks.dtype
    assert aug_img.shape == images.shape
    assert aug_mask.shape == masks.shape

    # check if data got flipped
    assert not np.array_equal(aug_img, images)
    assert not np.array_equal(aug_mask, masks)
    assert np.sum(aug_img) == np.sum(images)
    assert np.sum(aug_mask) == np.sum(masks)


@parametrize("batch_num", [2, 4, 8])
def test_rot90(batch_num):
    images, _, masks = prepare_data(batch_num, True)
    rot90 = Rot90(prob=1.0, rotate_count=2)
    aug_img, aug_mask = rot90(images, masks)

    # check if right types and shapes are returned
    assert aug_img.dtype == images.dtype
    assert aug_mask.dtype == masks.dtype
    assert aug_img.shape == images.shape
    assert aug_mask.shape == masks.shape

    # check if data got rotated
    assert not np.array_equal(aug_img, images)
    assert not np.array_equal(aug_mask, masks)
    assert np.sum(aug_img) == np.sum(images)
    assert np.sum(aug_mask) == np.sum(masks)


@parametrize("batch_num", [2, 4, 8])
def test_gaussian_noise(batch_num):
    images, _, masks = prepare_data(batch_num, True)
    gaussian_noise = GaussianNoise(prob=1.0)
    aug_img, aug_mask = gaussian_noise(images, masks)

    # check if right types and shapes are returned
    assert aug_img.dtype == images.dtype
    assert aug_mask.dtype == masks.dtype
    assert aug_img.shape == images.shape
    assert aug_mask.shape == masks.shape

    # check if data got augmented
    assert not np.array_equal(aug_img, images)
    assert np.array_equal(aug_mask, masks)
    assert np.isclose(np.mean(aug_img), np.mean(images), atol=0.1)


@parametrize("batch_num", [2, 4, 8])
def test_gaussian_blur(batch_num):
    images, _, masks = prepare_data(batch_num, True)
    gaussian_blur = GaussianBlur(1.0, 0.5, 1.5, 5)
    aug_img, aug_mask = gaussian_blur(images, masks)

    # check if right types and shapes are returned
    assert aug_img.dtype == images.dtype
    assert aug_mask.dtype == masks.dtype
    assert aug_img.shape == images.shape
    assert aug_mask.shape == masks.shape

    # check if data got augmented
    assert not np.array_equal(aug_img, images)
    assert np.array_equal(aug_mask, masks)
    assert np.isclose(np.mean(aug_img), np.mean(images), atol=0.2)


@parametrize("batch_num", [2, 4, 8])
def test_linear_contrast(batch_num):
    images, _, masks = prepare_data(batch_num, True)
    linear_contrast = LinearContrast(1.0, 0.75, 0.75)
    aug_img, aug_mask = linear_contrast(images, masks)

    # check if right types and shapes are returned
    assert aug_img.dtype == images.dtype
    assert aug_mask.dtype == masks.dtype
    assert aug_img.shape == images.shape
    assert aug_mask.shape == masks.shape

    # check if data got augmented
    assert not np.array_equal(aug_img, images)
    assert np.array_equal(aug_mask, masks)
    assert np.isclose(np.mean(aug_img), np.mean(images) * 0.75, atol=0.01)


@parametrize("batch_num", [2, 4, 8])
def test_mixup(batch_num):
    images, _, labels = prepare_data(batch_num, True)
    mixup = MixUp(1.0, 0.5)
    x_mplex, x_binary = tf.split(images, 2, axis=-1)
    loss_mask = tf.cast(labels, tf.float32)

    x_mplex_aug, x_binary_aug, labels_aug, loss_mask_aug = mixup(
        x_mplex, x_binary, labels, loss_mask
    )

    # check if right types and shapes are returned
    assert x_mplex_aug.dtype == x_mplex.dtype
    assert x_binary_aug.dtype == x_binary.dtype
    assert labels_aug.dtype == tf.float32
    assert loss_mask_aug.dtype == loss_mask.dtype
    assert x_mplex_aug.shape == x_mplex.shape
    assert x_binary_aug.shape == x_binary.shape
    assert labels_aug.shape == labels.shape
    assert loss_mask_aug.shape == loss_mask.shape

    # check if data got augmented
    assert not np.array_equal(x_mplex_aug, x_mplex)
    assert not np.array_equal(x_binary_aug, x_binary)
    assert not np.array_equal(labels_aug, labels)
    assert not np.array_equal(loss_mask_aug, loss_mask)

    # check if data got mixed up
    assert np.isclose(np.mean(x_mplex_aug), np.mean(x_mplex), atol=0.1)
    assert np.isclose(np.mean(x_binary_aug), np.mean(x_binary), atol=0.1)
    assert np.isclose(np.mean(labels_aug), np.mean(labels), atol=0.1)
    assert np.isclose(
        np.mean(loss_mask_aug), np.mean(loss_mask*tf.reverse(loss_mask, [0])), atol=0.1
    )
