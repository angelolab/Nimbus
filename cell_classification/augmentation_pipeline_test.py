import pytest
import numpy as np
from augmentation_pipeline import augment_images, get_augmentation_pipeline, prepare_tf_aug, py_aug
from augmentation_pipeline import prepare_keras_aug
import tensorflow as tf
import imgaug.augmenters as iaa
import tensorflow as tf
from copy import deepcopy

parametrize = pytest.mark.parametrize


def get_params():
    return {
        # flip
        "flip_prob": 1.0,
        # affine
        "affine_prob": 1.0, "scale_min": 0.5, "scale_max": 1.5, "shear_angle": 10,
        # elastic
        "elastic_prob": 1.0, "elastic_alpha": [0, 5.0], "elastic_sigma": 0.5,
        # rotate
        "rotate_prob": 1.0, "rotate_count": 3,
        # gaussian noise
        "gaussian_noise_prob": 1.0, "gaussian_noise_min": 0.1, "gaussian_noise_max": 0.5,
        # gaussian blur
        "gaussian_blur_prob": 0.0, "gaussian_blur_min": 0.1, "gaussian_blur_max": 0.5,
        # contrast aug
        "contrast_prob": 0.0, "contrast_min": 0.1, "contrast_max": 2.0,
    }


def test_get_augmentation_pipeline():
    params = get_params()
    augmentation_pipeline = get_augmentation_pipeline(params)
    assert type(augmentation_pipeline) == iaa.Sequential


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

    # check control flow for no channel dimensions for the masks
    masks = np.zeros([batch_num, 100, 100], dtype=np.int32)
    _, augmented_masks = augment_images(images, masks, augmentation_pipeline)
    assert augmented_masks.shape == masks.shape


def prepare_data(batch_num):
    mplex_img = np.zeros([batch_num, 100, 100, 2], dtype=np.float32)
    binary_mask = np.zeros([batch_num, 100, 100, 1], dtype=np.int32)
    marker_activity_mask = np.zeros([batch_num, 100, 100, 1], dtype=np.int32)
    mplex_img[0, :50, :50, :] = 10.1
    mplex_img[0, 50:, 50:, :] = 201.12
    binary_mask[0, :50, :50] = 1
    binary_mask[0, 50:, 50:] = 1
    marker_activity_mask[0, :50, :50] = 1
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
    assert type(mplex_aug) == np.ndarray
    assert type(mask_out) == np.ndarray
    assert type(marker_activity_aug) == np.ndarray
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
        assert not np.array_equal(batch_aug[key],  batch[key])


def test_prepare_keras_aug(batch_num=2, chan_num=2):
    params = get_params()
    augmentation_pipeline = prepare_keras_aug(params)
    images = np.zeros([batch_num, 100, 100, chan_num], dtype=np.float32)
    masks = np.zeros([batch_num, 100, 100, chan_num], dtype=np.int32)
    images[0, :30, :50, :] = 10.1
    images[0, 50:, 50:, :] = 21.12
    masks[0, :30, :50] = 1
    masks[0, 50:, 50:] = 2
    images = tf.constant(images, tf.float32)
    masks = tf.constant(masks, tf.int32)
    augmented_images, augmented_masks = augmentation_pipeline(images, masks)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(augmented_images[0, :, :, 0])
    ax[1].imshow(augmented_masks[0, :, :, 0])
    plt.show()
test_prepare_keras_aug()