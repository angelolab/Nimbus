import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.batches import Batch
import numpy as np


def augment_images(images, masks, augmentation_pipeline):
    """
    Augment images and masks.
    Args:
        images (np.array):
            The images (b,h,w,c) to augment
        masks (np.array):
            The masks (b,h,w,1) to augment
        augmentation_pipeline (imgaug.augmenters.meta.Augmenter):
            The augmentation pipeline
    Returns:
        np.array:
            The augmented images
        np.array:
            The augmented masks
    """
    masks = [SegmentationMapsOnImage(mask, shape=images.shape[1:-1]) for mask in masks]
    batch = Batch(images=images, segmentation_maps=masks)
    batch = augmentation_pipeline.augment_batch_(batch)
    augmented_masks = [np.squeeze(mask.arr) for mask in batch.segmentation_maps_aug]
    return np.stack(batch.images_aug, 0), np.squeeze(np.stack(augmented_masks, 0))


def get_augmentation_pipeline(params):
    """
    Get the augmentation pipeline.
    Args:
        params (dict):
            The parameters for the augmentation
    Returns:
        imgaug.augmenters.meta.Augmenter:
            The augmentation pipeline
    """
    augmentation_pipeline = iaa.Sequential(
        [
            iaa.Fliplr(params["flip_prob"]),
            iaa.Flipud(params["flip_prob"]),
            iaa.Sometimes(
                params["affine_prob"],
                iaa.Affine(
                    scale=(params["scale_min"], params["scale_max"]),
                    shear=(-params["shear_angle"], params["shear_angle"]),
                ),
            ),
            iaa.Sometimes(
                params["elastic_prob"],
                iaa.ElasticTransformation(
                    alpha=params["elastic_alpha"], sigma=params["elastic_sigma"]
                ),
            ),
            iaa.Rot90(params["rotate_count"]),
            iaa.Sometimes(
                params["gaussian_noise_prob"],
                iaa.AdditiveGaussianNoise(
                    loc=0,
                    scale=(params["gaussian_noise_min"], params["gaussian_noise_max"]),
                ),
            ),
            iaa.Sometimes(
                params["gaussian_blur_prob"],
                iaa.GaussianBlur(
                    sigma=(params["gaussian_blur_min"], params["gaussian_blur_max"]),
                ),
            ),
            iaa.Sometimes(
                params["contrast_prob"],
                iaa.LinearContrast(
                    (params["contrast_min"], params["contrast_max"]), per_channel=True
                ),
            ),
        ]
    )
    return augmentation_pipeline
