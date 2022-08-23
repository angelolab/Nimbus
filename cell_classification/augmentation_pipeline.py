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
    masks = [SegmentationMapsOnImage(mask, shape=mask.shape) for mask in masks]
    batch = Batch(images=images, segmentation_maps=masks)
    batch = augmentation_pipeline.augment_batch_(batch)
    augmented_masks = [mask.arr for mask in batch.segmentation_maps_aug]
    augmented_masks = np.stack(augmented_masks, 0)

    # remove additional channel from single channel masks
    if augmented_masks.shape[-1] == 1:
        augmented_masks = np.squeeze(augmented_masks, -1)
    return np.stack(batch.images_aug, 0), augmented_masks


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
            # random mirroring along horizontal and vertical axis
            iaa.Fliplr(params["flip_prob"]),
            iaa.Flipud(params["flip_prob"]),
            # random zooming and shearing
            iaa.Sometimes(
                params["affine_prob"],
                iaa.Affine(
                    scale=(params["scale_min"], params["scale_max"]),
                    shear=(-params["shear_angle"], params["shear_angle"]),
                ),
            ),
            # elastic transformations that apply a water-like effect onto the image
            iaa.Sometimes(
                params["elastic_prob"],
                iaa.ElasticTransformation(
                    alpha=params["elastic_alpha"], sigma=params["elastic_sigma"]
                ),
            ),
            # 90 degree rotations
            iaa.Rot90(params["rotate_count"]),
            # random gaussian noise added to the image
            iaa.Sometimes(
                params["gaussian_noise_prob"],
                iaa.AdditiveGaussianNoise(
                    loc=0,
                    scale=(params["gaussian_noise_min"], params["gaussian_noise_max"]),
                ),
            ),
            # random blurring with a gaussian filter
            iaa.Sometimes(
                params["gaussian_blur_prob"],
                iaa.GaussianBlur(
                    sigma=(params["gaussian_blur_min"], params["gaussian_blur_max"]),
                ),
            ),
            # random up-scaling and down-scaling of the image intensities
            iaa.Sometimes(
                params["contrast_prob"],
                iaa.LinearContrast(
                    (params["contrast_min"], params["contrast_max"]), per_channel=True
                ),
            ),
        ]
    )
    return augmentation_pipeline
