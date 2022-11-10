import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.batches import Batch
import numpy as np
import tensorflow as tf
from keras.layers.preprocessing.image_preprocessing import transform, get_zoom_matrix


def augment_images(images, masks, augmentation_pipeline):
    """
    Augment images and masks.
    Args:
        images (np.array):
            The images (b,h,w,c) to augment
        masks (np.array):
            The masks (b,h,w,c) to augment
        augmentation_pipeline (imgaug.augmenters.meta.Augmenter):
            The augmentation pipeline
    Returns:
        np.array:
            The augmented images
        np.array:
            The augmented masks
    """
    masks_ = [SegmentationMapsOnImage(mask, shape=mask.shape) for mask in masks]
    batch = Batch(images=images, segmentation_maps=masks_)
    batch = augmentation_pipeline.augment_batch_(batch)
    augmented_masks = [mask.arr for mask in batch.segmentation_maps_aug]
    augmented_masks = np.stack(augmented_masks, 0)

    # remove additional channel from single channel masks
    if augmented_masks.shape != masks.shape:
        augmented_masks = np.reshape(augmented_masks, masks.shape)
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


def prepare_tf_aug(augmentation_pipeline):
    def tf_aug(mplex_img, binary_mask, marker_activity_mask):
        """Boiler plate code necessary to apply augmentations onto tf.data.Dataset objects
        Args:
            mplex_img (tf.Tensor):
                The images (b,h,w,c) to augment
            binary_mask (tf.Tensor):
                The masks (b,h,w,c) to augment
            marker_activity_mask (tf.Tensor):
                The masks (b,h,w,c) to augment
        Returns:
            function:
                The augmentation function applicable on the input images and masks
        """
        aug_images, aug_masks = augment_images(
            mplex_img.numpy(),
            np.concatenate([binary_mask, marker_activity_mask], -1),
            augmentation_pipeline,
        )
        mplex_img = aug_images
        binary_mask = aug_masks[..., :1]
        marker_activity_mask = aug_masks[..., 1:]
        return mplex_img, binary_mask, marker_activity_mask

    return tf_aug


def py_aug(batch, tf_aug):
    """Python function wrapper to apply augmentations onto tf.data.Dataset objects
    Args:
        batch (dict):
            The batch to augment
        tf_aug (function):
            The augmentation function
    Returns:
        dict:
            The augmented batch
    """
    mplex_img, binary_mask, marker_activity_mask = tf.py_function(
        tf_aug,
        [batch["mplex_img"], batch["binary_mask"], batch["marker_activity_mask"]],
        [tf.float32, tf.uint8, tf.uint8],
    )
    batch["mplex_img"] = mplex_img
    batch["binary_mask"] = binary_mask
    batch["marker_activity_mask"] = marker_activity_mask
    return batch


class Flip(tf.Module):
    def __init__(self, prob=0.5):
        super(Flip, self).__init__()
        self.prob = prob

    def __call__(self, image, labels):
        if tf.random.uniform(()) < self.prob:
            image = tf.reverse(image, axis=[0])
            labels = tf.reverse(labels, axis=[0])
        if tf.random.uniform(()) < self.prob:
            image = tf.reverse(image, axis=[1])
            labels = tf.reverse(labels, axis=[1])
        return image, labels


class Rot90(tf.Module):
    def __init__(self, prob=0.5, rotate_count=3):
        super(Rot90, self).__init__()
        self.prob = prob
        self.rotate_count = rotate_count

    def __call__(self, image, labels):
        if tf.random.uniform(()) < self.prob:
            k = tf.random.uniform((), minval=1, maxval=self.rotate_count, dtype=tf.int32)
            image = tf.image.rot90(image, k)
            labels = tf.image.rot90(labels, k)
        return image, labels


class GaussianNoise(tf.Module):
    def __init__(self, prob=0.5, min_std=0.1, max_std=0.2):
        super(GaussianNoise, self).__init__()
        self.prob = prob
        self.min_std = min_std
        self.max_std = max_std

    def __call__(self, image, labels):
        if tf.random.uniform(()) < self.prob:
            noise = tf.random.normal(
                shape=tf.shape(image),
                mean=0.0,
                dtype=tf.float32,
                stddev=tf.random.uniform((), self.min_std, self.max_std),
            )
            image += noise
        return image, labels


class GaussianBlur(tf.Module):
    """Gaussian blur augmentation"""

    def __init__(self, prob=0.5, min_std=0.1, max_std=0.2, kernel_size=5):
        """
        Args:
            prob (float):
                The probability of applying the augmentation
            min_std (float):
                The minimum standard deviation of the gaussian filter
            max_std (float):
                The maximum standard deviation of the gaussian filter
            kernel_size (int):
                The size of the kernel
        """
        super(GaussianBlur, self).__init__()
        self.prob = prob
        self.min_std = min_std
        self.max_std = max_std
        self.size = kernel_size

    def gaussian_kernel(self, sigma, n_channels):
        """Returns 2D Gaussian kernel for convolutions.
        Args:
            sigma (float):
                Standard deviation of the gaussian filter
            n_channels (int):
                The number of channels of the image
        Returns:
            tf.Tensor:
                The gaussian kernel
        """
        x = tf.range(-self.size, self.size + 1, dtype=tf.float32)
        g = tf.math.exp(-(x ** 2) / (2.0 * sigma ** 2))
        g_kernel = tf.tensordot(g, g, axes=0)
        g_kernel = g_kernel / tf.reduce_sum(g_kernel)
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        g_kernel = tf.tile(g_kernel, [1, 1, n_channels, 1])
        return g_kernel

    def __call__(self, image, labels):
        """Applies gaussian blurring
        Args:
            image (tf.Tensor):
                The image to blur
            labels (tf.Tensor):
                The labels
        Returns:
            tf.Tensor:
                The blurred image
            tf.Tensor:
                The labels
        """
        squeeze = False
        if tf.rank(image) == 3:
            image = tf.expand_dims(image, axis=0)
            squeeze = True
        if tf.random.uniform(()) < self.prob:
            sigma = tf.random.uniform((), self.min_std, self.max_std)
            kernel = self.gaussian_kernel(sigma, n_channels=tf.shape(image)[-1])
            image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")
        if squeeze:
            image = tf.squeeze(image, axis=0)
        return image, labels


class Zoom(tf.Module):
    """Zoom augmentation"""

    def __init__(self, prob=0.5, min_zoom=0.8, max_zoom=1.2, fill_mode="constant"):
        """
        Args:
            prob (float):
                The probability of applying the augmentation
            min_zoom (float):
                The minimum zoom factor
            max_zoom (float):
                The maximum zoom factor
        """
        super(Zoom, self).__init__()
        self.prob = prob
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.fill_mode = fill_mode

    def __call__(self, image, labels):
        """Applies zoom
        Args:
            image (tf.Tensor):
                The image to zoom
            labels (tf.Tensor):
                The labels
        Returns:
            tf.Tensor:
                The zoomed image
            tf.Tensor:
                The labels
        """
        if tf.random.uniform(()) < self.prob:
            zoom_factor = tf.random.uniform([1], self.min_zoom, self.max_zoom)
            zoom_factor = 1 / zoom_factor
            zooms = tf.expand_dims(
                tf.cast(tf.concat([zoom_factor, zoom_factor], axis=0), dtype=tf.float32), 0
            )
            squeeze = False
            if tf.rank(image) == 3:
                image = tf.expand_dims(image, axis=0)
                labels = tf.expand_dims(labels, axis=0)
                squeeze = True
            _, h, w, _ = image.shape
            image = transform(
                image,
                get_zoom_matrix(zooms, h, w),
                fill_mode=self.fill_mode,
                interpolation="bilinear",
            )
            labels = transform(
                labels,
                get_zoom_matrix(zooms, h, w),
                fill_mode=self.fill_mode,
                interpolation="nearest",
            )
            if squeeze:
                image = tf.squeeze(image, axis=0)
                labels = tf.squeeze(labels, axis=0)
        return image, labels


class LinearContrast(tf.Module):
    """Linear contrast augmentation"""

    def __init__(self, prob=0.5, min_factor=0.5, max_factor=1.5):
        """
        Args:
            prob (float):
            The probability of applying the augmentation
        min_factor (float):
            The minimum contrast factor
        max_factor (float):
            The maximum contrast factor
        """
        super(LinearContrast, self).__init__()
        self.prob = prob
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, image, labels):
        """Apply linear contrast augmentation
        Args:
            image (tf.Tensor):
                The image to augment
                labels (tf.Tensor):
                The labels to augment
        Returns:
            tf.Tensor:
                The augmented image
            tf.Tensor:
                The augmented labels
        """
        if tf.random.uniform(()) < self.prob:
            factor = tf.random.uniform((), self.min_factor, self.max_factor)
            image *= factor
        return image, labels


class MixUp(tf.Module):
    """MixUp augmentation for segmentation data"""

    def __init__(self, prob=0.5, alpha=0.2):
        """
        Args:
            prob (float):
                The probability of applying the augmentation
            alpha (float):
                The alpha parameter of the beta distribution
        """
        super(MixUp, self).__init__()
        self.prob = prob
        self.alpha = alpha

    def __call__(self, x_mplex, x_binary, labels, loss_mask):
        """Apply mixup augmentation
        Args:
            x_mplex (tf.Tensor):
                The batch of images to augment
            x_binary (tf.Tensor):
                The batch of binary images to augment
            labels (tf.Tensor):
                The batch of labels to augment
        Returns:
            tf.Tensor:
                The augmented image batch
            tf.Tensor:
                The augmented label batch
        """
        # cast to float
        loss_mask = tf.cast(loss_mask, tf.float32)
        labels = tf.cast(labels, tf.float32)
        x_binary = tf.cast(x_binary, tf.float32)
        # calculate mixup coefficient beta and prob [0,1] if an image gets transformed
        b = tf.shape(x_mplex)[0]
        prob = tf.random.uniform([b], 0, 1) < self.prob
        beta = tf.random.uniform([b], 0, 1)
        beta = tf.math.maximum(beta, 1 - beta)
        beta = tf.math.pow(beta, 1 / self.alpha)
        beta = tf.where(prob, beta, tf.ones_like(beta))
        prob = tf.cast(tf.reshape(prob, [b, 1, 1, 1]), tf.float32)
        # apply mixup on data
        beta = tf.reshape(beta, [b, 1, 1, 1])
        x_mplex = beta * x_mplex + (1 - beta) * tf.reverse(x_mplex, axis=[0])
        x_binary = beta * x_binary + (1 - beta) * tf.reverse(x_binary, axis=[0])
        labels = beta * labels + (1 - beta) * tf.reverse(labels, axis=[0])
        # loss mask should be 1 where both mixup loss masks == 1 and zero otherwise
        loss_mask_reverse = prob * tf.cast(tf.reverse(loss_mask, axis=[0]), tf.float32)
        # make sure that the loss_mask doesn't change for prob = 0
        loss_mask = prob * (loss_mask * loss_mask_reverse) + tf.math.abs(prob - 1) * loss_mask
        return x_mplex, x_binary, labels, loss_mask


class Augmenter(tf.Module):
    """Augmenter class to apply a list of augmentations to a batch of images and labels"""

    def __init__(self, augmentations, parallel_calls=4, dtype=(tf.float32, tf.int32)):
        """Augmentation module
        Args:
            augmentations (list):
                List of augmentation functions
        """
        super(Augmenter, self).__init__()
        self.augmentations = augmentations
        self.parallel_calls = parallel_calls
        self.dtype = dtype

    def aug_sample(self, sample):
        """Apply augmentations to a single image and label
        Args:
            image (tf.Tensor):
                The image to augment
            labels (tf.Tensor):
                The labels to augment
        Returns:
            tf.Tensor:
                The augmented image
            tf.Tensor:
                The augmented labels
        """
        image, labels = sample
        for aug in self.augmentations:
            image, labels = aug(image, labels)
        return image, labels

    def __call__(self, image_batch, labels_batch):
        """Apply the augmentations to a batch of images and labels
        Args:
            image_batch (tf.Tensor):
                The batch of images to augment
            labels_batch (tf.Tensor):
                The batch of labels to augment
        Returns:
            tf.Tensor:
                The augmented batch of images
            tf.Tensor:
                The augmented batch of labels
        """
        image_batch, labels_batch = tf.map_fn(
            self.aug_sample,
            elems=(image_batch, labels_batch),
            dtype=self.dtype,
            parallel_iterations=self.parallel_calls,
        )
        return image_batch, labels_batch


def prepare_keras_aug(params, parallel_calls=4, dtype=(tf.float32, tf.int32)):
    """Prepare the augmentation pipeline for use within keras
    Args:
        params (dict):
            The parameters for the augmentation
    Returns:
        keras_cv.layers.Augmenter:
            The augmentation pipeline
    """
    augmenter = Augmenter(
        augmentations=[
            Zoom(
                prob=params["affine_prob"],
                min_zoom=params["scale_min"],
                max_zoom=params["scale_max"],
            ),
            Flip(prob=params["flip_prob"]),
            Rot90(prob=params["rotate_prob"], rotate_count=params["rotate_count"]),
            GaussianBlur(
                params["gaussian_blur_prob"],
                min_std=params["gaussian_blur_min"],
                max_std=params["gaussian_blur_max"],
            ),
            LinearContrast(
                prob=params["contrast_prob"],
                min_factor=params["contrast_min"],
                max_factor=params["contrast_max"],
            ),
            GaussianNoise(
                prob=params["gaussian_noise_prob"],
                min_std=params["gaussian_noise_min"],
                max_std=params["gaussian_noise_max"],
            ),
        ],
        parallel_calls=parallel_calls,
        dtype=dtype,
    )
    return augmenter
