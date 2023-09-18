# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-data-processing/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility functions that may be used in other transforms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from scipy.signal import windows

from skimage import transform
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes
from skimage.segmentation import find_boundaries


def resize(data, shape, data_format='channels_last', labeled_image=False):
    """Resize the data to the given shape.
    Uses openCV to resize the data if the data is a single channel, as it
    is very fast. However, openCV does not support multi-channel resizing,
    so if the data has multiple channels, use skimage.

    Args:
        data (np.array): data to be reshaped. Must have a channel dimension
        shape (tuple): shape of the output data in the form (x,y).
            Batch and channel dimensions are handled automatically and preserved.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.
        labeled_image (bool): flag to determine how interpolation and floats are handled based
         on whether the data represents raw images or annotations

    Raises:
        ValueError: ndim of data not 3 or 4
        ValueError: Shape for resize can only have length of 2, e.g. (x,y)

    Returns:
        numpy.array: data reshaped to new shape.
    """
    if len(data.shape) not in {3, 4}:
        raise ValueError('Data must have 3 or 4 dimensions, e.g. '
                         '[batch, x, y], [x, y, channel] or '
                         '[batch, x, y, channel]. Input data only has {} '
                         'dimensions.'.format(len(data.shape)))

    if len(shape) != 2:
        raise ValueError('Shape for resize can only have length of 2, e.g. (x,y).'
                         'Input shape has {} dimensions.'.format(len(shape)))

    original_dtype = data.dtype

    # cv2 resize is faster but does not support multi-channel data
    # If the data is multi-channel, use skimage.transform.resize
    channel_axis = 0 if data_format == 'channels_first' else -1
    batch_axis = -1 if data_format == 'channels_first' else 0

    # Use skimage for multichannel data
    if data.shape[channel_axis] > 1:
        # Adjust output shape to account for channel axis
        if data_format == 'channels_first':
            shape = tuple([data.shape[channel_axis]] + list(shape))
        else:
            shape = tuple(list(shape) + [data.shape[channel_axis]])

        # linear interpolation (order 1) for image data, nearest neighbor (order 0) for labels
        # anti_aliasing introduces spurious labels, include only for image data
        order = 0 if labeled_image else 1
        anti_aliasing = not labeled_image

        _resize = lambda d: transform.resize(d, shape, mode='constant', preserve_range=True,
                                             order=order, anti_aliasing=anti_aliasing)
    # single channel image, resize with cv2
    else:
        shape = tuple(shape)[::-1]  # cv2 expects swapped axes.

        # linear interpolation for image data, nearest neighbor for labels
        # CV2 doesn't support ints for linear interpolation, set to float for image data
        if labeled_image:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
            data = data.astype('float32')

        _resize = lambda d: np.expand_dims(cv2.resize(np.squeeze(d), shape,
                                                      interpolation=interpolation),
                                           axis=channel_axis)

    # Check for batch dimension to loop over
    if len(data.shape) == 4:
        batch = []
        for i in range(data.shape[batch_axis]):
            d = data[i] if batch_axis == 0 else data[..., i]
            batch.append(_resize(d))
        resized = np.stack(batch, axis=batch_axis)
    else:
        resized = _resize(data)

    return resized.astype(original_dtype)


def tile_image(image, model_input_shape=(512, 512),
               stride_ratio=0.75, pad_mode='constant'):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The image to tile, must be rank 4.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile size.
        pad_mode (str): Padding mode passed to ``np.pad``.

    Returns:
        tuple: (numpy.array, dict): A tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).

    Raises:
        ValueError: image is not rank 4.
    """
    if image.ndim != 4:
        raise ValueError('Expected image of rank 4, got {}'.format(image.ndim))

    image_size_x, image_size_y = image.shape[1:3]
    tile_size_x = model_input_shape[0]
    tile_size_y = model_input_shape[1]

    ceil = lambda x: int(np.ceil(x))
    round_to_even = lambda x: int(np.ceil(x / 2.0) * 2)

    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_x * rep_number_y

    tiles_shape = (new_batch_size, tile_size_x, tile_size_y, image.shape[3])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, pad_mode)

    counter = 0
    batches = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_x = []
    overlaps_y = []

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                x_axis = 1
                y_axis = 2

                # Compute the start and end for each tile
                if i != rep_number_x - 1:  # not the last one
                    x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                else:
                    x_start, x_end = image.shape[x_axis] - tile_size_x, image.shape[x_axis]

                if j != rep_number_y - 1:  # not the last one
                    y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                else:
                    y_start, y_end = image.shape[y_axis] - tile_size_y, image.shape[y_axis]

                # Compute the overlaps for each tile
                if i == 0:
                    overlap_x = (0, tile_size_x - stride_x)
                elif i == rep_number_x - 2:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - image.shape[x_axis] + x_end)
                elif i == rep_number_x - 1:
                    overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                else:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                if j == 0:
                    overlap_y = (0, tile_size_y - stride_y)
                elif j == rep_number_y - 2:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - image.shape[y_axis] + y_end)
                elif j == rep_number_y - 1:
                    overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                else:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                tiles[counter] = image[b, x_start:x_end, y_start:y_end, :]
                batches.append(b)
                x_starts.append(x_start)
                x_ends.append(x_end)
                y_starts.append(y_start)
                y_ends.append(y_end)
                overlaps_x.append(overlap_x)
                overlaps_y.append(overlap_y)
                counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches
    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['overlaps_x'] = overlaps_x
    tiles_info['overlaps_y'] = overlaps_y
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['tile_size_x'] = tile_size_x
    tiles_info['tile_size_y'] = tile_size_y
    tiles_info['stride_ratio'] = stride_ratio
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype
    tiles_info['pad_x'] = pad_x
    tiles_info['pad_y'] = pad_y

    return tiles, tiles_info


def spline_window(window_size, overlap_left, overlap_right, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (windows.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (windows.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window


def window_2D(window_size, overlap_x=(32, 32), overlap_y=(32, 32), power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    window_x = np.expand_dims(np.expand_dims(window_x, -1), -1)
    window_y = np.expand_dims(np.expand_dims(window_y, -1), -1)

    window = window_x * window_y.transpose(1, 0, 2)
    return window


def untile_image(tiles, tiles_info, power=2, **kwargs):
    """Untile a set of tiled images back to the original model shape.

     Args:
         tiles (numpy.array): The tiled images image to untile.
         tiles_info (dict): Details of how the image was tiled (from tile_image).
         power (int): The power of the window function

     Returns:
         numpy.array: The untiled image.
     """
    # Define mininally acceptable tile_size and stride_ratio for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']
    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    overlaps_x = tiles_info['overlaps_x']
    overlaps_y = tiles_info['overlaps_y']
    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    stride_ratio = tiles_info['stride_ratio']
    x_pad = tiles_info['pad_x']
    y_pad = tiles_info['pad_y']

    image_shape = [image_shape[0], image_shape[1], image_shape[2], tiles.shape[-1]]
    window_size = (tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=float)

    window_cache = {}
    for x, y in zip(overlaps_x, overlaps_y):
        if (x, y) not in window_cache:
            w = window_2D(window_size, overlap_x=x, overlap_y=y, power=power)
            window_cache[(x, y)] = w

    for tile, batch, x_start, x_end, y_start, y_end, overlap_x, overlap_y in zip(
            tiles, batches, x_starts, x_ends, y_starts, y_ends, overlaps_x, overlaps_y):

        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (min_tile_size <= tile_size_x < image_shape[1] and
                min_tile_size <= tile_size_y < image_shape[2] and
                stride_ratio >= min_stride_ratio):
            window = window_cache[(overlap_x, overlap_y)]
            image[batch, x_start:x_end, y_start:y_end, :] += tile * window
        else:
            image[batch, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = x_pad[0]
    y_start = y_pad[0]
    x_end = image_shape[1] - x_pad[1]
    y_end = image_shape[2] - y_pad[1]

    image = image[:, x_start:x_end, y_start:y_end, :]

    return image


def tile_image_3D(image, model_input_shape=(10, 256, 256), stride_ratio=0.5):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The 3D image to tile, must be rank 5.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile sizet

    Returns:
        tuple(numpy.array, dict): An tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).

    Raises:
        ValueError: image is not rank 5.
    """
    if image.ndim != 5:
        raise ValueError('Expected image of 5, got {}'.format(
            image.ndim))

    image_size_z, image_size_x, image_size_y = image.shape[1:4]
    tile_size_z = model_input_shape[0]
    tile_size_x = model_input_shape[1]
    tile_size_y = model_input_shape[2]

    ceil = lambda x: int(np.ceil(x))
    round_to_even = lambda x: int(np.ceil(x / 2.0) * 2)

    stride_z = min(round_to_even(stride_ratio * tile_size_z), tile_size_z)
    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_z = max(ceil((image_size_z - tile_size_z) / stride_z + 1), 1)
    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_z * rep_number_x * rep_number_y

    # catches error caused by interpolation along z axis with rep number = 1
    # TODO - create a better solution or figure out why it doesn't occur in x and y planes
    if rep_number_z == 1:
        stride_z = tile_size_z

    tiles_shape = (new_batch_size, tile_size_z, tile_size_x, tile_size_y, image.shape[4])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile along each axis
    overlap_z = (tile_size_z + stride_z * (rep_number_z - 1)) - image_size_z
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_z = (int(np.ceil(overlap_z / 2)), int(np.floor(overlap_z / 2)))
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_z, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, 'constant', constant_values=0)

    counter = 0
    batches = []
    z_starts = []
    z_ends = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_z = []
    overlaps_x = []
    overlaps_y = []
    z_axis = 1
    x_axis = 2
    y_axis = 3

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                for k in range(rep_number_z):
                    # Compute the start and end for each tile
                    if i != rep_number_x - 1:  # not the last one
                        x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                    else:
                        x_start, x_end = image.shape[x_axis] - tile_size_x, image.shape[x_axis]

                    if j != rep_number_y - 1:  # not the last one
                        y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                    else:
                        y_start, y_end = image.shape[y_axis] - tile_size_y, image.shape[y_axis]

                    if k != rep_number_z - 1:  # not the last one
                        z_start, z_end = k * stride_z, k * stride_z + tile_size_z
                    else:
                        z_start, z_end = image.shape[z_axis] - tile_size_z, image.shape[z_axis]

                    # Compute the overlaps for each tile
                    if i == 0:
                        overlap_x = (0, tile_size_x - stride_x)
                    elif i == rep_number_x - 2:
                        overlap_x = (tile_size_x - stride_x,
                                     tile_size_x - image.shape[x_axis] + x_end)
                    elif i == rep_number_x - 1:
                        overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                    else:
                        overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                    if j == 0:
                        overlap_y = (0, tile_size_y - stride_y)
                    elif j == rep_number_y - 2:
                        overlap_y = (tile_size_y - stride_y,
                                     tile_size_y - image.shape[y_axis] + y_end)
                    elif j == rep_number_y - 1:
                        overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                    else:
                        overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                    if k == 0:
                        overlap_z = (0, tile_size_z - stride_z)
                    elif k == rep_number_z - 2:
                        overlap_z = (tile_size_z - stride_z,
                                     tile_size_z - image.shape[z_axis] + z_end)
                    elif k == rep_number_z - 1:
                        overlap_z = ((k - 1) * stride_z + tile_size_z - z_start, 0)
                    else:
                        overlap_z = (tile_size_z - stride_z, tile_size_z - stride_z)

                    tiles[counter] = image[b, z_start:z_end, x_start:x_end, y_start:y_end, :]
                    batches.append(b)
                    x_starts.append(x_start)
                    x_ends.append(x_end)
                    y_starts.append(y_start)
                    y_ends.append(y_end)
                    z_starts.append(z_start)
                    z_ends.append(z_end)
                    overlaps_x.append(overlap_x)
                    overlaps_y.append(overlap_y)
                    overlaps_z.append(overlap_z)
                    counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches
    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['z_starts'] = z_starts
    tiles_info['z_ends'] = z_ends
    tiles_info['overlaps_x'] = overlaps_x
    tiles_info['overlaps_y'] = overlaps_y
    tiles_info['overlaps_z'] = overlaps_z
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['stride_z'] = stride_z
    tiles_info['tile_size_x'] = tile_size_x
    tiles_info['tile_size_y'] = tile_size_y
    tiles_info['tile_size_z'] = tile_size_z
    tiles_info['stride_ratio'] = stride_ratio
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype
    tiles_info['pad_x'] = pad_x
    tiles_info['pad_y'] = pad_y
    tiles_info['pad_z'] = pad_z

    return tiles, tiles_info


def window_3D(window_size, overlap_z=(5, 5), overlap_x=(32, 32), overlap_y=(32, 32), power=3):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_z = spline_window(window_size[0], overlap_z[0], overlap_z[1], power=power)
    window_x = spline_window(window_size[1], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[2], overlap_y[0], overlap_y[1], power=power)

    window_z = np.expand_dims(np.expand_dims(np.expand_dims(window_z, -1), -1), -1)
    window_x = np.expand_dims(np.expand_dims(np.expand_dims(window_x, -1), -1), -1)
    window_y = np.expand_dims(np.expand_dims(np.expand_dims(window_y, -1), -1), -1)

    window = window_z * window_x.transpose(1, 0, 2, 3) * window_y.transpose(1, 2, 0, 3)

    return window


def untile_image_3D(tiles, tiles_info, power=3, force=False, **kwargs):
    """Untile a set of tiled images back to the original model shape.

     Args:
         tiles (numpy.array): The tiled images image to untile.
         tiles_info (dict): Details of how the image was tiled (from tile_image).
         power (int): The power of the window function
         force (bool): If set to True, forces use spline interpolation regardless of
                       tile size or stride_ratio.

     Returns:
         numpy.array: The untiled image.
     """
    # Define mininally acceptable tile_size and stride_ratios for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    if force:
        min_tile_size = 0
        min_stride_ratio = 0

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']

    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    z_starts = tiles_info['z_starts']
    z_ends = tiles_info['z_ends']

    overlaps_x = tiles_info['overlaps_x']
    overlaps_y = tiles_info['overlaps_y']
    overlaps_z = tiles_info['overlaps_z']

    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    tile_size_z = tiles_info['tile_size_z']
    pad_x = tiles_info['pad_x']
    pad_y = tiles_info['pad_y']
    pad_z = tiles_info['pad_z']

    image_shape = tuple(list(image_shape[:4]) + [tiles.shape[-1]])
    window_size = (tile_size_z, tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=float)

    tile_data_zip = zip(tiles, batches, x_starts, x_ends, y_starts,
                        y_ends, z_starts, z_ends, overlaps_x, overlaps_y, overlaps_z)

    for (tile, batch, x_start, x_end, y_start, y_end, z_start,
         z_end, overlap_x, overlap_y, overlap_z) in tile_data_zip:

        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (min_tile_size <= tile_size_x < image_shape[2] and
                min_tile_size <= tile_size_y < image_shape[3] and
                min_stride_ratio <= stride_ratio):

            window = window_3D(window_size, overlap_z=overlap_z, overlap_x=overlap_x,
                               overlap_y=overlap_y, power=power)
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] += tile * window
        else:
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = pad_x[0]
    y_start = pad_y[0]
    z_start = pad_z[0]
    x_end = image_shape[2] - pad_x[1]
    y_end = image_shape[3] - pad_y[1]
    z_end = image_shape[1] - pad_z[1]

    image = image[:, z_start:z_end, x_start:x_end, y_start:y_end, :]

    return image


def fill_holes(label_img, size=10, connectivity=1):
    """Fills holes located completely within a given label with pixels of the same value

    Args:
        label_img (numpy.array): a 2D labeled image
        size (int): maximum size for a hole to be filled in
        connectivity (int): the connectivity used to define the hole

    Returns:
        numpy.array: a labeled image with no holes smaller than ``size``
            contained within any label.
    """
    output_image = np.copy(label_img)

    props = regionprops(np.squeeze(label_img.astype('int')), cache=False)
    for prop in props:
        if prop.euler_number < 1:

            patch = output_image[prop.slice]

            filled = remove_small_holes(
                ar=(patch == prop.label),
                area_threshold=size,
                connectivity=connectivity)

            output_image[prop.slice] = np.where(filled, prop.label, patch)

    return output_image