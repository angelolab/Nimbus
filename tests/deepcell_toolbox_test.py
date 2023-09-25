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
"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product

import numpy as np
from skimage.measure import label
from skimage.morphology import binary_dilation

import pytest

from deepcell import deepcell_toolbox as utils


def ceil(x):
    return int(np.ceil(x))


def round_to_even(x):
    return int(np.ceil(x / 2.0) * 2)


def test_resize():
    base_shape = (32, 32)
    out_shapes = [
        (40, 40),
        (42, 40),
        (40, 42),
        (24, 24),
        (16, 24),
        (24, 16),
        (17, 37),
    ]
    channel_sizes = (1, 3)

    for out in out_shapes:
        for c in channel_sizes:
            # batch, channel first
            c = tuple([c])
            in_shape = c + base_shape + (4,)
            out_shape = c + out + (4,)
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_first')
            assert out_shape == rs.shape

            # batch, channel last
            in_shape = (4,) + base_shape + c
            out_shape = (4,) + out + c
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_last')
            assert out_shape == rs.shape

            # no batch, channel first
            in_shape = c + base_shape
            out_shape = c + out
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_first')
            assert out_shape == rs.shape

            # no batch, channel last
            in_shape = base_shape + c
            out_shape = out + c
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_last')
            assert out_shape == rs.shape

            # make sure label data is not linearly interpolated and returns only the same ints

            # no batch, channel last
            in_shape = base_shape + c
            out_shape = out + c
            in_data = np.random.choice(a=[0, 1, 9, 20], size=in_shape, replace=True)
            rs = utils.resize(in_data, out, data_format='channels_last', labeled_image=True)
            assert out_shape == rs.shape
            assert np.all(rs == np.floor(rs))
            assert np.all(np.unique(rs) == [0, 1, 9, 20])

            # batch, channel first
            in_shape = c + base_shape + (4,)
            out_shape = c + out + (4,)
            in_data = np.random.choice(a=[0, 1, 9, 20], size=in_shape, replace=True)
            rs = utils.resize(in_data, out, data_format='channels_first', labeled_image=True)
            assert out_shape == rs.shape
            assert np.all(rs == np.floor(rs))
            assert np.all(np.unique(rs) == [0, 1, 9, 20])

    # Wrong data size
    with pytest.raises(ValueError):
        im = np.random.rand(20, 20)
        out_shape = (10, 10)
        rs = utils.resize(im, out_shape)

    # Wrong shape
    with pytest.raises(ValueError):
        im = np.random.rand(20, 20, 1)
        out_shape = (10, 10, 1)
        rs = utils.resize(im, out_shape, data_format='channels_last')


def test_tile_image():
    shapes = [
        (4, 21, 21, 1),
        (4, 21, 31, 2),
        (4, 31, 21, 3),
    ]
    model_input_shapes = [(3, 3), (5, 5), (7, 7), (12, 12)]

    stride_ratios = [0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 1]

    dtypes = ['int32', 'float32', 'uint16', 'float16']

    prod = product(shapes, model_input_shapes, stride_ratios, dtypes)

    for shape, input_shape, stride_ratio, dtype in prod:
        big_image = (np.random.random(shape) * 100).astype(dtype)
        tiles, tiles_info = utils.tile_image(
            big_image, input_shape,
            stride_ratio=stride_ratio)

        assert tiles.shape[1:] == input_shape + (shape[-1],)
        assert tiles.dtype == big_image.dtype

        image_size_x, image_size_y = big_image.shape[1:3]
        tile_size_x = input_shape[0]
        tile_size_y = input_shape[1]

        stride_x = round_to_even(stride_ratio * tile_size_x)
        stride_y = round_to_even(stride_ratio * tile_size_y)

        if stride_x > tile_size_x:
            stride_x = tile_size_x
        if stride_y > tile_size_y:
            stride_y = tile_size_y

        rep_number_x = ceil((image_size_x - tile_size_x) / stride_x + 1)
        rep_number_y = ceil((image_size_y - tile_size_y) / stride_y + 1)

        expected_batches = big_image.shape[0] * rep_number_x * rep_number_y

        assert tiles.shape[0] == expected_batches

    # test bad input shape
    bad_shape = (21, 21, 1)
    bad_image = (np.random.random(bad_shape) * 100)
    with pytest.raises(ValueError):
        utils.tile_image(bad_image, (5, 5), stride_ratio=0.75)


def test_untile_image():
    shapes = [
        (3, 8, 16, 2),
        (1, 64, 64, 1),
        (1, 41, 58, 1),
        (1, 93, 61, 1)
    ]
    rand_rel_diff_thresh = 2e-2
    model_input_shapes = [(16, 20), (32, 32), (41, 51), (64, 64), (100, 90)]
    stride_ratios = [0.33, 0.5, 0.51, 0.66, 0.75, 1]
    dtypes = ['int32', 'float32', 'uint16', 'float16']
    prod = product(shapes, model_input_shapes, stride_ratios, dtypes)

    # Test that randomly generated arrays are unchanged within a moderate tolerance
    for shape, input_shape, stride_ratio, dtype in prod:

        big_image = (np.random.random(shape) * 100).astype(dtype)
        tiles, tiles_info = utils.tile_image(big_image,
                                             model_input_shape=input_shape,
                                             stride_ratio=stride_ratio)

        untiled_image = utils.untile_image(tiles, tiles_info)

        assert untiled_image.dtype == dtype
        assert untiled_image.shape == shape

        np.testing.assert_allclose(big_image, untiled_image,
                                   rand_rel_diff_thresh)

    # Test that constant arrays are unchanged by tile/untile
    for shape, input_shape, stride_ratio, dtype in prod:
        for x in [0, 1, np.random.randint(2, 99)]:
            big_image = np.empty(shape).astype(dtype).fill(x)
            tiles, tiles_info = utils.tile_image(big_image,
                                                 model_input_shape=input_shape,
                                                 stride_ratio=stride_ratio)
            untiled_image = utils.untile_image(tiles, tiles_info)
            assert untiled_image.dtype == dtype
            assert untiled_image.shape == shape
            np.testing.assert_equal(big_image, untiled_image)

    # test that a stride_fraction of 0 raises an error
    with pytest.raises(ValueError):

        big_image_test = np.zeros((4, 4)).astype('int32')
        tiles, tiles_info = utils.tile_image(big_image_test, model_input_shape=(2, 2),
                                             stride_ratio=0)
        untiled_image = utils.untile_image(tiles, tiles_info)


def test_fill_holes():
    example_arr = np.zeros((50, 50), dtype='int')
    example_arr[:5, :5] = 1
    example_arr[10:20, 10:20] = 2
    example_arr[30:40, 30:40] = 3
    example_arr[30:40, 40:50] = 4

    # create hole of size 4
    example_arr[2:4, 2:4] = 0

    # create hole of size 25
    example_arr[12:17, 12:17] = 0

    # create hole that borders two cells
    example_arr[32:34, 38:40] = 0

    filled = utils.fill_holes(label_img=example_arr, size=5)

    # small hole has been filled
    assert np.sum(filled == 1) == 25

    # large hole has not been filled
    assert np.sum(filled == 2) == 75

    # hole bordering other cell has not been filled
    assert np.all(filled[32:34, 38:40] == 0)

    # set size so that large hole is filled
    filled = utils.fill_holes(label_img=example_arr, size=26)
    assert np.sum(filled == 2) == 100

    # set size so that small hole is not filled
    filled = utils.fill_holes(label_img=example_arr, size=3)
    assert np.sum(filled == 1) == 21


def test_tile_image_3D():
    shapes = [
        (3, 5, 21, 21, 1),
        (1, 10, 21, 31, 2),
        (1, 15, 31, 21, 1),
    ]
    model_input_shapes = [(4, 3, 4), (3, 5, 5), (3, 7, 7), (5, 12, 15)]

    stride_ratios = [0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 1]

    dtypes = ['int32', 'float32', 'uint16', 'float16']

    prod = product(shapes, model_input_shapes, stride_ratios, dtypes)

    for shape, input_shape, stride_ratio, dtype in prod:
        big_image = (np.random.random(shape) * 100).astype(dtype)
        tiles, tiles_info = utils.tile_image_3D(
            big_image, input_shape,
            stride_ratio=stride_ratio)

        assert tiles.shape[1:] == input_shape + (shape[-1],)
        assert tiles.dtype == big_image.dtype

        image_size_z, image_size_x, image_size_y = big_image.shape[1:4]
        tile_size_z = input_shape[0]
        tile_size_x = input_shape[1]
        tile_size_y = input_shape[2]

        stride_x = round_to_even(stride_ratio * tile_size_x)
        stride_y = round_to_even(stride_ratio * tile_size_y)
        stride_z = round_to_even(stride_ratio * tile_size_z)

        if stride_z > tile_size_z:
            stride_z = tile_size_z

        if stride_x > tile_size_x:
            stride_x = tile_size_x

        if stride_y > tile_size_y:
            stride_y = tile_size_y

        rep_number_x = ceil((image_size_x - tile_size_x) / stride_x + 1)
        rep_number_y = ceil((image_size_y - tile_size_y) / stride_y + 1)
        rep_number_z = ceil((image_size_z - tile_size_z) / stride_z + 1)

        expected_batches = big_image.shape[0] * rep_number_x * rep_number_y * rep_number_z

        assert tiles.shape[0] == expected_batches

    # test bad input shape
    bad_shape = (21, 21, 1)
    bad_image = (np.random.random(bad_shape) * 100)
    with pytest.raises(ValueError):
        utils.tile_image(bad_image, (5, 5), stride_ratio=0.75)


def test_untile_image_3D():
    shapes = [
        (1, 30, 60, 51, 2),
        (2, 20, 90, 30, 1)
    ]

    rand_rel_diff_thresh = 2e-2
    model_input_shapes = [(4, 60, 70), (30, 20, 30), (70, 40, 50)]

    stride_ratios = [0.33, 0.5, 0.51, 0.66, 1]
    dtypes = ['int32', 'float32', 'uint16', 'float16']
    power = 3

    prod = product(shapes, model_input_shapes, stride_ratios, dtypes)

    # Test that randomly generated arrays are unchanged within a moderate tolerance
    for shape, input_shape, stride_ratio, dtype in prod:

        big_image = (np.random.random(shape) * 100).astype(dtype)
        tiles, tiles_info = utils.tile_image_3D(big_image,
                                                model_input_shape=input_shape,
                                                stride_ratio=stride_ratio)

        untiled_image = utils.untile_image_3D(tiles, tiles_info, power=power)   # add utils

        assert untiled_image.dtype == dtype
        assert untiled_image.shape == shape

        np.testing.assert_allclose(big_image, untiled_image, rand_rel_diff_thresh)

    # Test that constant arrays are unchanged by tile/untile
    for shape, input_shape, stride_ratio, dtype in prod:
        for x in [0, 1, np.random.randint(2, 99)]:
            big_image = np.empty(shape).astype(dtype).fill(x)
            tiles, tiles_info = utils.tile_image_3D(big_image,
                                                    model_input_shape=input_shape,
                                                    stride_ratio=stride_ratio)
            untiled_image = utils.untile_image_3D(tiles, tiles_info, power=power)
            assert untiled_image.dtype == dtype
            assert untiled_image.shape == shape
            np.testing.assert_equal(big_image, untiled_image)

    # test that a stride_fraction of 0 raises an error
    big_image_test = np.zeros((4, 4)).astype('int32')
    with pytest.raises(ValueError, match="Expected image of rank 4, got 2"):
        tiles, tiles_info = utils.tile_image(big_image_test, model_input_shape=(2, 2),
                                             stride_ratio=0)
