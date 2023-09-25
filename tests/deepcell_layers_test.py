# Copyright 2016-2023 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
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
# ==============================================================================
"""Tests for the upsampling layers"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from absl.testing import parameterized
from keras import keras_parameterized
from keras import testing_utils
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test
from tensorflow.keras.utils import custom_object_scope

from deepcell import layers


@keras_parameterized.run_all_keras_modes
class TestUpsampleLike(keras_parameterized.TestCase):

    def test_simple(self):
        # channels_last
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike()

        # create input source
        source = np.zeros((1, 2, 2, 1), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 5, 5, 1), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])

        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)
        # channels_first
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike(
            data_format='channels_first')

        # create input source
        source = np.zeros((1, 1, 2, 2), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 1, 5, 5), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])
        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)

    def test_simple_3d(self):
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike()

        # create input source
        source = np.zeros((1, 2, 2, 2, 1), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 5, 5, 5, 1), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])

        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)

        # channels_first
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike(
            data_format='channels_first')

        # create input source
        source = np.zeros((1, 1, 2, 2, 2), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 1, 5, 5, 5), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])
        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)

    def test_mini_batch(self):
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike()

        # create input source
        source = np.zeros((2, 2, 2, 1), dtype=K.floatx())
        source = K.variable(source)

        target = np.zeros((2, 5, 5, 1), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertAllEqual(actual, expected)


@keras_parameterized.run_all_keras_modes
@parameterized.named_parameters(
    *tf_test_util.generate_combinations_with_testcase_name(
        norm_method=[None, 'std', 'max', 'whole_image']))
class ImageNormalizationTest(keras_parameterized.TestCase):

    def test_normalize_2d(self, norm_method):
        custom_objects = {'ImageNormalization2D': layers.ImageNormalization2D}
        with tf.keras.utils.custom_object_scope(custom_objects):
            testing_utils.layer_test(
                layers.ImageNormalization2D,
                kwargs={'norm_method': norm_method,
                        'filter_size': 3,
                        'data_format': 'channels_last'},
                input_shape=(3, 5, 6, 4))
            testing_utils.layer_test(
                layers.ImageNormalization2D,
                kwargs={'norm_method': norm_method,
                        'filter_size': 3,
                        'data_format': 'channels_first'},
                input_shape=(3, 4, 5, 6))
            # test constraints and bias
            k_constraint = tf.keras.constraints.max_norm(0.01)
            b_constraint = tf.keras.constraints.max_norm(0.01)
            layer = layers.ImageNormalization2D(
                use_bias=True,
                kernel_constraint=k_constraint,
                bias_constraint=b_constraint)
            layer(tf.keras.backend.variable(np.ones((3, 5, 6, 4))))
            # self.assertEqual(layer.kernel.constraint, k_constraint)
            # self.assertEqual(layer.bias.constraint, b_constraint)
            # test bad norm_method
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization2D(norm_method='invalid')
            # test bad input dimensions
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization2D()
                layer.build([3, 10, 11, 12, 4])
            # test invalid channel
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization2D()
                layer.build([3, 5, 6, None])


@keras_parameterized.run_all_keras_modes
class LocationTest(keras_parameterized.TestCase):

    def test_location_2d(self):
        with custom_object_scope({'Location2D': layers.Location2D}):
            testing_utils.layer_test(
                layers.Location2D,
                kwargs={'data_format': 'channels_last'},
                input_shape=(3, 5, 6, 4))
            testing_utils.layer_test(
                layers.Location2D,
                kwargs={'in_shape': (4, 5, 6),
                        'data_format': 'channels_first'},
                input_shape=(3, 4, 5, 6))
