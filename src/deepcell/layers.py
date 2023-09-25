# Copyright 2016-2023 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/tf-keras-retinanet/LICENSE
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
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from keras.utils import conv_utils

logger = tf.get_logger()


class UpsampleLike(Layer):
    """Layer for upsampling a Tensor to be the same shape as another Tensor.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """

    def __init__(self, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def _resize_drop_axis(self, image, size, axis):
        image_shape = tf.shape(image)

        new_shape = []
        axes_resized = list(set([0, 1, 2, 3, 4]) - set([0, 4, axis]))
        for ax in range(K.ndim(image) - 1):
            if ax != axis:
                new_shape.append(image_shape[ax])
            if ax == 3:
                new_shape.append(image_shape[-1] * image_shape[axis])

        new_shape_2 = []
        for ax in range(K.ndim(image)):
            if ax == 0 or ax == 4 or ax == axis:
                new_shape_2.append(image_shape[ax])
            elif ax == axes_resized[0]:
                new_shape_2.append(size[0])
            elif ax == axes_resized[1]:
                new_shape_2.append(size[1])

        new_image = tf.reshape(image, new_shape)
        new_image_resized = tf.image.resize(
            new_image,
            size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        new_image_2 = tf.reshape(new_image_resized, new_shape_2)

        return new_image_2

    def resize_volumes(self, volume, size):
        # TODO: K.resize_volumes?
        if self.data_format == 'channels_first':
            volume = tf.transpose(volume, (0, 2, 3, 4, 1))
            new_size = (size[2], size[3], size[4])
        else:
            new_size = (size[1], size[2], size[3])

        new_shape_0 = (new_size[1], new_size[2])
        new_shape_1 = (new_size[0], new_size[1])

        resized_volume = self._resize_drop_axis(volume, new_shape_0, axis=1)
        resized_volume = self._resize_drop_axis(resized_volume, new_shape_1, axis=3)

        new_shape_static = [None, None, None, None, volume.get_shape()[-1]]
        resized_volume.set_shape(new_shape_static)

        if self.data_format == 'channels_first':
            resized_volume = tf.transpose(resized_volume, (0, 4, 1, 2, 3))

        return resized_volume

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        if source.get_shape().ndims == 4:
            if self.data_format == 'channels_first':
                source = tf.transpose(source, (0, 2, 3, 1))
                new_shape = (target_shape[2], target_shape[3])
                # TODO: K.resize_images?
                output = tf.image.resize(
                    source, new_shape,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                output = tf.transpose(output, (0, 3, 1, 2))
                return output
            new_shape = (target_shape[1], target_shape[2])
            return tf.image.resize(
                source, new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if source.get_shape().ndims == 5:
            output = self.resize_volumes(source, target_shape)
            return output

        else:
            raise ValueError('Expected input[0] to have ndim of 4 or 5, found'
                             ' %s.' % source.get_shape().ndims)

    def compute_output_shape(self, input_shape):
        in_0 = tensor_shape.TensorShape(input_shape[0]).as_list()
        in_1 = tensor_shape.TensorShape(input_shape[1]).as_list()
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape([in_0[0], in_0[1]] + in_1[2:])
        return tensor_shape.TensorShape([in_0[0]] + in_1[1:-1] + [in_0[-1]])

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ImageNormalization2D(Layer):
    """Image Normalization layer for 2D data.

    Args:
        norm_method (str): Normalization method to use, one of:
            "std", "max", "whole_image", None.
        filter_size (int): The length of the convolution window.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        activation (function): Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: ``a(x) = x``).
        use_bias (bool): Whether the layer uses a bias.
        kernel_initializer (function): Initializer for the ``kernel`` weights
            matrix, used for the linear transformation of the inputs.
        bias_initializer (function): Initializer for the bias vector. If None,
            the default initializer will be used.
        kernel_regularizer (function): Regularizer function applied to the
            ``kernel`` weights matrix.
        bias_regularizer (function): Regularizer function applied to the
            bias vector.
        activity_regularizer (function): Regularizer function applied to.
        kernel_constraint (function): Constraint function applied to
            the ``kernel`` weights matrix.
        bias_constraint (function): Constraint function applied to the
            bias vector.
    """
    def __init__(self,
                 norm_method='std',
                 filter_size=61,
                 data_format=None,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.valid_modes = {'std', 'max', None, 'whole_image'}
        if norm_method not in self.valid_modes:
            raise ValueError(f'Invalid `norm_method`: "{norm_method}". '
                             f'Use one of {self.valid_modes}.')
        if 'trainable' not in kwargs:
            kwargs['trainable'] = False
        super().__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)  # hardcoded for 2D data

        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = conv_utils.normalize_data_format(data_format)

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3  # hardcoded for 2D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4, '
                             'received input shape: %s' % input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})

        kernel_shape = (self.filter_size, self.filter_size, input_dim, 1)
        # self.kernel = self.add_weight(
        #     name='kernel',
        #     shape=kernel_shape,
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     trainable=False,
        #     dtype=self.compute_dtype)

        W = K.ones(kernel_shape, dtype=self.compute_dtype)
        W = W / K.cast(K.prod(K.int_shape(W)), dtype=self.compute_dtype)
        self.kernel = W
        # self.set_weights([W])

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filter_size, self.filter_size),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=False,
                dtype=self.compute_dtype)
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def _average_filter(self, inputs):
        # Depthwise convolution on CPU is only supported for NHWC format
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 1])
        outputs = tf.nn.depthwise_conv2d(inputs, self.kernel, [1, 1, 1, 1],
                                         padding='SAME', data_format='NHWC')
        if self.data_format == 'channels_first':
            outputs = K.permute_dimensions(outputs, pattern=[0, 3, 1, 2])
        return outputs

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(K.square(inputs))
        output = K.sqrt(c2 - c1 * c1) + epsilon
        return output

    def call(self, inputs):
        if not self.norm_method:
            outputs = inputs

        elif self.norm_method == 'whole_image':
            axes = [2, 3] if self.channel_axis == 1 else [1, 2]
            outputs = inputs - K.mean(inputs, axis=axes, keepdims=True)
            outputs = outputs / (K.std(inputs, axis=axes, keepdims=True) + K.epsilon())

        elif self.norm_method == 'std':
            outputs = inputs - self._average_filter(inputs)
            outputs = outputs / self._window_std_filter(outputs)

        elif self.norm_method == 'max':
            outputs = inputs / K.max(inputs)
            outputs = outputs - self._average_filter(outputs)

        else:
            raise NotImplementedError(f'"{self.norm_method}" is not a valid norm_method')

        return outputs

    def get_config(self):
        config = {
            'norm_method': self.norm_method,
            'filter_size': self.filter_size,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Location2D(Layer):
    """Location Layer for 2D cartesian coordinate locations.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self, data_format=None, **kwargs):
        in_shape = kwargs.pop('in_shape', None)
        if in_shape is not None:
            logger.warn('in_shape (from deepcell.layerse.location) is '
                        'deprecated and will be removed in a future version.')
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        channel_axis = 1 if self.data_format == 'channels_first' else 3
        input_shape[channel_axis] = 2
        return tensor_shape.TensorShape(input_shape)

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            x = K.arange(0, input_shape[2], dtype=inputs.dtype)
            y = K.arange(0, input_shape[3], dtype=inputs.dtype)
        else:
            x = K.arange(0, input_shape[1], dtype=inputs.dtype)
            y = K.arange(0, input_shape[2], dtype=inputs.dtype)

        x = x / K.max(x)
        y = y / K.max(y)

        loc_x, loc_y = tf.meshgrid(x, y, indexing='ij')

        if self.data_format == 'channels_first':
            loc = K.stack([loc_x, loc_y], axis=0)
        else:
            loc = K.stack([loc_x, loc_y], axis=-1)

        location = K.expand_dims(loc, axis=0)
        if self.data_format == 'channels_first':
            location = K.permute_dimensions(location, pattern=[0, 2, 3, 1])

        location = tf.tile(location, [input_shape[0], 1, 1, 1])

        if self.data_format == 'channels_first':
            location = K.permute_dimensions(location, pattern=[0, 3, 1, 2])

        return location

    def get_config(self):
        config = {
            'data_format': self.data_format
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Location3D(Layer):
    """Location Layer for 3D cartesian coordinate locations.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self, data_format=None, **kwargs):
        in_shape = kwargs.pop('in_shape', None)
        if in_shape is not None:
            logger.warn('in_shape (from deepcell.layerse.location) is '
                        'deprecated and will be removed in a future version.')
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        channel_axis = 1 if self.data_format == 'channels_first' else 4
        input_shape[channel_axis] = 3
        return tensor_shape.TensorShape(input_shape)

    def call(self, inputs):
        input_shape = K.shape(inputs)

        if self.data_format == 'channels_first':
            z = K.arange(0, input_shape[2], dtype=inputs.dtype)
            x = K.arange(0, input_shape[3], dtype=inputs.dtype)
            y = K.arange(0, input_shape[4], dtype=inputs.dtype)
        else:
            z = K.arange(0, input_shape[1], dtype=inputs.dtype)
            x = K.arange(0, input_shape[2], dtype=inputs.dtype)
            y = K.arange(0, input_shape[3], dtype=inputs.dtype)

        x = x / K.max(x)
        y = y / K.max(y)
        z = z / K.max(z)

        loc_z, loc_x, loc_y = tf.meshgrid(z, x, y, indexing='ij')

        if self.data_format == 'channels_first':
            loc = K.stack([loc_z, loc_x, loc_y], axis=0)
        else:
            loc = K.stack([loc_z, loc_x, loc_y], axis=-1)

        location = K.expand_dims(loc, axis=0)

        if self.data_format == 'channels_first':
            location = K.permute_dimensions(location, pattern=[0, 2, 3, 4, 1])

        location = tf.tile(location, [input_shape[0], 1, 1, 1, 1])

        if self.data_format == 'channels_first':
            location = K.permute_dimensions(location, pattern=[0, 4, 1, 2, 3])

        return location

    def get_config(self):
        config = {
            'data_format': self.data_format
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))