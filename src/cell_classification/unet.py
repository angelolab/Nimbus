# based on https://github.com/jakeret/unet/blob/master/src/unet/unet.py
from typing import Optional, Union, Callable, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal, HeNormal


class Pad2D(layers.Layer):
    def __init__(self, padding=(1, 1), data_format="channels_last", mode="VALID", **kwargs):
        """ Padding for 2D input (e.g. images).
        Args:
            padding: tuple of 2 ints, how many zeros to add at the beginning and at the end of
                the 2 padding dimensions (rows and cols)
            data_format: channels_last or channels_first
            mode: "VALID", "CONSTANT", "REFLECT", or "SYMMETRIC"
        """
        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        if mode not in ["CONSTANT", "REFLECT", "SYMMETRIC", "VALID"]:
            raise ValueError("Padding mode must be 'VALID', 'CONSTANT', 'REFLECT', or 'SYMMETRIC'")
        self.padding = tuple(padding)
        self.data_format = data_format
        self.mode = mode
        super(Pad2D, self).__init__(**kwargs)

    def get_config(self):
        """Returns the config of a Pad2D."""
        return dict(padding=self.padding,
                    data_format=self.data_format,
                    mode=self.mode,
                    **super(Pad2D, self).get_config(),
                    )

    def get_output_shape_for(self, s):
        """Returns the output shape after reflection padding was applied.
        Args:
            s: shape tuple, (nb_samples, nb_channels, nb_rows, nb_cols)
        Returns:
            tuple of ints
        """
        if self.data_format == "channels_last":
            return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])
        elif self.data_format == "channels_first":
            return (s[0], s[1], s[2] + 2 * self.padding[0], s[3] + 2 * self.padding[1])

    def call(self, x):
        """Apply reflection padding to 2D tensor.
        Args:
            x: tensor of shape (nb_samples, nb_channels, nb_rows, nb_cols)
        Returns:
            padded 2D tensor
        """
        if self.mode == "VALID":
            return x
        w_pad,h_pad = self.padding
        if self.data_format == "channels_last":
            return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], self.mode)
        elif self.data_format == "channels_first":
            return tf.pad(x, [[0,0], [0,0], [h_pad,h_pad], [w_pad,w_pad] ], self.mode)
    

class ConvBlock(layers.Layer):
    """Convolutional block consisting of two convolutional layers with same number of filters
    and a bn layer in between.
    """
    def __init__(
            self, layer_idx, filters_root, kernel_size, padding, activation, data_format, **kwargs
        ):
        """Initialize ConvBlock.
        Args:
            layer_idx: index of the layer, used to compute the number of filters
            filters_root: number of filters in the first convolutional layer
            kernel_size: size of convolutional kernels
            padding: padding, either "VALID", "CONSTANT", "REFLECT", or "SYMMETRIC"
            activation: activation to be used
            data_format: data format, either "channels_last" or "channels_first"
        """
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.padding=padding
        self.activation=activation
        self.data_format=data_format

        filters = _get_filter_count(layer_idx, filters_root)
        self.padding_layer = Pad2D(padding=(1, 1), data_format=data_format, mode=padding)
        self.conv2d_0 = layers.Conv2D(filters=filters,
                                      kernel_size=(1, 1),
                                      kernel_initializer=HeNormal(),
                                      strides=1,
                                      padding="valid",
                                      data_format=data_format)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding="valid",
                                      data_format=data_format)
        self.activation_1 = layers.Activation(activation)
        self.bn_1 = layers.BatchNormalization(axis=1 if data_format == "channels_first" else -1)
        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding="valid",
                                      data_format=data_format)
        self.activation_2 = layers.Activation(activation)
        self.bn_2 = layers.BatchNormalization(axis=1 if data_format == "channels_first" else -1)
        self.add = layers.Add()

    def call(self, inputs, **kwargs):
        """Apply ConvBlock to inputs.
        Args:
            inputs: input tensor
        Returns:
            output tensor
        """
        skip = self.conv2d_0(inputs)
        x = self.padding_layer(skip)
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.activation_1(x)
        x = self.padding_layer(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.add([x, skip])
        return x

    def get_config(self):
        """Returns the config of a ConvBlock."""
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    data_format=self.data_format,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(layers.Layer):
    """Upconvolutional block consisting of an upsampling layer and a convolutional layer.
    """
    def __init__(
            self, layer_idx, filters_root, kernel_size, pool_size, padding, activation,
            data_format, **kwargs
        ):
        """UpconvBlock initializer.
        Args:
            layer_idx: index of the layer, used to compute the number of filters
            filters_root: number of filters in the first convolutional layer
            kernel_size: size of convolutional kernels
            pool_size: size of the pooling layer
            padding: padding, either "VALID", "CONSTANT", "REFLECT", or "SYMMETRIC"
            activation: activation to be used
            data_format: data format, either "channels_last" or "channels_first"
        """
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation
        self.data_format=data_format

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.padding_layer = Pad2D(padding=(1, 1), data_format=data_format, mode=padding)
        self.upconv = layers.Conv2DTranspose(
            filters // 2,
            kernel_size=(pool_size, pool_size), strides=pool_size, padding="valid",
            kernel_initializer=_get_kernel_initializer(filters, kernel_size),
            data_format=data_format
        )

        self.activation_1 = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        # x = self.padding_layer(x)
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    data_format=self.data_format,
                    **super(UpconvBlock, self).get_config(),
                    )

class CropConcatBlock(layers.Layer):
    """CropConcatBlock that makes crops spatial dimensions and concatenates filter maps.
    """
    def __init__(self, data_format, **kwargs):
        """CropConcatBlock initializer.
        Args:
            data_format: data format, either "channels_last" or "channels_first"
        """
        super(CropConcatBlock, self).__init__(**kwargs)
        self.data_format = data_format

    def call(self, x, down_layer, **kwargs):
        """Apply CropConcatBlock to inputs.
        Args:
            x: input tensor
            down_layer: tensor from the contracting path
        Returns:
            output tensor
        """
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        if self.data_format == "channels_last":
            height_diff = tf.abs(x1_shape[1] - x2_shape[1]) // 2 # 64 - 68 = 4 // 2 = 2
            width_diff = tf.abs(x1_shape[2] - x2_shape[2]) // 2
            down_layer_cropped = down_layer[:,
                                            height_diff: (x2_shape[1] + height_diff),
                                            width_diff: (x2_shape[2] + width_diff), :]
            x = tf.concat([down_layer_cropped, x], axis=-1)
        elif self.data_format == "channels_first":
            height_diff = tf.abs(x1_shape[2] - x2_shape[2]) // 2
            width_diff = tf.abs(x1_shape[3] - x2_shape[3]) // 2
            down_layer_cropped = down_layer[:,:,
                                            height_diff: (x2_shape[2] + height_diff),
                                            width_diff: (x2_shape[3] + width_diff)]
            x = tf.concat([down_layer_cropped, x], axis=1)
        return x


def build_model(nx: Optional[int] = None,
                ny: Optional[int] = None,
                channels: int = 1,
                num_classes: int = 2,
                data_format = "channels_first",
                layer_depth: int = 5,
                filters_root: int = 64,
                kernel_size: int = 3,
                pool_size: int = 2,
                padding:str="VALID",
                activation:Union[str, Callable]="relu") -> Model:
    """
    Constructs a U-Net model

    :param nx: (Optional) image size on x-axis
    :param ny: (Optional) image size on y-axis
    :param channels: number of channels of the input tensors
    :param num_classes: number of classes
    :param data_format: data format to be used in convolutions
    :param layer_depth: total depth of unet
    :param filters_root: number of filters in top unet layer
    :param kernel_size: size of convolutional layers
    :param pool_size: size of maxplool layers
    :param padding: padding, either "VALID", "CONSTANT", "REFLECT", or "SYMMETRIC"
    :param activation: activation to be used
    :return: A TF Keras model
    """
    if data_format == "channels_first":
        inputs = Input(shape=(channels, nx, ny), name="inputs")
    elif data_format == "channels_last":
        inputs = Input(shape=(nx, ny, channels), name="inputs")

    x = inputs
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       padding=padding,
                       activation=activation,
                       data_format=data_format)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = layers.MaxPooling2D((pool_size, pool_size),data_format=data_format)(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation,
                        data_format)(x)
        x = CropConcatBlock(data_format)(x, contracting_layers[layer_idx])
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = layers.Conv2D(filters=num_classes,
                      kernel_size=(1, 1),
                      kernel_initializer=_get_kernel_initializer(filters_root, kernel_size),
                      strides=1,
                      padding="valid",
                      data_format=data_format)(x)

    outputs = layers.Activation("sigmoid", name="semantic_head")(x)
    model = Model(inputs, outputs, name="unet")

    return model


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)
