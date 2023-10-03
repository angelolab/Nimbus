# based on https://github.com/jakeret/unet/blob/master/src/unet/unet.py
from typing import Optional, Union, Callable, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal


class ConvBlock(layers.Layer):

    def __init__(
            self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation,
            data_format, **kwargs
        ):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.padding=padding
        self.activation=activation
        self.data_format=data_format

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding,
                                      data_format=data_format)
        self.dropout_1 = layers.Dropout(rate=dropout_rate)
        self.activation_1 = layers.Activation(activation)

        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding,
                                      data_format=data_format)
        self.dropout_2 = layers.Dropout(rate=dropout_rate)
        self.activation_2 = layers.Activation(activation)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)

        if training:
            x = self.dropout_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)

        if training:
            x = self.dropout_2(x)

        x = self.activation_2(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    data_format=self.data_format,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(layers.Layer):

    def __init__(
            self, layer_idx, filters_root, kernel_size, pool_size, padding, activation,
            data_format, **kwargs
        ):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation
        self.data_format=data_format

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.upconv = layers.Conv2DTranspose(
            filters // 2,
            kernel_size=(pool_size, pool_size), strides=pool_size, padding=padding,
            kernel_initializer=_get_kernel_initializer(filters, kernel_size),
            data_format=data_format
        )

        self.activation_1 = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
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

    def __init__(self, data_format, **kwargs):
        super(CropConcatBlock, self).__init__(**kwargs)
        self.data_format = data_format

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        if self.data_format == "channels_last":
            height_diff = (x1_shape[1] - x2_shape[1]) // 2
            width_diff = (x1_shape[2] - x2_shape[2]) // 2
            down_layer_cropped = down_layer[:,
                                            height_diff: (x2_shape[1] + height_diff),
                                            width_diff: (x2_shape[2] + width_diff), :]
            x = tf.concat([down_layer_cropped, x], axis=-1)
        elif self.data_format == "channels_first":
            height_diff = (x1_shape[2] - x2_shape[2]) // 2
            width_diff = (x1_shape[3] - x2_shape[3]) // 2
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
                dropout_rate: int = 0.5,
                padding:str="valid",
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
    :param dropout_rate: rate of dropout
    :param padding: padding to be used in convolutions
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
                       dropout_rate=dropout_rate,
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
                      padding=padding,
                      data_format=data_format)(x)

    x = layers.Activation(activation)(x)
    outputs = layers.Activation("softmax", name="outputs")(x)
    model = Model(inputs, outputs, name="unet")

    return model


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)
