# adapted from https://github.com/jakeret/unet/blob/master/tests/test_unet.py
from unittest.mock import Mock, patch
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

from src.cell_classification import unet


class TestConvBlock:

    def test_serialization(self):
        conv_block = unet.ConvBlock(layer_idx=1,
                                    filters_root=16,
                                    kernel_size=3,
                                    dropout_rate=0.1,
                                    padding="same",
                                    activation="relu",
                                    name="conv_block_test",
                                    data_format="channels_last")

        config = conv_block.get_config()
        new_conv_block = unet.ConvBlock.from_config(config)

        assert new_conv_block.layer_idx == conv_block.layer_idx
        assert new_conv_block.filters_root == conv_block.filters_root
        assert new_conv_block.kernel_size == conv_block.kernel_size
        assert new_conv_block.dropout_rate == conv_block.dropout_rate
        assert new_conv_block.padding == conv_block.padding
        assert new_conv_block.activation == conv_block.activation
        assert new_conv_block.activation == conv_block.activation
        assert new_conv_block.data_format == conv_block.data_format


class TestUpconvBlock:

    def test_serialization(self):
        upconv_block = unet.UpconvBlock(
            layer_idx=1, filters_root=16, kernel_size=3, pool_size=2, padding="same",
            activation="relu", name="upconv_block_test", data_format="channels_last"
        )

        config = upconv_block.get_config()
        new_upconv_block = unet.UpconvBlock.from_config(config)

        assert new_upconv_block.layer_idx == upconv_block.layer_idx
        assert new_upconv_block.filters_root == upconv_block.filters_root
        assert new_upconv_block.kernel_size == upconv_block.kernel_size
        assert new_upconv_block.pool_size == upconv_block.pool_size
        assert new_upconv_block.padding == upconv_block.padding
        assert new_upconv_block.activation == upconv_block.activation
        assert new_upconv_block.activation == upconv_block.activation
        assert new_upconv_block.data_format == upconv_block.data_format


class TestCropConcatBlock():

    def test_uneven_concat(self):
        layer = unet.CropConcatBlock(data_format="channels_last")
        down_tensor = np.ones([1, 61, 61, 32])
        up_tensor = np.ones([1, 52, 52, 32])

        concat_tensor = layer(up_tensor, down_tensor)

        assert concat_tensor.shape == (1, 52, 52, 64)


class TestUnetModel:

    def test_serialization(self, tmpdir):
        save_path = str(tmpdir / "unet_model")
        unet_model = unet.build_model(layer_depth=3, filters_root=2)
        unet_model.save(save_path)

        reconstructed_model = tf.keras.models.load_model(save_path)
        assert reconstructed_model is not None

    def test_build_model(self):
        nx = 572
        ny = 572
        channels = 3
        num_classes = 2
        kernel_size = 3
        pool_size = 2
        filters_root = 64
        layer_depth = 5
        model = unet.build_model(nx=nx,
                                 ny=ny,
                                 channels=channels,
                                 num_classes=num_classes,
                                 layer_depth=layer_depth,
                                 filters_root=filters_root,
                                 kernel_size=kernel_size,
                                 pool_size=pool_size,
                                 data_format="channels_last")

        input_shape = model.get_layer("inputs").output.shape
        assert tuple(input_shape) == (None, channels, nx, ny)
        output_shape = model.get_layer("outputs").output.shape
        assert tuple(output_shape) == (None, 388, 388, num_classes)

        filters_per_layer = [filters_root, 128, 256, 512, 1024, 512, 256, 128, filters_root]
        conv2D_layers = _collect_conv2d_layers(model)

        assert len(conv2D_layers) == 2 * len(filters_per_layer) + 1

        for conv2D_layer in conv2D_layers[:-1]:
            assert conv2D_layer.kernel_size == (kernel_size, kernel_size)

        for i, filters in enumerate(filters_per_layer):
            assert conv2D_layers[i*2].filters == filters
            assert conv2D_layers[i*2+1].filters == filters

        maxpool_layers = [layer for layer in model.layers if isinstance(layer, layers.MaxPool2D)]

        assert len(maxpool_layers) == layer_depth - 1

        for maxpool_layer in maxpool_layers[:-1]:
            assert maxpool_layer.pool_size == (pool_size, pool_size)


def _collect_conv2d_layers(model):
    conv2d_layers = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            conv2d_layers.append(layer)
        elif isinstance(layer, unet.ConvBlock):
            conv2d_layers.append(layer.conv2d_1)
            conv2d_layers.append(layer.conv2d_2)

    return conv2d_layers
