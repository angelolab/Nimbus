from cell_classification.loss import Loss
import numpy as np
import tensorflow as tf


def test_loss():
    pred = tf.constant(np.random.rand(1, 256, 256, 1))
    target = tf.constant(np.random.randint(0, 2, size=(1, 256, 256, 1)))
    loss_fn = Loss("BinaryCrossentropy", False)
    loss = loss_fn(target, pred)

    # check if loss has the right shape
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == (1, 256, 256)
    # check if loss is in the right range
    assert tf.reduce_mean(loss).numpy() >= 0

    # check if wrapper works with label_smoothing
    loss_fn = Loss("BinaryCrossentropy", False, label_smoothing=0.1)
    loss_smoothed = loss_fn(target, pred)

    # check if loss is a scalar
    assert isinstance(loss_smoothed, tf.Tensor)
    assert loss_smoothed.shape == (1, 256, 256)
    # check if loss is in the right range
    assert tf.reduce_mean(loss_smoothed).numpy() >= 0
    # check if smoothed loss is reasonably near the unsmoothed loss
    assert np.isclose(loss_smoothed.numpy().mean(), loss.numpy().mean(), atol=0.1)

    # check if loss is zero when target and pred are equal
    loss_fn = Loss("BinaryCrossentropy", False)
    loss = loss_fn(target, tf.cast(target, tf.float32))
    assert loss.numpy().mean() == 0

    # check if loss is zero when target is 2
    loss_fn = Loss("BinaryCrossentropy", True)
    loss = loss_fn(2 * np.ones_like(target), pred)
    assert loss.numpy().mean() == 0

    # check if works with FocalLoss
    loss_fn = Loss("BinaryFocalCrossentropy", True, label_smoothing=0.1, gamma=2)
    loss = loss_fn(target, pred)
    assert loss.numpy().mean() >= 0

    # check if config is returned correctly
    config = loss_fn.get_config()
    assert config["loss_name"] == "BinaryFocalCrossentropy"
    assert config["label_smoothing"] == 0.1
    assert config["gamma"] == 2
    assert config["selective_masking"]
