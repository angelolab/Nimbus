import tensorflow as tf


class Loss():
    """ Wrapper for loss functions that allows for selective masking of the loss
    """
    def __init__(self, loss_name, selective_masking, **kwargs):
        """ Initialize the loss function
        Args:
            loss_name (str):
                name of the loss function
            selective_masking (bool):
                whether to use selective masking
            **kwargs:
                additional arguments for the loss function
        """
        self.loss_fn = getattr(tf.keras.losses, loss_name)(
            reduction=tf.keras.losses.Reduction.NONE, **kwargs
        )
        self.selective_masking = selective_masking

    def mask_out(self, loss_img, y_true):
        """ Selectively mask the loss by setting it to zero where y_true == -1
        Args:
            loss_img (tf.Tensor):
                loss image
            y_true (tf.Tensor):
                ground truth image
        Returns:
            tf.Tensor:
                masked loss image
        """
        y_true = tf.reshape(y_true, tf.shape(loss_img))
        return tf.where(y_true == 2, tf.zeros_like(loss_img), loss_img)

    def __call__(self, y_true, y_pred):
        """ Call the loss function
        Args:
            y_true (tf.Tensor):
                ground truth image
            y_pred (tf.Tensor):
                prediction image
        Returns:
            tf.Tensor:
                loss image
        """
        loss_img = self.loss_fn(y_true=tf.clip_by_value(y_true, 0, 1), y_pred=y_pred)
        if self.selective_masking:
            loss_img = self.mask_out(loss_img, y_true)
        return loss_img

    def get_config(self):
        """ Get the configuration of the loss function
        Returns:
            dict:
                configuration of the loss function
        """
        return {
            "loss_name": self.loss_fn.__class__.__name__,
            "selective_masking": self.selective_masking,
            **self.loss_fn.get_config(),
        }
