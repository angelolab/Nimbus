import os
import argparse
import tensorflow as tf
import toml
from augmentation_pipeline import prepapre_tf_aug, py_aug, get_augmentation_pipeline
from segmentation_data_prep import parse_dict, feature_description
from deepcell.model_zoo.panopticnet import PanopticNet
from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler, get_callbacks, count_gpus
from deepcell import losses


def semantic_loss(n_classes):
    def _semantic_loss(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes)
    return _semantic_loss

def prep_batches(batch):
    """ Preprocess batches for training
    Args:
        batch (dict):
            Dictionary of numpy arrays
    Returns:
        inputs (tf.Tensor):
            Batch of images
        targets (tf.Tensor):
            Batch of labels
    """
    inputs = tf.concat([batch["mplex_img"], tf.cast(batch["binary_mask"], tf.float32)], axis=-1)
    targets = batch["marker_activity_mask"]
    return inputs, targets
    


tf.executing_eagerly()


def train(params):    

    # make datasets and splits
    dataset = tf.data.TFRecordDataset(params["record_path"])
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
    dataset = dataset.map(parse_dict)

    # split into train and validation
    validation_dataset = dataset.take(params["num_validation"])
    train_dataset = dataset.skip(params["num_validation"])
    
    # shuffle, batch and augment the training data
    train_dataset = train_dataset.shuffle(params["shuffle_buffer_size"]).batch(
        params["batch_size"])
    augmentation_pipeline = get_augmentation_pipeline(params)
    tf_aug = prepapre_tf_aug(augmentation_pipeline)
    train_dataset = train_dataset.map(lambda x: py_aug(x, tf_aug))
    train_dataset = train_dataset.map(prep_batches)
    validation_dataset = validation_dataset.batch(params["batch_size"])
    validation_dataset = validation_dataset.map(prep_batches)
    
    
    classes = {
    'marker_positive': 2,
    }
    params["experiment"]
    params["num_epochs"]
    
    optimizer = Adam(learning_rate=params["lr"], clipnorm=0.001)
    lr_sched = rate_scheduler(lr=params["lr"], decay=0.99)

    model = PanopticNet(
        backbone='resnet50',
        input_shape=params["input_shape"],
        norm_method='std',
        num_semantic_classes=classes)
    
    loss = {}
    # Give losses for all of the semantic heads
    for layer in model.layers:
        if layer.name.startswith('semantic_'):
            n_classes = layer.output_shape[-1]
            loss[layer.name] = semantic_loss(n_classes)
    
    model.compile(loss=loss, optimizer=optimizer)
    # prepare folders
    model_dir = os.path.join(os.path.normpath(params["path"]), params["experiment"])
    log_dir = os.path.join(model_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(model_dir, '{}.h5'.format(params["experiment"]))
    loss_path = os.path.join(model_dir, '{}.npz'.format(params["experiment"]))

    num_gpus = count_gpus()
    print('Training on', num_gpus, 'GPUs.')

    train_callbacks = get_callbacks(
        model_path,
        lr_sched=lr_sched,
        tensorboard_log_dir=log_dir,
        save_weights_only=num_gpus >= 2,
        monitor='val_loss',
        verbose=1)

    loss_history = model.fit(
        train_dataset,
        epochs=params["num_epochs"],
        validation_data=validation_dataset,
        callbacks=train_callbacks)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        default="cell_classification/configs/params.toml",
    )
    args = parser.parse_args()
    params = toml.load(args.params)
    train(params)
