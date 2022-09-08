import os
import argparse
import tensorflow as tf
import toml
from augmentation_pipeline import prepapre_tf_aug, py_aug, get_augmentation_pipeline
from segmentation_data_prep import parse_dict, feature_description
from data_generator import DataGenerator
from segmentation_data_prep import parse_dict, feature_description
from deepcell.model_zoo.panopticnet import PanopticNet
from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler, get_callbacks, count_gpus


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
    
    
#     classes = {
#     'inner_distance': 1,  # inner distance
#     'outer_distance': 1,  # outer distance
#     'fgbg': 2,  # foreground/background separation
#     }

#     model_name = 'watershed_centroid_nuclear_general_std'

#     n_epoch = 5  # Number of training epochs

#     lr = 1e-3
#     optimizer = Adam(learning_rate=lr, clipnorm=0.001)
#     lr_sched = rate_scheduler(lr=lr, decay=0.99)

#     model = PanopticNet(
#         backbone='resnet50',
#         input_shape=X_train.shape[1:],
#         norm_method='std',
#         num_semantic_classes=classes)
    
#     loss = {}
#     # Give losses for all of the semantic heads
#     for layer in model.layers:
#         if layer.name.startswith('semantic_'):
#             n_classes = layer.output_shape[-1]
#             loss[layer.name] = semantic_loss(n_classes)
    
#     model.compile(loss=loss, optimizer=optimizer)
#     model_path = os.path.join(MODEL_DIR, '{}.h5'.format(model_name))
#     loss_path = os.path.join(MODEL_DIR, '{}.npz'.format(model_name))

#     num_gpus = count_gpus()
#     print('Training on', num_gpus, 'GPUs.')

#     train_callbacks = get_callbacks(
#         model_path,
#         lr_sched=lr_sched,
#         tensorboard_log_dir=LOG_DIR,
#         save_weights_only=num_gpus >= 2,
#         monitor='val_loss',
#         verbose=1)

#     loss_history = model.fit(
#         train_dataset,
#         steps_per_epoch=train_dataset.y.shape[0] // batch_size,
#         epochs=n_epoch,
#         validation_data=validation_dataset,
#         validation_steps=validation_dataset.y.shape[0] // batch_size,
#         callbacks=train_callbacks)



# def semantic_loss(n_classes):
#     def _semantic_loss(y_true, y_pred):
#         if n_classes > 1:
#             return 0.01 * losses.weighted_categorical_crossentropy(
#                 y_true, y_pred, n_classes=n_classes)
#         return MSE(y_true, y_pred)
#     return _semantic_loss



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
