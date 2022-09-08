import os
import argparse
import tensorflow as tf
import toml
from augmentation_pipeline import augment_images, get_augmentation_pipeline
from segmentation_data_prep import parse_dict, feature_description
from data_generator import DataGenerator
from segmentation_data_prep import parse_dict, feature_description



def train(params):
    
    # make datasets and splits
    dataset = tf.data.TFRecordDataset(params['record_path'])
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
    dataset = dataset.map(parse_dict)
    validation_dataset = dataset.take(params['num_validation']) 
    train_dataset = dataset.skip(params['num_validation'])
    train_dataset = train_dataset.shuffle(params['shuffle_buffer_size'])
    augmenter = get_augmentation_pipeline(params)
    train_generator = DataGenerator(train_dataset, params['raw_keys'], params['label_keys'], params['batch_size'],
    augmentation_pipeline=augmenter)
    for batch in train_generator:
        print(batch.keys())
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        default="C:/Users/lorenz/Desktop/angelo_lab/MIBI_test/configs/params.toml",
    )
    args = parser.parse_args()
    params = toml.load(args.params)
    train(params)
