
import logging
import argparse
from typing import Tuple

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from sklearn.model_selection import train_test_split

from utils import rle2mask, wrangle_df, save_tfrecord


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__PREPROCESSING__")




def generator(dataframe, images_dir, img_shape):
    """
    Function that creates a generator function for yielding images and masks.

    Args:
        dataframe (pd.DataFrame): DataFrame.
        images_dir (str): Path to the images directory.
        img_shape (tuple): Desired images and masks size.

    Returns:
        generator (tuple): generator that yields (image, mask).
    """

    def _generator():            
        for i, (_, row) in enumerate(dataframe.iterrows()):
            image_path = f"{images_dir}/{row.ImageId}"
            # Read the image
            image = cv.imread(image_path)
            
            # Check if there are encoded pixels for the mask
            if not pd.isna(row.EncodedPixels):
                # Convert encoded mask to a binary mask
                mask = rle2mask(row.EncodedPixels, image.shape[:2])
            else:
                # If no encoded pixels, create an empty mask
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
            # Resize and Normalize the image
            image = cv.resize(image, img_shape)
            image = image / 255.0
            
            # Resize the mask
            mask = cv.resize(mask, img_shape, interpolation=cv.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

            yield image, mask

    return _generator


def create_tf_dataset(dataframe, images_dir, img_shape, batch):
    """
    Function that creates a TensorFlow dataset.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image file names and RLE encoded masks.
        images_dir (str): Path to the images directory.
        img_shape (tuple): Desired images and masks size.
        batch (int): Number of batch size.

    Returns:
        dataset (tf.data.Dataset): TensorFlow dataset.
    """

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        generator(dataframe, images_dir, img_shape),
        output_signature=(
            tf.TensorSpec(shape=(img_shape[0], img_shape[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(img_shape[0], img_shape[1], 1), dtype=tf.float32)
        )
    )

    # Batch the dataset and prefetch for better performance
    dataset = dataset.batch(batch).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def run(images_dir: str, masks_dir: str, batch: int, img_shape: Tuple[int, int]):
    """
    A main function that runs all the data preprocessing for image segmentation.

    Args:
        images_dir (str): Path to the images directory.
        masks_dir (str): Path to the masks directory.
        img_shape (tuple): Desired shape for the loaded images.

    Returns:
        train_dataset
        val_dataset
    """

    # wrangle the dataframe.
    df = wrangle_df(masks_dir)

    # split data into training and validation sets.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info("Split the dataset into training and validation sets.")

    # preprocess the training and validation data.
    train_dataset = create_tf_dataset(train_df, images_dir, img_shape, batch)
    val_dataset = create_tf_dataset(val_df, images_dir, img_shape, batch)
    logger.info("Completed the preprocessing of training and validation datasets.")
   
    # save training and validation dataset.
    save_tfrecord(train_dataset, '/mnt/data/train.tfrecord')
    save_tfrecord(val_dataset, '/mnt/data/val.tfrecord')
    logger.info("Successfully saved the training and validation datasets in TFRecord format.")


def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True, help="Path to the train directory.")
    parser.add_argument("--masks-dir", type=str, required=True, help="Path to the mask directory.")
    parser.add_argument("--batch", type=int, default=8, help="Number of batch size.")
    parser.add_argument("--img-shape", type=tuple, default=(640, 640), help="Image shape")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting preprocessing data....")
    args = opt_parser()

    run(args.images_dir, 
        args.masks_dir, 
        args.batch,
        args.img_shape
    )
