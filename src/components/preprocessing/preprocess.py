
import logging
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from utils import rle2mask, wrangle_df


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__PRE-PROCESSING__")



def load_data(images_dir: str,
              masks_dir: str,
              img_shape: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Loads images and masks from directories and load into numpy array.

    Args:
        
        images_dir (str): Path to the directory of the images.
        masks_dir (str): Path to the directory of the masks.
        img_shape (tuple): Shape of the loaded images.
    
    Returns:
        Tuples of array contains the images and masks.

    """

    # wrangle the dataframe.
    df = wrangle_df(masks_dir)

    images = []
    masks = []

    # loof through each row in the dataframe.
    for _, row in df.iterrows():
        
        img_path = f"{images_dir}/{row.ImageId}"
        # load the image
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=img_shape)
        # convert the loaded image to numpy array.
        image = tf.keras.preprocessing.image.img_to_array(image)

        # check if images  contains encoded pixels.
        if not pd.isna(row.EncodedPixels):
            # decode the RLE encoded pixels and create a mask.
            mask = rle2mask(row.EncodedPixels, image.shape)
        else:
            # create an empty mask with the same image size
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # append to the list.
        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


def split_dataset(images: np.ndarray, masks: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Splits the datasets into training and validation sets.

    Args:
        images (np.ndarray): Numpy array containing the image data.
        masks (np.ndarray): Numpy array containing the mask data.

    Return:
        train_dataset (tf.data.Dataset): Trining dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
    """
    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    return train_dataset, val_dataset



def preprocess(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses the image and mask."""

    # convert the image and mask into float32.
    image = tf.image.convert_image_dtype(image, tf.float32)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    return image, mask


def data_augmentation(image, mask):
    """
    A function to augment actual image and masked image.

    Args:
        image (np.ndarray):
        mask (np.ndarray):

    Return:
        image (np.ndarray)
        mask (np.ndarray)
    
    """

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
        tf.keras.layers.RandomRotation(factor=0.01, seed=42),
        tf.keras.RandomContrast(factor=0.2, seed=42)
    ])

    return image, mask



def run(images_dir: str, masks_dir: str, img_shape: Tuple[int, int], batch: int = 8):
    """
    A main function that runs all the data preprocessing for image segmentation.

    Args:
        images_dir (str): Path to the images directory.
        masks_dir (str): Path to the masks directory.
        img_shape (tuple): Desired shape for the loaded images.
        batch (int): Batch size for training and validation datasets.

    Returns:
        train_dataset
        val_dataset
    """

    # Load images and masks.
    images, masks = load_data(images_dir, masks_dir)
    logger.info("Completed loaded images and masks from path into numpy arrays.")

    # split data into training and validation sets.
    train_dataset, val_dataset = split_dataset(images, masks)
    logger.info("Split the dataset into training and validation sets.")

    # preprocess and augment the training data.
    train_dataset = train_dataset.map(preprocess)
    # apply augmentation to the training datasets.
    train_dataset = train_dataset.map(lambda image, mask: (data_augmentation(image, mask)))
    # batch and prefetch training data for efficient training.
    train_dataset = train_dataset.batch(batch).prefetch(tf.data.experimental.AUTOTUNE)

    # preprocess the validation data.
    val_dataset = val_dataset.map(preprocess)
    # batch and prefetch validation data for efficient training.
    val_dataset = val_dataset.batch(batch).prefetch(tf.data.experimental.AUTOTUNE)
    logger.info("Completed the preprocessing and augmentation of the images and masks datasets.")

    return train_dataset, val_dataset


def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True, help="Path to the train directory.")
    parser.add_argument("--masks-dir", type=str, required=True, help="Path to the mask directory.")
    parser.add_argument("--img-shape", type=tuple, default=(640, 640), help="Image shape")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting preprocessing data....")
    args = opt_parser()

    run(args.images_dir, 
        args.masks_dir, 
        args.img_shape, 
        args.batch
    )



