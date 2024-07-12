
import os
import logging
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from utils import rle2mask



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def wrangle_df(masks_dir):
    """
    A function wrangle the dataframe.

    Args:
        masks_dir (str): path to the masks directory.
    
    Returns:
        df (pd.DataFrame): 
    """

    df = pd.read_csv(masks_dir)

    df = (df
          .groupby("ImageId")["EncodedPixels"]
          .apply(lambda x: ' '.join(x.dropna()))
          .reset_index()
    )

    return df


def load_data(images_dir: str,
              masks_dir: str,
              img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Loads image

    Args:
        
        images_dir (str):
        masks_dir (str):
        img_shape (tuple): 
    
    Returns:

    """

    df = wrangle_df(masks_dir)

    images = []
    masks = []

    for index, row in df.iterrows():
        row.ImageId, row.EncodedPixels

        img_path = f"{images_dir}/{row.ImageId}"
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=None)
        image = tf.keras.preprocessing.image.img_to_array(image)

        mask = rle2mask(row.EncodedPixels, image.shape)

        images.append(image)
        masks.append(mask)


    return np.array(images), np.array(masks)



def split_dataset(images: np.ndarray, masks: np.ndarray):
    """
    A function that split the dataset and create a tensorflow datasets.

    Args:
        images (np.ndarray)
        masks (np.ndarray)

    Return:
        images 
        masks
    """
    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    return train_dataset, val_dataset



def preprocess(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image (np.ndarray)
        mask (np.ndarray)

    Return:
        image (np.ndarray)
        mask (np.ndarray)
    
    """

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
    A main function to run all the preprocess.

    Args:
        images_dir (str):
        masks_dir (str):
        img_shape (tuple):
        batch (int): 

    Returns:
        train_dataset
        val_dataset
    """

    images, masks = load_data(images_dir, masks_dir)
    
    train_dataset, val_dataset = split_dataset(images, masks)

    train_dataset = train_dataset.map(preprocess)
    train_dataset = train_dataset.map(lambda image, mask: (data_augmentation(image, mask)))
    train_dataset = train_dataset.batch(batch).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(preprocess)
    val_dataset = val_dataset.batch(batch).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset



def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True, help="Path to the train directory.")
    parser.add_argument("--masks-dir", type=str, required=True, help="Path to the mask directory.")
    parser.add_argument("--img-shape", type=tuple, default=(640, 640), help="Image shape")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")

    return parser.parse_args()



if __name__ == "__main__":

    args = parser_opt()

    run(args.images_dir, 
        args.masks_dir, 
        args.img_shape, 
        args.batch
    )



