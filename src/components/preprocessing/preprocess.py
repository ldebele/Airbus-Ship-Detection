
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



class DataGenerator(Sequence):
    """Custom data generator"""
    def __init__(self, dataframe, image_dir, batch_size, image_size=(256, 256), **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_df = self.dataframe.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self.__data_generation(batch_df)
        return images, masks

    def on_epoch_end(self):
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, batch_df):
        images = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32)
        masks = np.empty((self.batch_size, *self.image_size, 1), dtype=np.float32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            image_path = f"{self.image_dir}/{row.ImageId}"
            image = cv.imread(image_path)
            
            if not pd.isna(row.EncodedPixels):
                mask = rle2mask(row.EncodedPixels, image.shape[:2])
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            image = cv.resize(image, self.image_size)
            image = image / 255.0
            
            mask = cv.resize(mask, self.image_size, interpolation=cv.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

            images[i,] = image
            masks[i,] = mask

        return images, masks


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
    train_dataset = DataGenerator(train_df, images_dir, batch, img_shape) 
    val_dataset = DataGenerator(val_df, images_dir, batch, img_shape)
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
