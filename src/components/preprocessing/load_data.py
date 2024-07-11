
import os
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from preprocessing import rle2mask



TRAIN_DIR = ""



def load_data(df: pd.DataFrame,
              img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Loads image

    Args:
        image_path (str)
        mask_paths (str)
        img_shape (tuple): 
    
    Returns:

    """

    images = []
    masks = []

    for index, row in df.iterrows():
        row.ImageId, row.EncodedPixels

        img_path = f"{TRAIN_DIR}/{row.ImageId}"
        image = load_img(img_path, target_size=None)
        image = img_to_array(image)

        mask = rle2mask(row.EncodedPixels, image.shape)

        images.append(image)
        masks.append(mask)


    return np.array(images), np.array(masks)




    

def split_dataset(images: np.ndarray, masks: np.ndarray):
    """
    
    """
    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    return train_dataset, val_dataset