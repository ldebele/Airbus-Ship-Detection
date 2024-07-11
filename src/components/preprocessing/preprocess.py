from typing import Tuple

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf


def retrieve_rle(df: pd.DataFrame, img_name: str) -> str:
    """
    Retrieve the encoded pixels values for the image from the dataframe.

    Args:
        df (pd.DataFrame): a dataframe where the images and encoded pixels values stored.
        img_name (str): name of specific image form the dataframe.

    Returns:
        rle (str): run-length encoded values.
    """

    # select all the encoded pixels for the image.
    df_img = df.loc[df["ImageId"] == img_name]

    # combine encoded pixels values into a single string.
    rle = " ".join(df_img.EncodedPixels.values)

    return rle



def rle2mask(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Converts a run-lenght-encoder (RLE) string to a binary mask.

    Args:
        mask_rle: Run-length encoded string
        shape: Tuple (height, width) of the output mask.

    Returns:
        binary mask of the given shape.
    """

    # split the RLE string and convert to integres.
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]

    # Adjust starts to be zero-indexed
    starts -= 1
    # Compute the ending positions of the runs
    ends = starts + lengths

    # Initialize the mask with zeros
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Set the pixels within the runs to 255
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255

    # Reshape the flat array to the specified shape
    return img.reshape((shape[1], shape[0])).T



def overlay_mask(image: np.ndarray, 
                 mask: np.ndarray, 
                 overlay_color: Tuple[int, int, int] = (0, 0, 255), 
                 alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a segmentation mask on a real image with a solid transparent color on segmented areas.

    Args:
        image (np.ndarray): The real image in BGR format.
        mask (np.ndarray):  The mask image
        overlay_color (tuple): solid color for overlay.
        transparency: value between 0 and 1 for overlay color opacity.

    Returns:
        (np.array): the overlaid image.
    """

    # convert mask to same number of channels as actual image.
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # Create a color image the same size as the original image
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[:] = overlay_color
    
    # Apply the mask to the colored mask
    mask = mask.astype(bool)
    colored_mask = cv.bitwise_and(colored_mask, colored_mask, mask=mask.astype(np.uint8))
    
    # Blend the original image with the colored mask
    overlay = cv.addWeighted(image, 1.0, colored_mask, alpha, 0)
    
    return overlay


def overlay_mask(mask: np.ndarray,
                 image: np.ndarray,
                 overlay_color: Tuple[int, int, int] = (0, 0, 255),
                 alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a segmentation mask on a real image with a solid transparent color on segmented areas.

    Args:
        image (np.array): The real image in BGR format.
        mask (np.array):  The mask image
        overlay_color (tuple): solid color for overlay.
        transparency: value between 0 and 1 for overlay color opacity.

    Returns:
        (np.array): the overlaid image.
    """
    # convert and reshape overlay_color.
    color = np.array(overlay_color).reshape((1, 1, 3))

    # create a colored mask with the same size of the image.
    colored_mask = np.zeros_like(image, dtype=np.uint8)

    # fill the colored mask with the overlay color
    colored_mask[:] = color

    # convert the mask to a boolean array
    ismasked = mask.astype(bool)

    # apply the mask to the colored mask.
    colored_mask = cv.bitwise_and(colored_mask, colored_mask, mask=ismasked.astype(np.uint8))

    # copy the image for overlay
    overlayed_image = image.copy

    # apply the weighted overlay within the masked region
    overlayed_image[mask != 0] = cv.addWeighted(
        image[mask != 0], 1 - alpha, colored_mask[mask != 0], alpha, 0
    )

    return overlayed_image












def data_augmentation():
    """
    
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
        tf.keras.layers.RandomRotation(factor=0.01, seed=42),
        tf.keras.RandomContrast(factor=0.2, seed=42)
    ])




