from typing import Tuple

import cv2 as cv
import numpy as np
import pandas as pd



def wrangle_df(masks_dir: str) -> pd.DataFrame:
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

    df["EncodedPixels"] = df["EncodedPixels"].replace("", np.nan)

    return df


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
        img[lo:hi] = 1

    # Reshape the flat array to the specified shape
    return img.reshape((shape[1], shape[0])).T


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








