
import cv2 as cv
import numpy as np
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.01, seed=42),
    tf.keras.RandomContrast(factor=0.2, seed=42)
])


def rle2mask(mask_rle, shape):
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



def overlay_mask(image, mask, overlay_color=(0, 0, 255), alpha=0.5):
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








