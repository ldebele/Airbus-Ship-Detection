
import tensorflow as tf




def dice_coeff(y_true, y_pred, smooth = 1):
    """
    Calculate the dice coefficient, a metric for measuring the similarity between two sets.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        smooth: Smoothing factor to prevent division by zero.

    Returns:
        A tensor of shape (batch_size, ) representing the dice coefficient.
    """

    intersection = tf.keras.backend.sum(y_true * y_pred, axis=-1)
    union = tf.keras.backend.sum(y_true, axis=-1) + tf.keras.backend.sum(y_pred, axis=-1)
    dice_coeff = (2 * intersection + smooth) / (union + smooth)

    return dice_coeff


def dice_coeff_loss(y_true, y_pred):
    """
    Calculate the dice coefficient loss.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        A tensor of shape (batch_size, ) representing the dice coefficient loss.
    """
    return 1 - dice_coeff(y_true, y_pred)
