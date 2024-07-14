import os
import datetime
from typing import Tuple

import tensorflow as tf
import matplotlib.pyplot as plt




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


def load_tfrecord(file_path):
    """Load dataset from TFRecord."""
    dataset = tf.data.TFRecordDataset(file_path)
    


def save_model(model):
    """Saves the trained model."""

    output_dir = "/mnt/data/models"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.today()
    file_path = os.path.join(output_dir, f'{str(timestamp)}_unet.h5')
    model.save(file_path)

    return model


def plot_model(model, filename):
    """Saves a plot for the model architecture."""
    tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)


def plot_history(history, eval_type: str, y: Tuple[int, int]) -> str:
    """
    Plots training vs. validation for evaluation metric.

        Args:
            history (List[float]): Train and validation values.
            eval_type (str): Evaluation type [dice_coeff_loss, dice_coeff].
            y : tuple(float, float): Coordinate values (y limiter).

        Returns:
            plot_filename (str): path to the saved file.
    """


    plt.figure(figsize=(5, 4))
    plt.plot(history.history[type], label=f"Training {type}")
    plt.plot(history.history[f'val_{type}'], label=f"Validation {type}")
    plt.xlabel("Epoch")
    plt.ylabel(type)
    plt.ylim(y)
    plt.title(f"Training vs. Validation {type.capitalize()}")
    plt.grid(True)
    plt.legend(loc='upper left')

    outputs_dir = "/mnt/data/results"
    os.makedirs(outputs_dir, exist_ok=True)

    plot_filename = f"{outputs_dir}/training_validation_{eval_type}.png"
    plt.savefig(plot_filename)

    return plot_filename


def plot_learning_rate(history):
    """Plots the learning rate curve."""

    plt.figure(figsize=(10, 6))
    plt.semelogx(history.history['lr'], history.history['loss'])
    plt.tick_params('both', length=10, width=1, which='both')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate")
    plt.grid(True)

    outputs_dir = "/mnt/data/results"
    os.makedirs(outputs_dir, exist_ok=True)

    plot_filename = f"{outputs_dir}/Learning_rate.png"
    plt.savefig(plot_filename)

    return plot_filename
