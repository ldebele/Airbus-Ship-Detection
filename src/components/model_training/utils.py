import os
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt




def dice_coeff(y_true, y_pred, smooth = 1e-6):
    """
    Calculate the dice coefficient, a metric for measuring the similarity between two sets.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        smooth: Smoothing factor to prevent division by zero.

    Returns:
        A tensor of shape (batch_size, ) representing the dice coefficient.
    """
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=-1)
    union = tf.keras.backend.sum(y_true, axis=-1) + tf.keras.backend.sum(y_pred, axis=-1)
    dice_coeff = (2. * intersection + smooth) / (union + smooth)

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


def save_model(model, output_dir = "./outputs/results/models"):
    """Saves the trained model."""

    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y%m%d%H%M")
    file_path = os.path.join(output_dir, f'unet_{formatted_timestamp}.keras')
    model.save(file_path)

    return file_path


def plot_model(model, output_dir: str = './outputs/results'):
    """Saves a plot for the model architecture."""

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'model_summary.png')
    plot = tf.keras.utils.plot_model(model, to_file=file_path, show_shapes=True)

    return file_path


def plot_history(history, eval_type: str, outputs_dir: str = "./outputs/results") -> str:
    """
    Plots training vs. validation for evaluation metric.

        Args:
            history (List[float]): Train and validation values.
            eval_type (str): Evaluation type [dice_coeff_loss, dice_coeff].
            outputs_dir (str): Path to the outputs directory.

        Returns:
            plot_filename (str): path to the saved file.
    """


    plt.figure(figsize=(5, 4))
    plt.plot(history.history[eval_type], label=f"Training {eval_type}")
    plt.plot(history.history[f'val_{eval_type}'], label=f"Validation {eval_type}")
    plt.xlabel("Epoch")
    plt.ylabel(eval_type)
    plt.title(f"Training vs. Validation {eval_type.capitalize()}")
    plt.grid(True)
    plt.legend(loc='upper left')

    os.makedirs(outputs_dir, exist_ok=True)

    plot_filename = f"{outputs_dir}/training_validation_{eval_type}.png"
    plt.savefig(plot_filename)

    return plot_filename


def plot_learning_rate(history, outputs_dir = "./outputs/results"):
    """Plots the learning rate curve."""

    plt.figure(figsize=(10, 6))
    plt.semelogx(history.history['learning_rate'], history.history['loss'])
    plt.tick_params('both', length=10, width=1, which='both')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate")
    plt.grid(True)

    os.makedirs(outputs_dir, exist_ok=True)

    plot_filename = f"{outputs_dir}/Learning_rate.png"
    plt.savefig(plot_filename)

    return plot_filename

