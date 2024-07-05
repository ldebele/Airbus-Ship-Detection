import os
import datetime
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt




def dice_coef(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true)
    dice = np.mean(2*intersection / union)

    return round(dice, 3)



def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def save_model(model, outputs: str = './models'):
    """
        Args:
            model:
            outputs: str
    """

    timestamp = datetime.datetime.today()
    file_path = os.path.join(outputs, f'{str(timestamp)}_unet.h5')
    model.save(file_path)



def plot_model(model, filename):
    tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)


def plot_history(history, type, y):
    """
        Args:
            history : List[float]         -> train and validation values
            type : str                    -> evaluation type [loss, accuracy]
            y : tuple(float, float)       -> y coordinate values (y limiter)
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
    plt.savefig('./outputs/{type}.jpg')


def plot_learning_rate(history):
    """
        Args:
            history : List[float]       -> 
    """

    plt.figure(figsize=(10, 6))
    plt.semelogx(history.history['lr'], history.history['loss'])
    plt.tick_params('both', length=10, width=1, which='both')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate")
    plt.grid(True)
    plt.savefig('./outputs/Learning_rate.jpg')
