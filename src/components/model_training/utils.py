import tensorflow as tf

import matplotlib.pyplot as plt



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
