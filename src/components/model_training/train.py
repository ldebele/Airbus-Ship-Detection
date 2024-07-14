
import os
import sys
import logging
import argparse
from typing import Tuple

import mlflow 
from mlflow.tensorflow import MlflowCallback

import numpy as np
import tensorflow as tf

sys.path.append('./')
from utils import *
from unet_model import build_unet


EXPERIMENT_NAME = "Airbus-Ship-Detection"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__MODEL_TRAINING__")

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.tensowflow.autolog(
    log_datasets=False,
    save_model_kwargs="h5",
    checkpoint=True,
)




# def tune_learning_rate(train, model, epochs: int, learning_rate: float):
#     """
#         Args:
#             train:
#             model:
#             epochs: int
#             learning_rate: float
        
#         Return:
#             history: 
#     """

#     # set learning rate scheduler
#     lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#         lambda epochs: 1e-8 * 10 ** (epochs/10))

#     # compile the model
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss=dice_coef_loss,
#         metrics=[dice_coef]
#     )

#     history = model.fit(train, epochs=epochs, callbacks=[lr_schedule])

#     return history



def compile_model(model, train_data, val_data, epochs: int, learning_rate: float):
    """
        Args:
            model:
            train:
            val:
            epochs: int
        
        Return:
            history:
    """

    
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06),
        loss = dice_coef_loss,
        metrics=[dice_coef])
    
    mlflow.log_param("optimizer", "adam")


    # define early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=5,
                                                 verbose=1,
                                                 mode='max',
                                                 restore_best_weights=True)

    # save checkpoints
    checkpoint_path = './models/checkpoint'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    
    # define model checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_freq='epoch')

    # fit the model
    with mlflow.start_run() as run:
        history = model.fit(train_data,
                            validation_data=val_data,
                            epochs=epochs,
                            callbacks=[early_stopping, checkpoint, MlflowCallback(run)])


    return history



def train(train_dataset: np.ndarray,
          val_dataset: np.ndarray,
          n_classes: int, 
          img_shape: Tuple[int, int, int], 
          epochs: int, 
          learning_rate: float,
          outputs_dir: str) -> None:
    """
        Args:
            train_dataset (np.ndarray):
            val_dataset (np.ndarray)
            n_classes (int):
            img_shape (Tuple(int, int, int)):
            epochs (int):
            learning_rate (float):
    """

    # build the unet model
    model = build_unet(n_classes=n_classes, height=img_shape[0], width=img_shape[1], channel=img_shape[2])

    # with mlflow.start_run:
    #     mlflow.log_param("Learning Rate", learning_rate)
    #     mlflow.log_param("Epochs", epochs)
    #     mlflow.log_param("Batch_size", )
        

    # compile and fit the model.
    history = compile_model(model, train_dataset, val_dataset, epochs, learning_rate)

        # # Accessing training and validation metrics
        # training_loss = history.history["loss"]
        # training_accuracy = history.history["accuracy"]
        # validation_loss = history.history["val_loss"]
        # validation_accuracy = history.history["val_accuracy"]

        # # Log metrics per epoch
        # for epoch in range(len(training_loss)):
        #     mlflow.log_metric("train_loss", training_loss[epoch], step=epoch)
        #     mlflow.log_metric("train_accuracy", training_accuracy[epoch], step=epoch)
        #     mlflow.log_metric("val_loss", validation_loss[epoch], step=epoch)
        #     mlflow.log_metric("val_accuracy", validation_accuracy[epoch], step=epoch)

    # plot the accuracy and loss of the unet model
    plot_filename = plot_history(history, type="loss")
    mlflow.log_artifact(plot_filename)
    plot_filename = plot_history(history, type="accuracy")
    mlflow.log_artifact(plot_filename)




def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trai-dir", type=str, required=True, help="path to the training dataset.")
    parser.add_argument("--val-dir", type=str, required=True, help="path to validation directory.")
    parser.add_argument("--img-shape", type=Tuple(int, int, int), default=(640, 640, 3), help="image shape")
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("--learning-rate", required=True, type=float, help="learning rate.")
    parser.add_argument("--outputs-dir", type=str, default="./outputs", help="path to the output directory.")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting model training....")
