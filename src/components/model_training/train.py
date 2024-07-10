
import os
import sys
import logging
import argparse

import mlflow 

import numpy as np
import tensorflow as tf

sys.path.append('./')
from utils import *
from unet_model import build_unet


EXPERIMENT_NAME = ""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__MODEL_TRAINING__")

mlflow.set_experiment(EXPERIMENT_NAME)


def build_model(n_classes: int, height: int, width: int, channel: int):
    """
        Args:
            n_classes: int          -> number of classes
            height : int            -> image height
            widht : int             -> image widht
            channel : int           -> number of channel
    """

    model = build_unet(n_classes=n_classes, height=height, width=width, channel=channel)

    return model



def tune_learning_rate(train, model, epochs: int, learning_rate: float):
    """
        Args:
            train:
            model:
            epochs: int
            learning_rate: float
        
        Return:
            history: 
    """

    # set learning rate scheduler
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epochs: 1e-8 * 10 ** (epochs/10))

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=dice_coef_loss,
        metrics=[dice_coef]
    )

    history = model.fit(train, epochs=epochs, callbacks=[lr_schedule])

    return history



def compile_model(model, train, val, epochs: int, learning_rate: float):
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = dice_coef_loss,
        metrics=[dice_coef])
    
    mlflow.log_param("optimizer", "adam")


    # define callback
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
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
    history = model.fit(train,
                        validation_data=val,
                        epochs=epochs,
                        callbacks=[earlystop, checkpoint])
    
    # # save model
    # model_filename = "airbus-ship-detection-unet.h5"
    # model.save(model_filename)    
    # mlflow.log_artifact(model_filename)

    mlflow.tensorflow.log_model(
        model, "model", keras_model_kwargs={"save_format": "h5"}
    )


    return history



def train(n_classes: int, height: int, width: int, channel: int, train: list, val: list, epochs: int, learning_rate: float):

    # build the unet model
    model = build_unet(n_classes=n_classes, height=height, width=width, channel=channel)

    with mlflow.start_run:
        mlflow.log_param("Learning Rate", learning_rate)
        mlflow.log_param("Epochs", epochs)
        mlflow.log_param("Batch_size", )
        

        # compile and fit the model.
        history = compile_model(model, train, val, epochs, learning_rate)

        # Accessing training and validation metrics
        training_loss = history.history["loss"]
        training_accuracy = history.history["accuracy"]
        validation_loss = history.history["val_loss"]
        validation_accuracy = history.history["val_accuracy"]

        # Log metrics per epoch
        for epoch in range(len(training_loss)):
            mlflow.log_metric("train_loss", training_loss[epoch], step=epoch)
            mlflow.log_metric("train_accuracy", training_accuracy[epoch], step=epoch)
            mlflow.log_metric("val_loss", validation_loss[epoch], step=epoch)
            mlflow.log_metric("val_accuracy", validation_accuracy[epoch], step=epoch)

        # plot the accuracy and loss of the unet model
        plot_filename = plot_history(history, type="loss")
        mlflow.log_artifact(plot_filename)
        plot_filename = plot_history(history, type="accuracy")
        mlflow.log_artifact(plot_filename)











def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trai-dir", type=str, required=True, help="path to the training dataset.")
    parser.add_argument("--val-dir", type=str, required=True, help="path to validation directory.")
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("--learning-rate", required=True, type=float, help="learning rate.")
    parser.add_argument("--outputs-dir", type=str, default="./outputs", help="path to the output directory.")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting model training....")
