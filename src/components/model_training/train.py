
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



def compile_model(model, train_data, val_data, epochs: int, learning_rate: float):
    """
        Compiles and trains the model.

        Args:
            model: The model to compile and train.
            train: Training dataset.
            val: validation dataset.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
        
        Return:
            history: Training history.
    """

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06),
        loss = dice_coeff_loss,
        metrics=[dice_coeff])
    
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

    # fit the model with callbacks and mlflow logging.
    with mlflow.start_run() as run:
        history = model.fit(train_data,
                            validation_data=val_data,
                            epochs=epochs,
                            callbacks=[early_stopping, checkpoint, MlflowCallback(run)])
    
    logger.info("Training model successfully completed.")

    return history



def train(train_dataset: np.ndarray,
          val_dataset: np.ndarray,
          num_classes: int, 
          img_shape: Tuple[int, int, int], 
          epochs: int, 
          learning_rate: float,
          outputs_dir: str) -> None:
    """
        Trains the U-Net model.

        Args:
            train_dataset (np.ndarray): Training data.
            val_dataset (np.ndarray): Validation data.
            n_classes (int): Number of classes.
            img_shape (Tuple(int, int, int)): Image shape.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            outputs_dir (str): Path to the output directory.
    """

    # build the unet model
    model = build_unet(n_classes=num_classes, height=img_shape[0], width=img_shape[1], channel=img_shape[2])

    # compile and fit the model.
    history = compile_model(model, train_dataset, val_dataset, epochs, learning_rate)

    # plot the dice_coeff and loss of the unet model
    plot_filename = plot_history(history, eval_type="loss", outputs_dir=outputs_dir)
    logger.info(f"Dice coefficient loss plot saved in {plot_filename}.")
    plot_filename = plot_history(history, eval_type="dice_coeff", outputs_dir=outputs_dir)
    logger.info(f"Dice coefficient accuracy plot saved in {plot_filename}.")



def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to validation directory.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--img-shape", type=Tuple(int, int, int), default=(640, 640, 3), help="Image shape.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--learning-rate", required=True, type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--outputs-dir", type=str, default="./outputs", help="Path to the output directory.")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting model training....")

    args = opt_parser()

    # train the model
    train(
        train_dataset=args.train_dir,
        val_dataset=args.val_dir,
        num_classes=args.num_classes,
        img_shape=args.img_shape,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        outputs_dir=args.outputs_dir
    )
