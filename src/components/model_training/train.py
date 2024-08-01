
import os
import logging
import argparse
from typing import Tuple

import mlflow 

import tensorflow as tf

from utils import *
from unet_model import build_unet
from read_tfrecord import ReadTFRecord




EXPERIMENT_NAME = "Airbus-Ship-Detection"
MLFLOW_TRACKING_URI="http://mlflow:5000"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__MODEL_TRAINING__")

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Connected to MLflow tracking server")
except Exception as e:
    logger.error(f"Error connecting to MLflow: {e}")


mlflow.set_experiment(EXPERIMENT_NAME)
# mlflow.tensorflow.autolog(og_datasets=False)


class MLflowMetricsLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=epoch)


def load_dataset(train_dir: str, val_dir: str, batch: int):
    """
    A function that loads the training and validition dataset from TFRecord format.

    Args:
        train_dir (str): Path to the training directory.
        val_dir (str): Path to the validition directory.
        batch (int): Number of batch size.
    
    Returns:
        train_dataset (tf.data.Dataset): Trining dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
    """

    # load training and validation dataset.
    train_dataset = ReadTFRecord.load_tfrecord(train_dir, batch)
    val_dataset = ReadTFRecord.load_tfrecord(val_dir, batch)

    return train_dataset, val_dataset
  

def compile_model(model: tf.keras.models, 
                  train_data: tf.data.Dataset, 
                  val_data: tf.data.Dataset, 
                  epochs: int, 
                  learning_rate: float):
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
    # mlflow.log_param("Epsilon", epsilon)
    mlflow.log_param("Optimizer", "adam")

    # define early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_dice_loss',
                                                      patience=5,
                                                      verbose=1,
                                                      mode='max',
                                                      restore_best_weights=True)

    # save checkpoints
    checkpoint_dir = './outputs/results/models/checkpoint'
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.keras')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Checkpoint saved in the directory {checkpoint_path}")

    # define model checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    monitor='val_dice_loss',
                                                    save_best_only=True,
                                                    save_freq='epoch')

    # fit the model with callbacks and mlflow logging.
    # with mlflow.start_run() as run: 
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs,
                        callbacks=[early_stopping, checkpoint, MLflowMetricsLogger()])
            
    return history


def train(train_dir: str,
          val_dir: str,
          num_classes: int, 
          img_shape: Tuple[int, int, int], 
          batch: int,
          epochs: int, 
          learning_rate: float) -> None:
    """
        Trains the U-Net model.

        Args:
            train_dir (str): Path to the training directory.
            val_dir (str): Path to the validition directory.
            n_classes (int): Number of classes.
            img_shape (Tuple(int, int, int)): Image shape.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
    """
    # load training and validation dataset
    train_dataset, val_dataset = load_dataset(train_dir, val_dir, batch)
    logger.info("Loading training and validation dataset successfully completed.")

    # build the unet model
    model = build_unet(n_classes=num_classes, height=img_shape[0], width=img_shape[1], channel=img_shape[2])
    logger.info("U-net model successfully built.")

    # compile and fit the model.
    with mlflow.start_run():

        mlflow.log_param("Epochs", epochs)
        mlflow.log_param("Batch Size", batch)
        mlflow.log_param("Learning rate", learning_rate)

        
        _ = compile_model(model, train_dataset, val_dataset, epochs, learning_rate)
        logger.info("Training model successfully completed.")

        # Log the model
        mlflow.keras.log_model(model, "model")

        # save the model
        saved_path = save_model(model)
        mlflow.log_artifact(local_path=saved_path, artifact_path="models")
        logger.info(f"The model is successfully saved in the {saved_path}")


def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to validation directory.")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes.")
    parser.add_argument("--img-shape", type=Tuple[int, int, int], default=(512, 512, 3), help="Image shape.")
    parser.add_argument("--batch", type=int, default=8, help="Number of batch size.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate for the optimizer.")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting model training....")

    args = opt_parser()

    # train the model
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        num_classes=args.num_classes,
        img_shape=args.img_shape,
        batch=args.batch,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
