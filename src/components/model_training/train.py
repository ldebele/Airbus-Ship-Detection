
import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('./')
from utils import *
from unet_model import build_unet



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__MODEL_TRAINING__")



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


    return history



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
