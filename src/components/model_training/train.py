
import os
import sys
import datetime
import numpy as np
import tensorflow as tf

sys.path.append('./')
from build_model import unet



def build_model(n_classes: int, height: int, width: int, channel: int):
    """
        Args:
            n_classes: int          -> number of classes
            height : int            -> image height
            widht : int             -> image widht
            channel : int           -> number of channel
    """

    model = unet(n_classes=n_classes, height=height, width=width, channel=channel)

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



def dice_coef(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true)
    dice = np.mean(2*intersection / union)

    return round(dice, 3)



def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def save_model(model):
    """
        Args:
            model :
    """

    timestamp = datetime.date.today()
    file_path = os.path.join('./models', f'{str(timestamp)}_unet.h5')
    model.save(file_path)

