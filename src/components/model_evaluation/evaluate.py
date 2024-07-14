import logging
import argparse
from typing import Tuple

import mlflow

import tensorflow as tf


EXPERIMENT_NAME = "Airbus-Ship-Detection"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__MODEL_EVALUATION__")

mlflow.set_experiment(EXPERIMENT_NAME)



def load_model(model_dir: str):
    """
    Load a pre-trained model.

    Args: 
        model_dir (str): The path to the saved model directory.

    Returns:
        model (tf.keras.Model): The pre-trained keras model.

    Raises:
        ValueError: If the model cannot be loaded from the directory.
    
    """

    try:
        model = tf.keras.models.load_model(model_dir)
    except OSError as e:
        logger.error("Error loading the model.")
        raise ValueError(f"Error loading model from {model_dir}: {e}")
        
    return model


def preprocess(test_dir: str, target_size: Tuple[int, int], batch: int):
    """
    Preprocesses test images for model evaluation.

    Args:
        test_dir (str): The path to the test images directory.
        target_size (Tuple(int, int)): The desired output image size.
        batch (int): The batch size for the test image generator.
    
    Returns:
        test_generator (tf.data.Dataset): A batch generator preprocessed test images. 
    """

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=target_size,
                                                      batch_size=batch,
                                                      shuffle=False,
                                                      class_mode='binary'
                                                      )
    
    return test_generator


def evaluate(test_dir: str, model_dir: str, target_size: Tuple[int, int], batch: int) -> None:
    """
    Evaluates a loaded model on a preprocessed datasets.

    Args:
        test_dir (str): The path to the test images directory.
        model_dir (str): The path to the saved model directory.
    """

    model = load_model(model_dir)
    logger.info("Loads a pre-trained model successfully.")

    test_generator = preprocess(test_dir, target_size, batch)
    logger.info("Completed preprocessing.")

    score = model.evaluate(test_generator)
    logger.info("Evaluate the model successfully.")

    

def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=str, required=True, help="Path to the test directory.")
    parser.add_argument('--model-dir', type=str, required=True, help="Path to the model directory.")
    parser.add_argument('--target-size', type=str, default=(768, 768, 3), help="The desired output image size.")
    parser.add_argument('--batch', type=int, default=8, help="Number of batch size.")
    
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting model evaluation....")
    args = opt_parser()

    evaluate(
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        target_size=args.target_size,
        batch=args.batch

    )

