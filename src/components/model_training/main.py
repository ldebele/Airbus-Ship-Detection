import sys
import argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__MODEL_TRAINING__")



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
