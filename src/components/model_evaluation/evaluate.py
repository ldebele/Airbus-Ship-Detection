import logging
import argparse

import mlflow


EXPERIMENT_NAME = "Airbus-Ship-Detection"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__MODEL_EVALUATION__")

mlflow.set_experiment(EXPERIMENT_NAME)


def load_model():
    model = None

    return model



def evaluate(model, test_dataset):

    score = model.evaluate(test_dataset)








def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dataset', type=str, required=True, help="path to the test directory")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting model evaluation....")
    args = opt_parser()

