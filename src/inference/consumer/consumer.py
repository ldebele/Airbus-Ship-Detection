import json
import logging
import requests

from kafka import KafkaConsumer 



TOPIC="ship_image"
BOOTSTRAP_SERVER="localhost:9092"
API_URL="localhost:8080/predict"

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("__CONSUMER_")



def get_prediction(img_data) -> dict:
    """
    Sends an image to a prediction endpoint and retrieves the prediction results.

    Args:
        img_data (): The image data

    Returns:
        dict: The prediction response as a dictionary.
    """

    payload = {"file": ("image.jpg", img_data)}

    try:
        # send a POST request to prediction endpoint
        response = requests.post(API_URL, files=payload)
        # Raise exception for non-200 status codes.
        response.raise_for_status()

        # Get the prediction response as JSON
        prediction = response.json()
        logger.info(f"Prediction received successfully.")
        return prediction
    
    except requests.exceptions.Requests.RequestException as e:
        logger.error(f"Error occurred while sending the image for prediction: {e}")


def consume_image_from_kafka(consumer):
    """
    Continuously consumers messages from the kafka topic and processes the image.

    Args:
        consumer (kafka.KafkaConsumer): Kafka consumer instance used to receive messages.
    """

    for message in consumer:
        img_data = message.value['image']
        logger.info("Image consumed from kafka.")
        
        # get the prediction.
        get_prediction(img_data)


if __name__ == "__main__":
    logger.info("Kafka Consumer started consuming...")    

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='image-consumer',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )