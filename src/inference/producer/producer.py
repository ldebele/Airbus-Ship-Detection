import json
import logging
import argparse
import requests

from kafka import KafkaProducer



TOPIC="ship_image"
BOOTSTRAP_SERVER="localhost:9092"


logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("__PRODUCER_")


def fetch_image_from_api(api_url: str):
    """
    Function to fetch a satellite image from API

        Args:
            api_url (str): URL of the API that provides the image data. 
    """

    response = requests.get(api_url, stream=True)
    if response.status_code == 200:
        return response.content
    else:
        logger.error(f"Failed to fetch image. Status code: {response.status_code}")


def send_image_to_kafka(producer, topic, api_url) -> None:
    """
    Continuously fetches image data from the given API URL and publishes it to the specified Kafka topic
    
        Args:
            producer (kafka.KafkaProducer): Kakfa producer instance used to send messages.
            topic (str): Name of the Kafka topic to publish messages to.
            api_url (str): URL of the API that provides the image data.
    """

    while True:
        img_data = fetch_image_from_api(api_url)

        if img_data:
            # publish image data to kafka topic
            producer.send(topic, {'image': img_data})
            logger.info("Image published to kafka topic.")



def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, required=True, help="API to the satellite Image.")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Kafka Producer streaming...")

    # parse the cli arguments
    args = opt_parser()

    # Initialize the kafka producer
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # send message to kafka topic
    send_image_to_kafka(producer=producer, topic=TOPIC, api_url=args.api_url)



