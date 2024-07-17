from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator 



with DAG(
    dag_id="inference_pipeline",
    start_date=datetime(2024, 7, 16),
    schedule_interval=None,
) as dag:
    
    # Task 1: Prediction API
    prediction_api = DockerOperator(
        image="",
        command="python3 main.py",
        task_id="prediction_api"
    )

    # Task 2: Image Consumer
    consumer = DockerOperator(
        image="",
        command="python3 consumer.py",
        task_id="image_consumer"
    )

    # Task 3: Image Producer
    producer = DockerOperator(
        image="",
        command="python3 preprocess.py --api-url='https://'",
        task_id="image_producer"
    )



    # set task dependencies
    prediction_api >> consumer
    prediction_api >> producer
