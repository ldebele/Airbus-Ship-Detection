import os

from datetime import datetime
from dotenv import load_dotenv

from docker.types import Mount

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator 


load_dotenv()


ROOT_DIR = os.getenv("ROOT_DIR", os.getcwd())


with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2024, 7, 13),
    schedule_interval=None,
) as dag:
    
    # Task 1: Data Ingestion
    data_ingestion = DockerOperator(
        task_id="data_ingestion",
        image='ship-detection-data-ingestion:latest',
        command="./download_dataset.sh airbus-ship-detection competitions",
        mounts=[
            Mount(source=f"{ROOT_DIR}/.kaggle", target="/app/.kaggle", type="bind"),
            Mount(source=f"{ROOT_DIR}/data", target="/mnt/data", type="bind")
        ],
        docker_url='tcp://docker-proxy:2375',
    )

    # Task 2: Data Preprocessing
    preprocessing = DockerOperator(
        task_id="preprocessing",
        image="ship-detection-preprocess:latest",
        command="python3 preprocess.py \
                --images-dir='/mnt/data/images' \
                --masks-dir='/mnt/data/mask' \
                --img-shape=(768, 768, 3)",
        mounts=[
            Mount(source=f"{ROOT_DIR}/data", target="/mnt/data", type="bind")
        ],
        docker_url='tcp://docker-proxy:2375',
    )

    # Task 3: Model Training
    model_training = DockerOperator(
        task_id="model_training",
        image="ship-detection-training:latest",
        command="python3 train.py \
                --train-dir='/mnt/data/train.tfrecord' \
                --val-dir='/mnt/data/val.tfrecord' \
                --num-classes=1 \
                --img-shape=(768, 768, 3) \
                --batch=8 \
                --epochs=50 \
                --learning-rate=0.001",
        mounts=[
            Mount(source=f"{ROOT_DIR}/data", target="/mnt/data", type="bind"),
            Mount(source=f"{ROOT_DIR}/outputs", target="/mnt/outputs", type="bind")
        ],
        docker_url='tcp://docker-proxy:2375',
    )

    # Task 4: Model Evaluation
    model_evaluation = DockerOperator(
        task_id="model_evaluation",
        image="ship-detection-evaluation:latest",
        command="python3 evaluation.py \
                --test-dir='/mnt/data/' \
                --model-dir='/mnt/data/models' \
                --target-size=(768. 768) \
                --batch=8",
        mounts=[
            Mount(source=f"{ROOT_DIR}/data/test", target="/mnt/data/test", type="bind")
        ],
        docker_url='tcp://docker-proxy:2375',
    )

    # set task dependencies
    data_ingestion >> preprocessing >> model_training >> model_evaluation
