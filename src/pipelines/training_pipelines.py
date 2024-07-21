import os

from datetime import datetime
from docker.types import Mount

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator 



PWD = os.getcwd()


with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2024, 7, 13),
    schedule_interval=None,
) as dag:
    
    # Task 1: Data Ingestion
    data_ingestion = DockerOperator(
        task_id="data_ingestion",
        image='airbus-test:latest',
        # command="./download_dataset.sh airbus-ship-detection competitions",
        command="python test.py",
        mounts=[
            Mount(source=f"{PWD}/.kaggle", target="/mnt/data", type="bind"),
            Mount(source=f"{PWD}/data", target="/mnt/data", type="bind")
        ]
    )

    # Task 2: Data Preprocessing
    preprocessing = DockerOperator(
        task_id="preprocessing",
        image="airbus-test:latest",
        # command="python3 preprocess.py --images-dir='/mnt/data/images' --masks-dir='/mnt/data/mask', --img-shape=(768, 768, 3)",
        command="python test.py",
        mounts=[
            Mount(source=f"{PWD}/data", target="/mnt/data", type="bind")
        ]
    )

    # Task 3: Model Training
    model_training = DockerOperator(
        task_id="model_training",
        image="airbus-test:latest",
        # command="python3 train.py --train-dir='/mnt/data/train.tfrecord' --val-dir='/mnt/data/val.tfrecord' --num-classes=1, --img-shape=(768, 768, 3) --batch=8, --epochs=50 --learning-rate=0.001",
        command="python test.py",
        mounts=[
            Mount(source=f"{PWD}/data", target="/mnt/data", type="bind"),
            Mount(source=f"{PWD}/outputs", target="/mnt/data/outputs", type="bind")
        ]
    )

    # Task 4: Model Evaluation
    model_evaluation = DockerOperator(
        task_id="model_evaluation",
        image="airbus-test:latest",
        # command="python3 evaluation.py --test-dir='/mnt/data/' --model-dir='/mnt/data/models' --target-size=(768. 768) --batch=8",
        command="python test.py",
        mounts=[
            Mount(source=f"{PWD}/data/test", target="/mnt/data/test", type="bind")
        ]
    )

    # set task dependencies
    data_ingestion >> preprocessing >> model_training >> model_evaluation
