from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator 



with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2024, 7, 13),
    schedule_interval=None,
) as dag:
    
    # Task 1: Data Ingestion
    data_ingestion = DockerOperator(
        image="",
        command="./download_dataset.sh airbus-ship-detection competitions",
        task_id="data_ingestion",
        volume_mounts=[
            "~/.kaggle:/root/.kaggle",
            "${pwd}/data:/app/data"
        ]
    )

    # Task 2: Preprocessing
    preprocessing = DockerOperator(
        image="",
        command="python preprocess.py --images-dir='' --masks-dir='', --img-shape=() --batch=8",
        task_id="preprocessing",
        volume_mounts=[
            "/data:/app/data",
            "/data/processed:/app/data/processed"
        ]
    )

    # Task 3: Model Training
    model_training = DockerOperator(
        image="",
        command="python3 train.py --train-dir='' --val-dir='' --learning-rate=0",
        task_id="model_training",
        volume_mounts=[
            "/data/processed:/app/data/processed"
        ]
    )

    # Task 4: Model Evaluation
    model_evaluation = DockerOperator(
        image="",
        command="",
        task_id="model_evaluation",
        volume_mounts=[
            "/data/test:/app/data/test"
        ]
    )


    # set task dependencies
    data_ingestion >> preprocessing >> model_training >> model_evaluation
