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
            "${pwd}/data:/mnt/data"
        ]
    )

    # Task 2: Data Preprocessing
    preprocessing = DockerOperator(
        image="",
        command="python3 preprocess.py --images-dir='/mnt/data/images' --masks-dir='/mnt/data/mask', --img-shape=(768, 768, 3)",
        task_id="preprocessing",
        volume_mounts=[
            "/data:/mnt/data"
        ]
    )

    # Task 3: Model Training
    model_training = DockerOperator(
        image="",
        command="python3 train.py --train-dir='/mnt/data/train.tfrecord' --val-dir='/mnt/data/val.tfrecord' --num-classes=1, --img-shape=(768, 768, 3) --batch=8, --epochs=50 --learning-rate=0.001",
        task_id="model_training",
        volume_mounts=[
            "/data/:/mnt/data/"
        ]
    )

    # Task 4: Model Evaluation
    model_evaluation = DockerOperator(
        image="",
        command="python3 evaluation.py --test-dir='/mnt/data/' --model-dir='/mnt/data/models' --target-size=(768. 768) --batch=8",
        task_id="model_evaluation",
        volume_mounts=[
            "/data/test:/mnt/data/test"
        ]
    )

    # set task dependencies
    data_ingestion >> preprocessing >> model_training >> model_evaluation
