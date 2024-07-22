

.PHONY: all_inference all_training base api producer consumer data-ingestion preprocessing model-training model-evaluation start-training start-inference

all_inference: base api producer consumer

all_training: base data-ingestion preprocessing model-training model-evaluation


start-training: base data-ingestion preprocessing model-training model-evaluation
	docker-compose -f docker-compose.training.yml down && \
	docker-compose -f docker-compose.training.yml up

start-inference:
	docker-compose down && \
	docker-compose up


base:
	docker build -f dockers/Dockerfile.base -t ship-detection-base:latest .

api: base
	docker build -f dockers/api/Dockerfile -t ship-detection-api:latest .

producer:
	docker build -f dockers/producer/Dockerfile -t ship-detection-producer:latest .

consumer:
	docker build -f dockers/consumer/Dockerfile -t ship-detection-consumer:latest .

data-ingestion:
	docker build -f dockers/data_ingestion/Dockerfile -t ship-detection-data-ingestion:latest .

preprocessing: base
	docker build -f dockers/processing/Dockerfile -t ship-detection-preprocessing:latest .

model-training: base
	docker build -f dockers/model_training/Dockerfile -t ship-detection-model-training:latest .

model-evaluation: base
	docker build -f dockers/model_evaluation/Dockerfile -t ship-detection-model-evaluation:latest .


airflow-init:
	docker-compose -f docker-compose.training.yml down && \
	docker-compose -f docker-compose.training.yml up airflow-init

cleanup:
	docker-compose -f --volumes --rmi all && \
	docker-compose -f docker-compose.training.yaml down --volumes --rmi all && \
	docker system prune -f
	
