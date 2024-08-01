
# Airbus-Ship-Detection


![GitHub contributors](https://img.shields.io/github/contributors/ldebele/Airbus-Ship-Detection)
![GitHub forks](https://img.shields.io/github/forks/ldebele/Airbus-Ship-Detection?style=social)
![GitHub stars](https://img.shields.io/github/stars/ldebele/Airbus-Ship-Detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/ldebele/Airbus-Ship-Detection)
![GitHub license](https://img.shields.io/github/license/ldebele/Airbus-Ship-Detection)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/lemi-debela?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BuAr7JLGOTc%2Br4epMeWrVMw%3D%3D)


<!-- Table of Contents -->
## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Licence](#license)
- [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## Project Overview
The aims of this project is to detect ships from satellite images using an event-driven architecture. The project implements an end-to-end U-Net based deep learning model for detecting ships. The model predicts segmentations masks indicating the ships within the images.

<!-- Architecture -->
## Architecture
The project consists of two main architectures, each containing specific pipelines for different purposes:

- #### Training Pipeline Architecture
For training the model and consists of four components. The components are orchestrated using Airflow:
1. Data Ingestion: Download the datasets from kaggle.
2. Preprocessing: Apply preprocessing techiniques to the datasets.
3. Model Training: Builds and trains the ship detection model.
4. Model Evaluation: Evaluates the performance of the trained model.
<p align="center">
  <img src="./assets/training-pipeline.png" alt="Training Workflow">
</p>

- #### Inference Pipeline 
This architecture is based on an event-driven approach for making predictions using the trained model.
1. Producer: Publishes new satellite images from a Satellite API.
2. Consumer: Consumes the published images and sends a Post request to the prediction API endpoint.
3. API: Handles the prediction requests and returns the results.
<p align="center">
  <img src="./assets/event-driven-architecture.png" alt="Inference Architecture">
</p>


<!-- GETTING STARTED -->
## Getting Started
1. Clone the Repository
    ``` bash
    git clone https://github.com/ldebele/Airbus-Ship-Detection.git
    cd Airbus-Ship-Detection 
    ```

2. Install Docker and Docker Compose

    Follow the instructions on the [Docker website](https://docs.docker.com/engine/install/) to install Docker and Docker Compose.

3. Build Docker images
- To build all inference-related images:
    ```sh
    make all_inference
    ```
- To build all training-related images:
    ```
    make all_training
    ```
4. Start the pipelines.

- To start the inference pipeline.
    ``` bash
    make start-inference
    ```

- To start the training pipeline.
    - Initialize the database
    ``` bash
    make airflow-init
    ```
    - Running the airflow
    ```bash
    make start-training
    ```

5. Accessing the web interfaces.

- Airflow Web Interface.

    Once the cluster has started up, you can log into the web interface and begin experimenting the pipelines.

    Access the Airflow web interface at [http://localhost:8080](http://localhost:8080) using the defult credentials: Username: `airflow` and Password: `airflow`

- MLflow Web Interface

    Access the MLflow experiment tracker at [http://localhost:5000](http://localhost:5000)

- API Web Server

    Access the prediction inference API web server at [http://localhost:8585](http://localhost:8585)


6. Stop and delete containers
    ```bash
    make cleanup
    ```

<!-- LICENSE -->
## License
This project is licensed under the MIT License. See [LICENSE](./LICENCE) file for more details.

<!-- CONTACT -->
## Contact
Lemi Debela - lemidebele@gmail.com

Project Link: [https://github.com/ldebele/Airbus-Ship-Detection](https://github.com/ldebele/Airbus-Ship-Detection)