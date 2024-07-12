#!/bin/bash

# check if the user provided a dataset
if [ -z "$1" ]; then 
    echo "Dataset path: Airbus-ship-detection"
    exit 1
fi

DATASET_NAME=$1

# check if kaggle.json file exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Please mount your kaggle.json file to the Dockerfile"
    exit 1
fi

# download the dataset
echo "Downloading dataset: $DATASET_NAME"
kaggle competitions c download $DATASET_NAME -p ./ --unzip

if [ $? -eq 0 ]; then
    echo "Dataset downloaded successfully."
else
    echo "Failed to download dataset."
    exit 1
fi
