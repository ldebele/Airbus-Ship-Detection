#!/bin/bash

DATASET_NAME=$1
TYPE=$2

# check if the user provided a dataset
if [ -z "$1" ]; then 
    echo "Usage: $0 <kaggle-dataset-path> <type>"
    echo "Example: $0 airbus-ship-detection competitions"
    exit 1
fi

# check if kaggle.json file exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Please mount your kaggle.json file to the docker container"
    exit 1
fi


download_dataset() {
    if [ "$TYPE" == "datasets" ]; then
        kaggle datasets download -d $DATASET_NAME -p ./data --unzip
    elif [ "$TYPE" == "competitions" ]; then
        kaggle competitions download -c $DATASET_NAME -p ./data --unzip
    else
        echo "Invalid type. please specify 'datasets' or 'competitions'."
        exit 1
    fi
}

# download the dataset
echo "Downloading dataset: $DATASET_NAME"
download_dataset


if [ $? -eq 0 ]; then
    echo "Dataset downloaded successfully."
else
    echo "Failed to download dataset."
    exit 1
fi
