

Go to your kaggle account settings and select "Create New API Token". This will download a `kaggle.json` a file that contains your API credentials. Make a directoyr `.kaggle` at root `~` and place `kaggle.json` in that directory.
```
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle
```


```
docker run \
    -v ~/.kaggle:/root/.kaggle \
    airbus-ship-detection-data-ingestion \
    ./download_dataset.sh airbus-ship-detection competitions
```