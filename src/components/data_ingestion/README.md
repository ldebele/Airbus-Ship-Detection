

Go to your kaggle account settings and select "Create New API Token". This will download a `kaggle.json` file containing your API credentials.


```
docker run \
    -v ~/.kaggle:/root/.kaggle 
    data-ingestion airbus-ship-detection
```