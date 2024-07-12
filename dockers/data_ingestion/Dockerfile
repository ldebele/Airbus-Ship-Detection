FROM python:3.11.1-slim

# set working directory
WORKDIR /app

# install kaggle CLI
RUN pip install kaggle

# copy the script into container
COPY /src/components/data_ingestion/download_dataset.sh .

# make the script executable
RUN chmod +x download_dataset.sh

# run the script
# CMD ["./download_dataset.sh"]