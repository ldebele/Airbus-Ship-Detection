FROM ship-detection-base:latest

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y git

# Copy the codes into container
COPY /src/components/model_training .
