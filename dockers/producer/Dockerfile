FROM python:3.11.1-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app 

# Install dependencies
RUN pip install kafka-python requests

# Copy producers file into container
COPY /src/inference/producer/ .
COPY /scripts/wait-for-it.sh .

# make the script executable
RUN chmod +x wait-for-it.sh
