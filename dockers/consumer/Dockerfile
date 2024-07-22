FROM python:3.11.1-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app 

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry add kafka-python requests

# Copy consumers file into container
COPY /src/inference/consumer/ .
COPY /scripts/wait-for-it.sh .

# make the script executable
RUN chmod +x wait-for-it.sh


