FROM python:3.11.1-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app 

# Install Poetry
RUN pip install poetry

# Copy poetry related files.
COPY poetry.lock pyproject.toml ./

# Install dependencies
RUN poetry install --no-root --with pubsub

# Copy producers file into container
COPY /src/inference/producer/ .
COPY /scripts/wait-for-it.sh .

# make the script executable
RUN chmod +x wait-for-it.sh
