FROM python:3.11.1-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app 

# Copy poetry related files.
COPY poetry.lock pyproject.toml /app/

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry install --no-root --with pub-sub

# Copy consumers file into container
COPY /src/inference/consumer/ .

# Run consumer
CMD ["python", "consumer.py"]

