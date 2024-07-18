FROM python:3.11.1-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app 

# Copy poetry related files.
COPY poetry.lock pyproject.toml /app/

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --only api

# Copy consumers file into container
COPY /src/inference/consumer/ .

# Run consumer
CMD ["python", "consumer.py"]

