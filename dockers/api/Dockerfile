FROM airbus-ship-detection:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-kafka --only api

# Copy the codes into container
COPY /src/inference/api/ .

# Expose the port
EXPOSE 8080

# start the api.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]




