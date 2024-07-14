FROM airbus-ship-detection:latest

# Set the working directory
WORKDIR /app/inference

# Install dependencies
RUN poetry install --without dev --only inference

# Copy the codes into container
COPY /src/inference/ .

# Expose the port
EXPOSE 8080

# start the api.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]




