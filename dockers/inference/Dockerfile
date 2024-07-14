FROM airbus-ship-detection:latest

# Set the working directory
WORKDIR /app/inference

# Install dependencies
RUN poetry install --without dev --only inference

# Copy the codes into container
COPY /src/inference/ .







