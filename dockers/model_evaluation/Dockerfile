FROM ship-detection-base:latest

# Set the working directory
WORKDIR /app/components

# Install dependencies
RUN poetry install --without dev inference

# Copy the codes into container
COPY /src/components/model_evaluation .

