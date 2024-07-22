FROM ship-detection-base:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry add fastapi

# Copy the codes into container
COPY /src/inference/api/ .

# Expose the port
EXPOSE 8585

# start the api.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8585"]




