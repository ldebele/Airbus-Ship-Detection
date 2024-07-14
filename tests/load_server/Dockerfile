FROM locustio/lucust:latest

# Set working directory.
WORKDIR /app

# Install dependencies to the container
RUN pip install pillow

# Copy file into container
COPY . .

# Expose port 8089
EXPOSE 8089

# Run the locust server
CMD ["locust", "-f", "locustfile.py"]
