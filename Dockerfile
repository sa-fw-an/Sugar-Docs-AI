# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
# Replace the below URL with the actual Ollama installation script or method
# Ensure you follow Ollama's official installation guide
# Example placeholder:
# RUN curl -L https://ollama.com/install.sh | bash

# Create and set the working directory
WORKDIR /app

COPY requirements_for_local.txt /app/
RUN pip install --no-cache-dir -r requirements_for_local.txt

# Copy the rest of the application code
COPY . /app

# Expose the necessary ports
EXPOSE 5000 8501

# Start the Streamlit application
CMD ["streamlit", "run", "containedlocal.py", "--server.port=8501", "--server.enableCORS=false", "--server.address=0.0.0.0"]