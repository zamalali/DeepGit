# Use a slim Python 3.10 image as the base
FROM python:3.10-slim

# Install system dependencies (if needed for building some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Expose the default port for Gradio (if you want to access the app externally)
EXPOSE 7860

# Set the command to run your app
CMD ["python", "app.py"]
