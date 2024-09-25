FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     python3-dev     libc-dev     libsndfile1     ffmpeg     build-essential     curl     && rm -rf /var/lib/apt/lists/*

# Copy all files to the working directory
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Create directories for input and output
RUN mkdir -p /app/transcriptions

# Set environment variable for Hugging Face token
# Note: You should set this when running the container
ENV HF_AUTH_TOKEN=""

# Run the Python script
CMD ["python", "vr.py"]