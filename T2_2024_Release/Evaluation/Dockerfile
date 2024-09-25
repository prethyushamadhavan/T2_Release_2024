# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/classify              /app/data/sentiment \ 
             /app/data/final_results

# Run the script when the container launches
CMD ["python", "BERT_ANA.py"]