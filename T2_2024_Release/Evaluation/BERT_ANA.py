
# Required Libraries and Their Purposes:
# - pandas: Used for data manipulation and analysis, particularly for reading and writing CSV files.
# - transformers: Provides the BERT model and tokenizer for sequence classification tasks.
# - tensorflow: Used for model training, including defining, compiling, and fitting neural networks.
# - numpy: Supports numerical operations, such as manipulating arrays and tensors.
# - sklearn: Used for splitting data into training and validation sets.

# Import Statements
import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
import sklearn
import os
import glob
import logging
from transformers import BertTokenizer, TFBertModel
from transformers import TFBertForSequenceClassification, BertTokenizer


print("Pandas version:", pd.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)
print("Transformers version:", transformers.__version__)

# Suppress TensorFlow warnings for cleaner output
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Required pip installs for the project (Uncomment to install)

# !pip install pandas==2.2.2
# !pip install tensorflow==2.17.0
# !pip install numpy==1.23.5
# !pip install scikit-learn==1.4.2
# !pip install transformers==4.44.1

# Cell 2: Data Loading and Initial Predictions

# Description:
# Prediction phase of your multi-task BERT model. It processes input data 
# containing interaction texts, performs tokenization, executes predictions 
# for three key elements—Category, Quality, and Sentiment—and saves the 
# results to the /data/predicted/ directory. The sentiment score is specifically 
# constrained between 0.1 and 0.9 to maintain a defined range. 

# Function to scale sentiment output
def scale_sentiment_output(x):
    return 0.1 + 0.8 * x

# Load the entire fine-tuned model and tokenizer
def load_fine_tuned_model():
    try:
        # Load the entire saved model
        model_path = os.path.join('fine_tuned_bert', 'saved_model')
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TFBertModel': TFBertModel,
                'scale_sentiment_output': scale_sentiment_output
            }
        )
        tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert')
        print("Entire model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise

# Load and preprocess the data (only 'Text' column is available)
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'transcript' in data.columns:
          data = data.rename(columns={'transcript': 'Text'})
        print(f"Data loaded successfully from {file_path}. Number of rows: {data.shape[0]}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Tokenize the texts
def tokenize_texts(tokenizer, texts, max_length=128):
    try:
        inputs = tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            return_tensors="tf",
            max_length=max_length
        )
        print("Tokenization completed successfully.")
        return inputs
    except Exception as e:
        print(f"Error during tokenization: {e}")
        raise

# Predict using the loaded model
def run_prediction_pipeline(data, output_file, model, tokenizer):
    # Tokenize the data
    inputs = tokenize_texts(tokenizer, data['Text'])
    
    # Make predictions on the dataset
    print("Making predictions on the dataset...")
    predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])
    
    # Extract predictions
    predicted_categories = np.argmax(predictions['category_output'], axis=1)
    predicted_qualities = np.argmax(predictions['quality_output'], axis=1)
    predicted_sentiments = predictions['sentiment_output'].flatten()
    
    # Convert predicted integer labels back to their respective categories
    inverse_category_mapping = {
        0: 'Greetings',
        1: 'Problem Investigation',
        2: 'Closure',
        3: 'Account Verification'
    }
    inverse_quality_mapping = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
    
    # Add prediction columns
    data['Predicted Category'] = pd.Series(predicted_categories).map(inverse_category_mapping)
    data['Predicted Quality'] = pd.Series(predicted_qualities).map(inverse_quality_mapping)
    data['Predicted Sentiment'] = predicted_sentiments.round(2)  # Rounded for readability
    
    # Save the predictions only
    data = data[['speaker', 'Text', 'Predicted Category', 'Predicted Quality', 'Predicted Sentiment']]
    
    # Save the updated DataFrame with the predictions
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}.")

# Run pipeline for all files in the `/data/original/` folder and save to `/data/predicted/`
def run_pipeline():
    # Load model and tokenizer once
    model, tokenizer = load_fine_tuned_model()
    
    # Get all CSV files in the '/data/original/' directory
    input_folder = 'data/transcriptions/'
    output_folder = 'data/classify/'
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all CSV files in the input folder
    file_paths = glob.glob(os.path.join(input_folder, '*.csv'))
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        data = load_data(file_path)
        output_file = os.path.join(output_folder, os.path.basename(file_path))  # Save with the same name in /predicted/
        run_prediction_pipeline(data, output_file, model, tokenizer)

# Execute the pipeline
if __name__ == "__main__":
    run_pipeline()

# Cell 3: Sub-Metric Predictions

# Description:
# This cell takes the output from Cell 2, which contains customer 
# interaction data with predictions for Category, Quality, and Sentiment, 
# and appends additional predictions for various sub-metrics like 
# "Thank Customer" or "Ask Permission" using a pre-trained BERT model. 
# It processes multiple files from the `/data/predicted/` folder, skips 
# any already predicted files, and saves the updated data with predictions 
# into the `/data/metrics/` folder. The structure of the original data is 
# preserved while adding prediction columns for further analysis.

# Tokenize the input text using BERT tokenizer
def tokenize_texts(tokenizer, texts, max_length=128):
    try:
        inputs = tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="tf"
        )
        print("Tokenization completed successfully.")
        return inputs
    except Exception as e:
        print(f"Error during tokenization: {e}")
        raise

# Function to load an existing model and tokenizer
def load_saved_model_and_tokenizer():
    try:
        # Attempt to load the Keras model
        model = tf.keras.models.load_model(
            'sent_ana_model',
            custom_objects={'TFBertForSequenceClassification': TFBertForSequenceClassification}
        )
        tokenizer = BertTokenizer.from_pretrained('sent_ana_model')
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print("No existing model found. Please ensure the model and tokenizer are available.")
        raise e

# Make predictions and append the sub-metrics to the DataFrame
def make_predictions(model, tokenizer, data, output_file):
    try:
        # Tokenize the input text
        inputs = tokenize_texts(tokenizer, data['Text'])
    
        # Make predictions for each sub-metric
        print("Making predictions on the dataset...")
        predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])
    
        # Assuming the model has multiple output layers for each sub-metric
        # Adjust the number of predictions based on your model's architecture
        # Example assumes 10 sub-metrics
        sub_metrics = [
            'Thank Customer',
            'Introduce Self',
            'Ask Reason',
            'Ask Accurate Details',
            'Ask Permission',
            'Resolve Issue',
            'Offer Assistance',
            'Thank Again',
            'Farewell'
        ]
    
        # Iterate over each sub-metric and add predictions to the DataFrame
        for idx, sub_metric in enumerate(sub_metrics, start=0):
            predicted_values = np.argmax(predictions[idx], axis=1)
            data[f'Predicted {sub_metric}'] = predicted_values
    
        # Save the results to a CSV file
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}.")
    except Exception as e:
        print(f"Error during prediction or saving results: {e}")
        raise

# Run the pipeline for multiple files in the predicted folder
def run_pipeline():
    try:
        # Load model and tokenizer once
        model, tokenizer = load_saved_model_and_tokenizer()
    
        # Define input and output directories
        input_folder = 'data/classify/'
        output_folder = 'data/sentiment/'
        os.makedirs(output_folder, exist_ok=True)
    
        # Find all CSV files in the input folder
        file_paths = glob.glob(os.path.join(input_folder, '*.csv'))
    
        for file_path in file_paths:
            # Skip files that contain "predictions" in their filename
            if "predictions" in file_path.lower():
                print(f"Skipping file: {file_path}")
                continue
    
            print(f"Processing file: {file_path}")
            data = load_data(file_path)
            output_file = os.path.join(output_folder, os.path.basename(file_path))  # Save with the same name in /metrics/
            make_predictions(model, tokenizer, data, output_file)
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        raise

# Execute the pipeline
if __name__ == "__main__":
    run_pipeline()

# Cell 4 – Evaluations

# Description:
    
# processes all CSV files in the data/metrics directory to comprehensively 
# evaluate customer interactions. It calculates key metrics such as category 
# diversity, average quality using a weighted scoring system, and average 
# predicted sentiment. Additionally, it assesses specific interaction aspects 
# through predefined sub-metrics, awarding points based on correct predictions. 
# After validating and standardizing the data, the aggregated results for each 
# file are compiled and saved into a single CSV file (combined_evaluation.csv) 
# in the data/evaluation directory for easy comparison and analysis of overall 
# performance.    

# Function to convert weighted average to a category
def convert_to_quality_category(avg_score):
    if avg_score >= 0.7:
        return 'Positive'
    elif 0.4 <= avg_score < 0.7:
        return 'Neutral'
    else:
        return 'Negative'

def calculate_scores(file_path):
    
    # Load the data
    data = pd.read_csv(file_path)
    
    # Define sub-metric points based on the rubric
    sub_metric_points = {
        'Thank Customer': 5,
        'Introduce Self': 5,
        'Ask Reason': 5,
        'Ask Accurate Details': 10,
        'Ask Permission': 10,
        'Resolve Issue': 50,
        'Offer Assistance': 5,
        'Thank Again': 5,
        'Farewell': 5
    }

    # Calculate the percentage of predicted categories
    total_categories = 4  # Total number of categories: Greetings, Account Verification, Problem Investigation, Closure
    unique_categories = data['Predicted Category'].nunique()
    category_percent = (unique_categories / total_categories) * 100

    # Assign weights to quality labels
    quality_weights = {'Positive': 0.8, 'Neutral': 0.5, 'Negative': 0.2}

    # Replace predicted quality with their corresponding weights
    weighted_quality = data['Predicted Quality'].map(quality_weights)

    # Calculate the average of the weighted quality
    average_weighted_quality = weighted_quality.mean()

    # Convert the weighted average into a category (Positive, Neutral, or Negative)
    avg_quality_category = convert_to_quality_category(average_weighted_quality)

    # Calculate Average Predicted Sentiment
    average_predicted_sentiment = round(data['Predicted Sentiment'].mean(), 2)

    # Initialize dictionary to store earned points for each sub-metric
    earned_points = {}

    # Iterate over each sub-metric to assign points
    for sub_metric, points in sub_metric_points.items():
        predicted_col = f'Predicted {sub_metric}'
        
        # Filter rows where the predicted sub-metric is 1
        relevant_rows = data[data[predicted_col] == 1]
        
        # Check if any of these rows have correctly predicted the sub-metric
        earned_points[sub_metric] = points if not relevant_rows.empty else 0
    
    # Calculate Overall Score by summing earned points
    overall_score = sum(earned_points.values())
    overall_score = round(overall_score, 2)

    # Prepare the results dictionary
    results = {
        'file': os.path.basename(file_path),
        'category %': round(category_percent, 2),  # The percentage of unique categories
        'avg quality': avg_quality_category,  # This is now the weighted category (Positive/Neutral/Negative)
        'average_predicted_sentiment': average_predicted_sentiment
    }
    
    # Add sub-metric points to the results
    results.update(earned_points)

    # Add Overall Score
    results['Overall Score'] = overall_score
    
    return results

def process_multiple_files(input_directory, output_file):
    
    # Find all CSV files in the input directory
    file_paths = glob.glob(os.path.join(input_directory, '*.csv'))
    
    # Initialize an empty list to store all evaluation results
    all_results = []

    for file in file_paths:
        print(f"Processing file: {file}")
        # Calculate scores and append the results to the list
        all_results.append(calculate_scores(file))
    
    # Convert the list of results into a DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save the results to the specified output CSV file
    results_df.to_csv(output_file, index=False)
    
    print(f"All results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Directory containing evaluation files
    input_dir = 'data/sentiment'
    
    # Path to save combined evaluation results
    output_file = 'data/final_results/combined_evaluation.csv'
    
    # Process all evaluation files in the directory and save to a single file
    process_multiple_files(input_dir, output_file)
