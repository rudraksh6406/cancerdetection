#!/usr/bin/env python3
"""
Basic usage example for the Cancer Patient Survival Prediction model.

This script demonstrates how to use the preprocessed data and trained model
to make predictions on new patient data.
"""

import pandas as pd
import logging
from src.preprocess import load_and_preprocess
from src.model import train_and_save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic usage of the cancer prediction model."""
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_and_preprocess('data/dataset_med.csv')
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(0)
    
    # Prepare features and target
    y = df['survived']
    X = df.drop('survived', axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    logger.info("Training model...")
    model = train_and_save_model(X_train, y_train, 'example_model.joblib')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")
    
    # Example prediction on new data
    logger.info("Making prediction on sample data...")
    sample_data = X_test.iloc[:1]  # Take first test sample
    prediction = model.predict(sample_data)
    probability = model.predict_proba(sample_data)
    
    logger.info(f"Sample prediction: {'Survived' if prediction[0] == 1 else 'Not Survived'}")
    logger.info(f"Survival probability: {probability[0][1]:.4f}")


if __name__ == "__main__":
    main() 