import logging
from typing import Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_and_save_model(X: pd.DataFrame, y: pd.Series, model_path: str = 'model.joblib') -> RandomForestClassifier:
    """
    Train a Random Forest model and save it to disk.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        model_path (str): Path to save the trained model.

    Returns:
        RandomForestClassifier: The trained model.
    """
    logging.info("Training Random Forest model...")
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    return model
