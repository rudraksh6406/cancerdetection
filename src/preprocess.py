import pandas as pd
import logging
from typing import Any

def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Load and preprocess the cancer patient dataset.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for modeling.
    """
    logging.info(f"Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in ['diagnosis_date', 'end_treatment_date']:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if {'diagnosis_date', 'end_treatment_date'}.issubset(df.columns):
        df['treatment_duration'] = (
            df['end_treatment_date'] - df['diagnosis_date']
        ).dt.days
    binary_cols = ['family_history', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer']
    for col in binary_cols:
        if col in df:
            df[col] = df[col].map({'yes': 1, 'no': 0})
    to_drop = [c for c in ['id', 'diagnosis_date', 'end_treatment_date'] if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    categorical_cols = [c for c in ['gender', 'country', 'cancer_stage', 'smoking_status', 'treatment_type'] if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logging.info("Preprocessing complete.")
    return df