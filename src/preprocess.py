import pandas as pd

def load_and_preprocess(path):
    # 1. Load the data
    df = pd.read_csv(path, low_memory=False)

    # 2. Convert date columns to datetime
    for col in ['diagnosis_date', 'end_treatment_date']:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. Feature engineering: Calculate treatment duration (in days)
    if {'diagnosis_date', 'end_treatment_date'}.issubset(df.columns):
        df['treatment_duration'] = (
            df['end_treatment_date'] - df['diagnosis_date']
        ).dt.days

    # 4. Encode binary columns (yes/no to 1/0)
    binary_cols = ['family_history', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer']
    for col in binary_cols:
        if col in df:
            df[col] = df[col].map({'yes': 1, 'no': 0})

    # 5. Drop columns that are not useful for prediction
    to_drop = [c for c in ['id', 'diagnosis_date', 'end_treatment_date'] if c in df.columns]
    df.drop(columns=to_drop, inplace=True)

    # 6. Identify categorical columns for encoding
    categorical_cols = [c for c in ['gender', 'country', 'cancer_stage', 'smoking_status', 'treatment_type'] if c in df.columns]

    # 7. One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 8. Return the processed DataFrame
    return df