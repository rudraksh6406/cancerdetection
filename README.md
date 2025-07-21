# Cancer Patient Survival Prediction

This project predicts the survival of lung cancer patients using a machine learning model.

## Dataset
- Place your dataset as `data/dataset_med.csv`.
- The dataset should include columns like age, gender, country, diagnosis_date, cancer_stage, family_history, smoking_status, bmi, cholesterol_level, hypertension, asthma, cirrhosis, other_cancer, treatment_type, end_treatment_date, and survived.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

## What it does
- Loads and preprocesses the data
- Trains a Random Forest model to predict survival
- Prints accuracy and a classification report

## Files
- `src/preprocess.py`: Data preprocessing
- `main.py`: Main script to run the pipeline
- `requirement.txt`: Python dependencies
- `README.md`: Project instructions
