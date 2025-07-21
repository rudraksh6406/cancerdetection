import streamlit as st
import pandas as pd
import joblib
from src.preprocess import load_and_preprocess

# Load the trained model
model = joblib.load('model.joblib')

st.title('Cancer Patient Survival Prediction')
st.write('Enter patient details to predict survival:')

# Collect user input
age = st.number_input('Age', min_value=0, max_value=120, value=50)
gender = st.selectbox('Gender', ['male', 'female'])
country = st.text_input('Country', 'USA')
cancer_stage = st.selectbox('Cancer Stage', ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])
family_history = st.selectbox('Family History of Cancer', ['yes', 'no'])
smoking_status = st.selectbox('Smoking Status', ['current smoker', 'former smoker', 'never smoked', 'passive smoker'])
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
cholesterol_level = st.number_input('Cholesterol Level', min_value=50.0, max_value=400.0, value=180.0)
hypertension = st.selectbox('Hypertension', ['yes', 'no'])
asthma = st.selectbox('Asthma', ['yes', 'no'])
cirrhosis = st.selectbox('Cirrhosis', ['yes', 'no'])
other_cancer = st.selectbox('Other Cancer', ['yes', 'no'])
treatment_type = st.selectbox('Treatment Type', ['surgery', 'chemotherapy', 'radiation', 'combined'])
treatment_duration = st.number_input('Treatment Duration (days)', min_value=0, max_value=5000, value=180)

# Prepare input as a DataFrame
input_dict = {
    'age': age,
    'gender': gender,
    'country': country,
    'cancer_stage': cancer_stage,
    'family_history': family_history,
    'smoking_status': smoking_status,
    'bmi': bmi,
    'cholesterol_level': cholesterol_level,
    'hypertension': hypertension,
    'asthma': asthma,
    'cirrhosis': cirrhosis,
    'other_cancer': other_cancer,
    'treatment_type': treatment_type,
    'treatment_duration': treatment_duration
}
input_df = pd.DataFrame([input_dict])

# Apply the same preprocessing as training (binary and categorical encoding)
binary_cols = ['family_history', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer']
for col in binary_cols:
    if col in input_df:
        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})

categorical_cols = ['gender', 'country', 'cancer_stage', 'smoking_status', 'treatment_type']
# Get dummies, align columns with training data
train_df = load_and_preprocess('data/dataset_med.csv')
train_df = train_df.fillna(train_df.median(numeric_only=True)).fillna(0)
X_train = train_df.drop('survived', axis=1)
input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

# Predict
if st.button('Predict Survival'):
    pred = model.predict(input_df)[0]
    st.write('Prediction:', 'Survived' if pred == 1 else 'Did not survive') 