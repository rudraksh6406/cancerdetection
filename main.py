import pandas as pd
from src.preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model
import numpy as np

# Inspect the raw data for the 'survived' column
raw_df = pd.read_csv('data/dataset_med.csv', low_memory=False)
print('First 10 rows of the raw data (survived column):')
print(raw_df[['survived']].head(10))

# 1. Load and preprocess the data
df = load_and_preprocess('data/dataset_med.csv')

# 2. Handle missing values (imputation)
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)

# Debug: Print unique values in the target column
y_counts = df['survived'].value_counts(dropna=False)
print('Unique values in survived column after preprocessing:', y_counts)

# 3. Separate features and target
y = df['survived']
X = df.drop('survived', axis=1)

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Upsample the minority class in the training set
from sklearn.utils import resample
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data['survived'] == 0]
minority = train_data[train_data['survived'] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
train_upsampled = pd.concat([majority, minority_upsampled])
X_train_up = train_upsampled.drop('survived', axis=1)
y_train_up = train_upsampled['survived']

# 6. Train a Random Forest model with class_weight balanced
model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
model.fit(X_train_up, y_train_up)

# Save the trained model
joblib.dump(model, 'model.joblib')

# 7. Make predictions and evaluate
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# 8. Print prediction distribution on test set
unique, counts = np.unique(y_pred, return_counts=True)
pred_dist = dict(zip(unique, counts))
print('Prediction distribution on test set:', pred_dist)
