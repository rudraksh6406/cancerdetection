import pandas as pd
from src.preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

raw_df = pd.read_csv('data/dataset_med.csv', low_memory=False)
print(raw_df[['survived']].head(10))

df = load_and_preprocess('data/dataset_med.csv')
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)
y_counts = df['survived'].value_counts(dropna=False)
print(y_counts)

y = df['survived']
X = df.drop('survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.utils import resample
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data['survived'] == 0]
minority = train_data[train_data['survived'] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
train_upsampled = pd.concat([majority, minority_upsampled])
X_train_up = train_upsampled.drop('survived', axis=1)
y_train_up = train_upsampled['survived']

model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
model.fit(X_train_up, y_train_up)

joblib.dump(model, 'model.joblib')

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

unique, counts = np.unique(y_pred, return_counts=True)
pred_dist = dict(zip(unique, counts))
print(pred_dist)
