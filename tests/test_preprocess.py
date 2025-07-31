import unittest
from src import preprocess
import pandas as pd

class TestPreprocess(unittest.TestCase):
    def test_preprocess_sample(self):
        # Create a sample dataframe
        df = pd.DataFrame({
            'age': [60, 70],
            'gender': ['M', 'F'],
            'country': ['US', 'IN'],
            'diagnosis_date': ['2020-01-01', '2021-01-01'],
            'cancer_stage': [2, 3],
            'family_history': [1, 0],
            'smoking_status': ['never', 'former'],
            'bmi': [25.0, 30.0],
            'cholesterol_level': [180, 200],
            'hypertension': [0, 1],
            'asthma': [0, 0],
            'cirrhosis': [0, 0],
            'other_cancer': [0, 1],
            'treatment_type': ['chemo', 'radio'],
            'end_treatment_date': ['2020-06-01', '2021-06-01'],
            'survived': [1, 0]
        })
        processed = preprocess.preprocess_data(df)
        self.assertIsNotNone(processed)

if __name__ == '__main__':
    unittest.main() 