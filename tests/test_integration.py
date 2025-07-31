"""
Integration tests for the cancer prediction pipeline.

These tests verify that the entire pipeline works correctly from data loading
to model training and prediction.
"""

import unittest
import pandas as pd
import tempfile
import os
from src.preprocess import load_and_preprocess
from src.model import train_and_save_model
from sklearn.model_selection import train_test_split


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'age': [60, 70, 65, 55, 75],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'country': ['US', 'IN', 'US', 'UK', 'CA'],
            'diagnosis_date': ['2020-01-01', '2021-01-01', '2020-06-01', '2021-03-01', '2020-12-01'],
            'cancer_stage': [2, 3, 1, 2, 4],
            'family_history': ['yes', 'no', 'yes', 'no', 'yes'],
            'smoking_status': ['never', 'former', 'current', 'never', 'former'],
            'bmi': [25.0, 30.0, 22.0, 28.0, 35.0],
            'cholesterol_level': [180, 200, 160, 220, 250],
            'hypertension': ['no', 'yes', 'no', 'yes', 'yes'],
            'asthma': ['no', 'no', 'yes', 'no', 'no'],
            'cirrhosis': ['no', 'no', 'no', 'yes', 'no'],
            'other_cancer': ['no', 'yes', 'no', 'no', 'yes'],
            'treatment_type': ['chemo', 'radio', 'surgery', 'chemo', 'radio'],
            'end_treatment_date': ['2020-06-01', '2021-06-01', '2020-09-01', '2021-09-01', '2021-03-01'],
            'survived': [1, 0, 1, 0, 0]
        })
    
    def test_full_pipeline(self):
        """Test the complete pipeline from data preprocessing to model training."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test preprocessing
            processed_df = load_and_preprocess(temp_file)
            self.assertIsInstance(processed_df, pd.DataFrame)
            self.assertGreater(len(processed_df), 0)
            
            # Test model training
            y = processed_df['survived']
            X = processed_df.drop('survived', axis=1)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = train_and_save_model(X_train, y_train, 'test_model.joblib')
            self.assertIsNotNone(model)
            
            # Test prediction
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(y_test))
            
            # Test probability prediction
            probabilities = model.predict_proba(X_test)
            self.assertEqual(probabilities.shape[1], 2)  # Two classes
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists('test_model.joblib'):
                os.unlink('test_model.joblib')
    
    def test_data_consistency(self):
        """Test that preprocessing maintains data consistency."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            processed_df = load_and_preprocess(temp_file)
            
            # Check that survived column exists
            self.assertIn('survived', processed_df.columns)
            
            # Check that survived values are binary
            unique_values = processed_df['survived'].unique()
            self.assertTrue(all(val in [0, 1] for val in unique_values))
            
            # Check that no NaN values in survived column
            self.assertFalse(processed_df['survived'].isna().any())
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main() 