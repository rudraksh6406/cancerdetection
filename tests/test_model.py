import unittest
import pandas as pd
from src.model import train_and_save_model

class TestModel(unittest.TestCase):
    def test_train_and_save_model(self):
        # Create a small sample dataset
        X = pd.DataFrame({
            'age': [60, 70],
            'bmi': [25.0, 30.0],
            'cholesterol_level': [180, 200]
        })
        y = pd.Series([1, 0])
        model = train_and_save_model(X, y, model_path='test_model.joblib')
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main() 