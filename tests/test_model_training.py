"""
Unit test for Issue #10 (ML-F-010 Model Training, TC-TRAIN-01).
Tests model training execution, input validation, execution timing,
and safe handling of training failures.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from src.services.model_service import ModelService, ModelTrainingError


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_classification = np.random.randn(50, 4)
        self.y_classification = np.array([0] * 25 + [1] * 25)

        self.X_regression = np.random.randn(50, 4)
        self.y_regression = np.random.randn(50)

    def test_train_classification_model_success(self):
        """TC-TRAIN-01 Step 1: Classification model trains successfully on configured data."""
        result = ModelService.train_model(
            model_name="Logistic Regression",
            X_train=self.X_classification,
            y_train=self.y_classification,
            params={"C": 1.0},
        )
        self.assertIn("model", result)
        self.assertIsInstance(result["model"], LogisticRegression)
        self.assertTrue(hasattr(result["model"], "classes_"))
        self.assertGreater(result["training_time_seconds"], 0)
        self.assertEqual(result["sample_count"], 50)
        self.assertEqual(result["feature_count"], 4)

    def test_train_regression_model_success(self):
        """TC-TRAIN-01 Step 2: Regression model trains successfully on configured data."""
        result = ModelService.train_model(
            model_name="Linear Regression",
            X_train=self.X_regression,
            y_train=self.y_regression,
        )
        self.assertIn("model", result)
        self.assertIsInstance(result["model"], LinearRegression)
        self.assertTrue(hasattr(result["model"], "coef_"))
        self.assertGreaterEqual(result["training_time_seconds"], 0)
        self.assertEqual(result["sample_count"], 50)

    def test_train_with_custom_hyperparameters(self):
        """TC-TRAIN-01 Step 3: Configured parameters are correctly passed into the trained model."""
        result = ModelService.train_model(
            model_name="Random Forest Classifier",
            X_train=self.X_classification,
            y_train=self.y_classification,
            params={"n_estimators": 25, "max_depth": 3},
        )
        model = result["model"]
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 25)
        self.assertEqual(model.max_depth, 3)

    def test_empty_training_data_raises_error(self):
        """TC-TRAIN-01 Step 4: Empty X_train or y_train raises controlled ModelTrainingError."""
        with self.assertRaises(ModelTrainingError) as ctx:
            ModelService.train_model(
                model_name="Logistic Regression",
                X_train=[],
                y_train=self.y_classification,
            )
        self.assertIn("empty", str(ctx.exception).lower())

        with self.assertRaises(ModelTrainingError) as ctx:
            ModelService.train_model(
                model_name="Logistic Regression",
                X_train=self.X_classification,
                y_train=[],
            )
        self.assertIn("empty", str(ctx.exception).lower())

    def test_mismatched_sample_sizes_raises_error(self):
        """TC-TRAIN-01 Step 5: Mismatched X and y lengths raise ModelTrainingError."""
        with self.assertRaises(ModelTrainingError) as ctx:
            ModelService.train_model(
                model_name="Logistic Regression",
                X_train=self.X_classification[:30],
                y_train=self.y_classification,  # 50 labels vs 30 samples
            )
        self.assertIn("mismatched", str(ctx.exception).lower())

    def test_unsupported_model_raises_error(self):
        """TC-TRAIN-01 Step 6: Unsupported or unregistered model name raises ModelTrainingError."""
        with self.assertRaises(ModelTrainingError):
            ModelService.train_model(
                model_name="Unsupported Quantum Neural Net",
                X_train=self.X_classification,
                y_train=self.y_classification,
            )


if __name__ == "__main__":
    unittest.main()
