"""
Unit test for Issue #9 (ML-F-009 Model and Parameter Configuration, TC-MODEL-02).
Tests model selection, hyperparameter configuration, and parameter validation.
"""

import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.services.model_service import ModelService


class TestModelConfiguration(unittest.TestCase):
    def test_select_supported_algorithms_by_task(self):
        """TC-MODEL-02 Step 1: User can select supported algorithms by task type."""
        clf_models = ModelService.get_supported_models(task_type="classification")
        self.assertIn("Logistic Regression", clf_models)
        self.assertIn("Random Forest Classifier", clf_models)
        self.assertIn("Decision Tree Classifier", clf_models)

        reg_models = ModelService.get_supported_models(task_type="regression")
        self.assertIn("Linear Regression", reg_models)
        self.assertIn("Random Forest Regressor", reg_models)
        self.assertIn("Decision Tree Regressor", reg_models)

    def test_configure_hyperparameters_logistic_regression(self):
        """TC-MODEL-02 Step 2: Configured parameters are applied to Logistic Regression."""
        model = ModelService.create_model_instance(
            model_name="Logistic Regression",
            params={"C": 0.25, "max_iter": 500},
        )
        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.C, 0.25)
        self.assertEqual(model.max_iter, 500)

    def test_configure_hyperparameters_random_forest(self):
        """TC-MODEL-02 Step 3: Configured parameters are applied to Random Forest."""
        model = ModelService.create_model_instance(
            model_name="Random Forest Classifier",
            params={"n_estimators": 50, "max_depth": 7},
        )
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 7)

    def test_unregistered_model_raises_error(self):
        """TC-MODEL-02 Step 4: Requesting an unregistered model raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ModelService.create_model_instance("Nonexistent Deep Neural Net 9000")
        self.assertIn("not registered", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
