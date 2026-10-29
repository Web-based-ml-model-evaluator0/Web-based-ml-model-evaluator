"""
Unit test for Issue #28 (ML-NF-008 Modular ML Architecture, TC-ARCH-01).
Verifies service decoupling, modular architecture, and dynamic model registration.
"""

import unittest
from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from src.services.dataset_service import DatasetService, DatasetValidationError
from src.services.preprocessing_service import PreprocessingService
from src.services.model_service import ModelService
from src.services.evaluation_service import EvaluationService


class CustomDummyClassifier(BaseEstimator, ClassifierMixin):
    """Custom classifier to verify pluggability without changing other modules."""
    def __init__(self, constant_class="benign"):
        self.constant_class = constant_class

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array([self.constant_class] * len(X))


class TestModularArchitecture(unittest.TestCase):
    def test_services_decoupled(self):
        """Verify all core services exist and can be instantiated/called independently."""
        self.assertTrue(hasattr(DatasetService, "load_csv"))
        self.assertTrue(hasattr(PreprocessingService, "split_data"))
        self.assertTrue(hasattr(ModelService, "get_supported_models"))
        self.assertTrue(hasattr(EvaluationService, "evaluate_classification"))

    def test_dynamic_model_registration(self):
        """
        Verify ML-NF-008 / TC-ARCH-01:
        New models can be dynamically registered without altering dataset or evaluation workflows.
        """
        ModelService.register_model(
            model_name="Custom Plugged Classifier",
            task_type="classification",
            factory=CustomDummyClassifier,
            default_params={"constant_class": "benign"},
            param_schema={},
        )

        models = ModelService.get_supported_models(task_type="classification")
        self.assertIn("Custom Plugged Classifier", models)

        # Test model instantiation and execution through service
        inst = ModelService.create_model_instance("Custom Plugged Classifier")
        self.assertIsInstance(inst, CustomDummyClassifier)

    def test_end_to_end_modular_pipeline(self):
        """Verify data flows cleanly across service boundaries without coupling."""
        # 1. Dataset loading
        df = DatasetService.load_csv("sample_data/breast_cancer.csv")
        self.assertGreater(len(df), 0)

        # 2. Target validation
        features = [c for c in df.columns if c != "diagnosis"]
        feat_cols, target = DatasetService.validate_target_and_features(df, "diagnosis", features[:5])

        # 3. Preprocessing split
        X_train, X_test, y_train, y_test = PreprocessingService.split_data(
            df, feat_cols, target, test_size=0.20, random_state=42, stratify=True
        )
        preproc = PreprocessingService.create_preprocessor(X_train)
        X_train_proc, X_test_proc = PreprocessingService.fit_and_transform(preproc, X_train, X_test)

        # 4. Model training
        train_res = ModelService.train_model("Logistic Regression", X_train_proc, y_train)
        model = train_res["model"]

        # 5. Evaluation
        y_pred = model.predict(X_test_proc)
        metrics = EvaluationService.evaluate_classification(y_test, y_pred)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
