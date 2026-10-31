"""
Unit test for Issue #6 (ML-F-006 Data Preprocessing, TC-PREP-02).
Tests missing-value imputation, feature scaling, categorical encoding,
and strict prevention of data leakage.
"""

import unittest
import pandas as pd
import numpy as np

from src.services.preprocessing_service import PreprocessingService, PreprocessingError


class TestPreprocessingService(unittest.TestCase):
    def test_missing_value_imputation(self):
        """TC-PREP-02 Step 1: Missing values are imputed correctly across types."""
        data_train = pd.DataFrame({
            "num_feat": [10.0, np.nan, 30.0, 40.0],
            "cat_feat": ["A", "B", np.nan, "A"],
        })
        data_test = pd.DataFrame({
            "num_feat": [np.nan, 20.0],
            "cat_feat": [np.nan, "B"],
        })

        preprocessor = PreprocessingService.create_preprocessor(
            X_train=data_train,
            numeric_strategy="median",
            scale_features=False,
        )

        X_train_proc, X_test_proc = PreprocessingService.fit_and_transform(
            preprocessor=preprocessor,
            X_train=data_train,
            X_test=data_test,
        )

        # Verify no NaN values remain
        self.assertFalse(np.isnan(X_train_proc).any())
        self.assertFalse(np.isnan(X_test_proc).any())

    def test_feature_scaling_standard_scaler(self):
        """TC-PREP-02 Step 2: Feature scaling standardizes numeric features."""
        data_train = pd.DataFrame({
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        data_test = pd.DataFrame({
            "feature": [25.0, 35.0],
        })

        preprocessor = PreprocessingService.create_preprocessor(
            X_train=data_train,
            scale_features=True,
        )

        X_train_proc, X_test_proc = PreprocessingService.fit_and_transform(
            preprocessor=preprocessor,
            X_train=data_train,
            X_test=data_test,
        )

        # Standard scaler on train should have mean ~0 and std ~1
        self.assertAlmostEqual(float(np.mean(X_train_proc)), 0.0, places=5)
        self.assertAlmostEqual(float(np.std(X_train_proc)), 1.0, places=5)

    def test_prevention_of_data_leakage(self):
        """
        TC-PREP-02 Step 3: Verify strict prevention of data leakage.
        Transformers fit strictly on training set statistics.
        """
        # Train has values centered at 10.0
        train_df = pd.DataFrame({"val": [8.0, 10.0, 12.0]})
        # Test has extreme outliers centered at 1000.0
        test_df = pd.DataFrame({"val": [900.0, 1100.0]})

        preprocessor = PreprocessingService.create_preprocessor(
            X_train=train_df,
            scale_features=True,
        )

        # Fit preprocessor on train, transform test
        X_train_proc, X_test_proc = PreprocessingService.fit_and_transform(
            preprocessor=preprocessor,
            X_train=train_df,
            X_test=test_df,
        )

        # Mean of train scaling must be 10.0 (from train data)
        scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        self.assertAlmostEqual(float(scaler.mean_[0]), 10.0, places=5)

        # Test set is transformed using train mean (10.0), not test mean (1000.0)
        expected_test_0 = (900.0 - 10.0) / scaler.scale_[0]
        self.assertAlmostEqual(float(X_test_proc[0][0]), expected_test_0, places=4)


if __name__ == "__main__":
    unittest.main()
