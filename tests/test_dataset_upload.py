
"""
Unit tests for:
- Issue #2: ML-F-002 Dataset Upload
- Issue #41: ML-F-005 Feature and Target Selection

Tests CSV uploads, file validation, feature selection,
target selection, and invalid selection handling.
"""

import unittest
import io
from pathlib import Path

import pandas as pd

from src.services.dataset_service import (
    DatasetService,
    DatasetValidationError,
)


class TestDatasetUpload(unittest.TestCase):
    """Tests for dataset upload functionality (ML-F-002)."""

    def setUp(self):
        self.sample_csv_path = Path("sample_data/breast_cancer.csv")

    def test_load_csv_from_valid_file_path(self):
        """TC-DATA-01: Valid CSV file from path is accepted."""
        df = DatasetService.load_csv(self.sample_csv_path)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 569)
        self.assertEqual(len(df.columns), 31)

    def test_load_csv_from_bytes_io_stream(self):
        """TC-DATA-01: File-like CSV buffer is accepted."""
        csv_content = (
            "feature1,feature2,feature3,label\n"
            "1.2,3.4,5.6,A\n"
            "2.3,4.5,6.7,B\n"
            "3.4,5.6,7.8,A\n"
        ).encode("utf-8")

        buffer = io.BytesIO(csv_content)
        buffer.name = "custom_test.csv"

        df = DatasetService.load_csv(buffer)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertListEqual(
            df.columns.tolist(),
            ["feature1", "feature2", "feature3", "label"],
        )

    def test_load_csv_empty_file_rejected(self):
        """TC-DATA-01: Empty file is rejected."""
        empty_buffer = io.BytesIO(b"")
        empty_buffer.name = "empty.csv"

        with self.assertRaises(DatasetValidationError) as ctx:
            DatasetService.load_csv(empty_buffer)

        self.assertIn("empty", str(ctx.exception).lower())

    def test_load_csv_none_source_rejected(self):
        """TC-DATA-01: None file source is rejected."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.load_csv(None)

    def test_uploaded_dataset_available_in_workspace(self):
        """TC-DATA-01: Uploaded dataset has readable tabular structure."""
        csv_content = (
            "col_a,col_b\n"
            "10,20\n"
            "30,40\n"
        ).encode("utf-8")

        buffer = io.BytesIO(csv_content)
        df = DatasetService.load_csv(buffer)

        self.assertTrue(hasattr(df, "head"))
        self.assertEqual(df.iloc[0]["col_a"], 10)
        self.assertEqual(df.iloc[1]["col_b"], 40)


class TestFeatureTargetSelection(unittest.TestCase):
    """Tests for feature and target selection (ML-F-005)."""

    def setUp(self):
        self.df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "salary": [30000, 40000, 50000, 60000],
            "experience": [2, 5, 8, 10],
            "department": ["HR", "IT", "Finance", "IT"],
            "performance": [1, 2, 2, 3],
        })

    def test_valid_feature_and_target_selection(self):
        """TC-PREP-01: Valid feature and target selections are accepted."""
        features = ["age", "salary", "experience"]
        target = "performance"

        selected_features, selected_target = (
            DatasetService.validate_target_and_features(
                self.df, target, features
            )
        )

        self.assertEqual(selected_features, features)
        self.assertEqual(selected_target, target)

    def test_multiple_features_can_be_selected(self):
        """TC-PREP-01: Multiple feature columns are accepted."""
        features = ["age", "salary", "department"]
        target = "performance"

        selected_features, selected_target = (
            DatasetService.validate_target_and_features(
                self.df, target, features
            )
        )

        self.assertEqual(len(selected_features), 3)
        self.assertEqual(selected_target, target)

    def test_empty_feature_selection_rejected(self):
        """TC-PREP-01: At least one feature must be selected."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.validate_target_and_features(
                self.df, "performance", []
            )

    def test_invalid_target_column_rejected(self):
        """TC-PREP-01: Target must exist in the dataset."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.validate_target_and_features(
                self.df, "unknown_target", ["age", "salary"]
            )

    def test_missing_feature_column_rejected(self):
        """TC-PREP-01: Selected features must exist in the dataset."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.validate_target_and_features(
                self.df,
                "performance",
                ["age", "unknown_feature"],
            )

    def test_target_cannot_be_selected_as_feature(self):
        """TC-PREP-01: Target cannot also be an input feature."""
        with self.assertRaises(DatasetValidationError) as ctx:
            DatasetService.validate_target_and_features(
                self.df,
                "performance",
                ["age", "salary", "performance"],
            )

        self.assertIn("target column", str(ctx.exception).lower())

    def test_duplicate_features_rejected(self):
        """TC-PREP-01: Duplicate feature selections are rejected."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.validate_target_and_features(
                self.df,
                "performance",
                ["age", "salary", "age"],
            )

    def test_empty_target_selection_rejected(self):
        """TC-PREP-01: Target selection cannot be empty."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.validate_target_and_features(
                self.df, "", ["age", "salary"]
            )

    def test_invalid_dataframe_rejected(self):
        """TC-PREP-01: Invalid dataset input is rejected."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.validate_target_and_features(
                None, "performance", ["age", "salary"]
            )

    def test_selected_features_are_returned_without_modification(self):
        """TC-PREP-01: Valid selections are retained."""
        features = ["age", "salary"]
        original_features = features.copy()

        selected_features, selected_target = (
            DatasetService.validate_target_and_features(
                self.df, "performance", features
            )
        )

        self.assertEqual(selected_features, original_features)
        self.assertEqual(selected_target, "performance")
        self.assertIsNot(selected_features, features)


if __name__ == "__main__":
    unittest.main()
