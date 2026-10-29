"""
Unit test for Issue #38: Verification of sample classification dataset.
Validates structure, row/column counts, and target distribution for sample_data/breast_cancer.csv.
"""

from pathlib import Path
import unittest
import pandas as pd


class TestSampleClassificationDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_path = Path("sample_data/breast_cancer.csv")

    def test_sample_classification_dataset_exists(self):
        self.assertTrue(self.dataset_path.exists(), "Sample dataset file sample_data/breast_cancer.csv does not exist.")
        self.assertGreater(self.dataset_path.stat().st_size, 0, "Sample dataset file is empty.")

    def test_sample_classification_dataset_schema(self):
        df = pd.read_csv(self.dataset_path)

        # Verify rows and columns
        self.assertEqual(df.shape, (569, 31), f"Expected shape (569, 31), but got {df.shape}.")

        # Verify target variable exists
        self.assertIn("diagnosis", df.columns, "Target variable 'diagnosis' is missing from dataset.")

        # Verify target classes
        target_counts = df["diagnosis"].value_counts().to_dict()
        self.assertEqual(set(target_counts.keys()), {"benign", "malignant"},
                         f"Unexpected classes in target column: {target_counts.keys()}")
        self.assertEqual(target_counts["benign"], 357, f"Expected 357 benign instances, got {target_counts['benign']}.")
        self.assertEqual(target_counts["malignant"], 212, f"Expected 212 malignant instances, got {target_counts['malignant']}.")

        # Verify feature columns are numeric
        feature_cols = [c for c in df.columns if c != "diagnosis"]
        self.assertEqual(len(feature_cols), 30, f"Expected 30 features, got {len(feature_cols)}.")
        for col in feature_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), f"Feature column {col} is not numeric.")

        # Verify zero null values
        self.assertEqual(df.isnull().sum().sum(), 0, "Sample classification dataset should not contain null values.")


if __name__ == "__main__":
    unittest.main()
