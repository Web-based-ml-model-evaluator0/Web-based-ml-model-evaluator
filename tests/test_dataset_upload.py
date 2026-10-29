"""
Unit test for Issue #2 (ML-F-002 Dataset Upload, TC-DATA-01).
Tests upload of CSV files, file-like streams, and error handling for missing/empty files.
"""

import unittest
import io
from pathlib import Path
import pandas as pd

from src.services.dataset_service import DatasetService, DatasetValidationError


class TestDatasetUpload(unittest.TestCase):
    def setUp(self):
        self.sample_csv_path = Path("sample_data/breast_cancer.csv")

    def test_load_csv_from_valid_file_path(self):
        """TC-DATA-01 Step 1: Valid CSV file from file path is accepted and loaded."""
        df = DatasetService.load_csv(self.sample_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 569)
        self.assertEqual(len(df.columns), 31)

    def test_load_csv_from_bytes_io_stream(self):
        """TC-DATA-01 Step 2: Uploaded file-like buffer (e.g. Streamlit upload) is accepted."""
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
        self.assertListEqual(df.columns.tolist(), ["feature1", "feature2", "feature3", "label"])

    def test_load_csv_empty_file_rejected(self):
        """TC-DATA-01 Step 3: Empty file (0 bytes) is rejected with clear error."""
        empty_buffer = io.BytesIO(b"")
        empty_buffer.name = "empty.csv"
        with self.assertRaises(DatasetValidationError) as ctx:
            DatasetService.load_csv(empty_buffer)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_load_csv_none_source_rejected(self):
        """TC-DATA-01 Step 4: None file source is safely rejected."""
        with self.assertRaises(DatasetValidationError):
            DatasetService.load_csv(None)

    def test_uploaded_dataset_available_in_workspace(self):
        """TC-DATA-01 Step 5: Uploaded dataset has readable tabular structure."""
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


if __name__ == "__main__":
    unittest.main()
