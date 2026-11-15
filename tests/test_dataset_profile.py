import unittest
import pandas as pd

from src.services.dataset_service import (
    DatasetService,
    DatasetValidationError,
)


class TestDatasetProfile(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "age": [22, 25, 30, 35],
            "salary": [30000, 40000, None, 60000],
            "department": ["HR", "IT", "Finance", "IT"],
        })

    def test_profile_returns_correct_dataset_dimensions(self):
        profile = DatasetService.get_dataset_profile(self.df)

        self.assertEqual(profile["num_rows"], 4)
        self.assertEqual(profile["num_columns"], 3)

    def test_profile_returns_column_names_and_types(self):
        profile = DatasetService.get_dataset_profile(self.df)

        self.assertEqual(
            profile["columns"],
            ["age", "salary", "department"],
        )

        details = profile["column_details"]

        self.assertEqual(details[0]["name"], "age")
        self.assertEqual(details[0]["dtype"], "int64")
        self.assertEqual(details[2]["name"], "department")
        self.assertIn(details[2]["dtype"], ["object", "str"])

    def test_profile_calculates_missing_values(self):
        profile = DatasetService.get_dataset_profile(self.df)

        self.assertTrue(profile["has_missing_values"])
        self.assertEqual(profile["total_missing_values"], 1)

        salary_details = profile["column_details"][1]

        self.assertEqual(salary_details["missing_count"], 1)
        self.assertEqual(salary_details["non_null_count"], 3)
        self.assertAlmostEqual(salary_details["missing_pct"], 25.0)

    def test_profile_identifies_numeric_and_categorical_columns(self):
        profile = DatasetService.get_dataset_profile(self.df)

        self.assertEqual(
            profile["numeric_columns"],
            ["age", "salary"],
        )

        self.assertEqual(
            profile["categorical_columns"],
            ["department"],
        )

    def test_profile_rejects_invalid_dataframe(self):
        with self.assertRaises(DatasetValidationError):
            DatasetService.get_dataset_profile(None)

    def test_profile_rejects_empty_dataframe(self):
        empty_df = pd.DataFrame()

        with self.assertRaises(DatasetValidationError):
            DatasetService.get_dataset_profile(empty_df)


if __name__ == "__main__":
    unittest.main()