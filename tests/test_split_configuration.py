"""
Unit test for Issue #7 (ML-F-007 Train/Test Split Configuration, TC-PREP-03).
Tests split ratio configuration, acceptance of valid ratios, rejection of invalid ratios,
and preservation of requested split proportions.
"""

import unittest
import pandas as pd

from src.services.preprocessing_service import PreprocessingService, PreprocessingError


class TestSplitConfiguration(unittest.TestCase):
    def setUp(self):
        # 100 sample dataset
        self.df = pd.DataFrame({
            "feat1": range(100),
            "feat2": [x * 2 for x in range(100)],
            "target": ["classA"] * 60 + ["classB"] * 40,
        })
        self.features = ["feat1", "feat2"]
        self.target = "target"

    def test_valid_split_default_20_percent(self):
        """TC-PREP-03 Step 1: Default 80/20 split produces exact proportions."""
        X_train, X_test, y_train, y_test = PreprocessingService.split_data(
            df=self.df,
            feature_columns=self.features,
            target_column=self.target,
            test_size=0.20,
            random_state=42,
        )
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

    def test_valid_split_custom_ratio(self):
        """TC-PREP-03 Step 2: Custom valid split ratio (e.g. 30%) is accepted."""
        X_train, X_test, y_train, y_test = PreprocessingService.split_data(
            df=self.df,
            feature_columns=self.features,
            target_column=self.target,
            test_size=0.30,
            random_state=42,
        )
        self.assertEqual(len(X_train), 70)
        self.assertEqual(len(X_test), 30)

    def test_invalid_split_ratios_rejected(self):
        """TC-PREP-03 Step 3: Out-of-bounds split ratios are rejected with clear error."""
        invalid_ratios = [-0.1, 0.0, 0.01, 0.60, 0.99, 1.5]
        for ratio in invalid_ratios:
            with self.subTest(ratio=ratio):
                with self.assertRaises(PreprocessingError) as ctx:
                    PreprocessingService.split_data(
                        df=self.df,
                        feature_columns=self.features,
                        target_column=self.target,
                        test_size=ratio,
                    )
                self.assertIn("ratio", str(ctx.exception).lower())

    def test_stratified_split_preserves_class_distribution(self):
        """TC-PREP-03 Step 4: Stratification preserves class balance across partitions."""
        X_train, X_test, y_train, y_test = PreprocessingService.split_data(
            df=self.df,
            feature_columns=self.features,
            target_column=self.target,
            test_size=0.20,
            random_state=42,
            stratify=True,
        )
        # Class A was 60% of original, should be 12 out of 20 test samples
        test_counts = y_test.value_counts()
        self.assertEqual(test_counts["classA"], 12)
        self.assertEqual(test_counts["classB"], 8)

    def test_too_few_rows_rejected(self):
        """TC-PREP-03 Step 5: Datasets with too few rows to split safely are rejected."""
        tiny_df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})
        with self.assertRaises(PreprocessingError) as ctx:
            PreprocessingService.split_data(
                df=tiny_df,
                feature_columns=["feat"],
                target_column="target",
                test_size=0.20,
            )
        self.assertIn("few rows", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
