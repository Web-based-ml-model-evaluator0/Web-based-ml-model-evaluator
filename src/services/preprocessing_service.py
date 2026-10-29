"""
Preprocessing Service module (ML-F-006, ML-F-007).
Handles data cleaning, missing-value imputation, feature scaling,
categorical encoding, and train/test split without data leakage.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class PreprocessingError(ValueError):
    """Raised when preprocessing or split validation fails."""
    pass


class PreprocessingService:
    """
    Manages data preparation pipeline components.
    Guarantees strict prevention of data leakage by fitting transformers
    only on the training split.
    """

    @classmethod
    def split_data(
        cls,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        test_size: float = 0.20,
        random_state: int = 42,
        stratify: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits dataset into train and test partitions.
        Validates that test_size is in (0.0, 1.0).
        """
        if not (0.05 <= test_size <= 0.50):
            raise PreprocessingError(
                f"Test split ratio must be between 0.05 and 0.50 (got {test_size})."
            )

        X = df[feature_columns].copy()
        y = df[target_column].copy()

        if len(X) < 10:
            raise PreprocessingError("Dataset has too few rows to perform a train/test split (minimum 10 rows).")

        stratify_target = y if (stratify and y.nunique() > 1 and y.value_counts().min() >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )

        return X_train, X_test, y_train, y_test

    @classmethod
    def create_preprocessor(
        cls,
        X_train: pd.DataFrame,
        numeric_strategy: str = "median",
        scale_features: bool = True,
    ) -> ColumnTransformer:
        """
        Creates a scikit-learn ColumnTransformer configured for numeric and categorical columns.
        """
        numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

        transformers = []

        if numeric_cols:
            num_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
            if scale_features:
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(steps=num_steps), numeric_cols))

        if categorical_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            transformers.append(("cat", Pipeline(steps=cat_steps), categorical_cols))

        if not transformers:
            raise PreprocessingError("No usable feature columns identified for preprocessing.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    @classmethod
    def fit_and_transform(
        cls,
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits preprocessor ONLY on X_train, and transforms both X_train and X_test.
        Ensures strict prevention of data leakage.
        """
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        return X_train_processed, X_test_processed
