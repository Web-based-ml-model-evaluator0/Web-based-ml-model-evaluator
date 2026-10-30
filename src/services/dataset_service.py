
"""
Dataset Service module (ML-F-002, ML-F-003, ML-F-004, ML-SR-004, ML-SR-007).
Handles loading, validation, profiling, and schema inspection for datasets.
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path
import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset fails validation checks."""
    pass


class DatasetService:
    """
    Encapsulates dataset ingestion, validation, and profiling logic.
    Decoupled from UI rendering according to SAD 3.4 & 4.1.
    """

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit

    @classmethod
    def load_csv(cls, file_source: Any) -> pd.DataFrame:
        """
        Loads a CSV dataset from a file path, file-like object, or bytes buffer.
        Validates basic tabular readability and UTF-8 / standard encoding.
        """
        if file_source is None:
            raise DatasetValidationError("No file source provided.")

        try:
            if isinstance(file_source, (str, Path)):
                path = Path(file_source)
                if not path.exists():
                    raise DatasetValidationError(f"File '{path}' does not exist.")
                if path.stat().st_size == 0:
                    raise DatasetValidationError("The uploaded CSV file is empty (0 bytes).")
                df = pd.read_csv(path)

            elif hasattr(file_source, "read"):
                # File-like object (e.g. Streamlit UploadedFile or io.BytesIO)
                if hasattr(file_source, "size") and file_source.size == 0:
                    raise DatasetValidationError("The uploaded CSV file is empty (0 bytes).")
                df = pd.read_csv(file_source)

            else:
                raise DatasetValidationError("Unsupported file source type.")

        except DatasetValidationError:
            raise
        except pd.errors.EmptyDataError:
            raise DatasetValidationError("The CSV file is empty (has no content or headers).")
        except pd.errors.ParserError as e:
            raise DatasetValidationError(f"Malformed CSV structure: {str(e)}")
        except Exception as e:
            raise DatasetValidationError(f"Unable to parse CSV file: {str(e)}")

        return cls.validate_dataset(df)

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates dataset structure, headers, and usable data.
        Ensures the DataFrame has at least two columns and valid rows.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError("Invalid DataFrame object.")

        if len(df.columns) < 2:
            raise DatasetValidationError(
                "The dataset must have at least 2 columns (features and target)."
            )

        if df.empty or len(df) == 0:
            raise DatasetValidationError("The dataset contains 0 rows.")

        # Normalize column names for consistent validation
        df.columns = [str(column).strip() for column in df.columns]

        # Check for missing or unnamed column headers
        unnamed_cols = [
            column for column in df.columns
            if not column or column.startswith("Unnamed:")
        ]

        if unnamed_cols:
            raise DatasetValidationError(
                f"Invalid or unnamed column headers found: {unnamed_cols}"
            )

        # Ensure column names are unique
        if not df.columns.is_unique:
            raise DatasetValidationError(
                "Column names must be unique. Please remove duplicate headers."
            )

        # Reject datasets that contain no usable values
        if df.dropna(how="all").empty:
            raise DatasetValidationError(
                "The dataset contains no usable data."
            )

        # Remove rows that contain no values at all
        df = df.dropna(how="all").reset_index(drop=True)

        # Reject columns containing no usable values
        empty_columns = [
            column for column in df.columns if df[column].isna().all()
        ]

        if empty_columns:
            raise DatasetValidationError(
                f"These columns contain no usable values: {empty_columns}"
            )

        return df

    @classmethod
    def get_dataset_profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates basic profiling information: dimensions, columns, data types,
        missing values, and descriptive statistics.
        """
        if df is None:
            raise DatasetValidationError("Cannot profile a null DataFrame.")

        columns_info = []

        for col in df.columns:
            columns_info.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct": float((df[col].isnull().sum() / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
            })

        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": [str(c) for c in df.columns],
            "column_details": columns_info,
            "has_missing_values": bool(df.isnull().sum().sum() > 0),
            "total_missing_values": int(df.isnull().sum().sum()),
            "numeric_columns": [
                str(c) for c in df.select_dtypes(include=["number"]).columns
            ],
            "categorical_columns": [
                str(c) for c in df.select_dtypes(exclude=["number"]).columns
            ],
        }

    @classmethod
    def validate_target_and_features(
        cls,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
    ) -> Tuple[List[str], str]:
        """
        Validates target variable and selected features.
        Ensures target column exists, is not in features, and features is non-empty.
        """
        if target_column not in df.columns:
            raise DatasetValidationError(
                f"Target column '{target_column}' does not exist in dataset."
            )

        if not feature_columns:
            raise DatasetValidationError(
                "At least one feature column must be selected."
            )

        if target_column in feature_columns:
            raise DatasetValidationError(
                "Target column cannot also be selected as an input feature."
            )

        missing_features = [
            col for col in feature_columns if col not in df.columns
        ]

        if missing_features:
            raise DatasetValidationError(
                f"Selected feature columns not found in dataset: {missing_features}"
            )

        return feature_columns, target_column
