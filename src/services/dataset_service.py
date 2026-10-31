
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
        Loads a CSV dataset from a file path or file-like object.
        Validates basic tabular readability and file size.
        """
        if file_source is None:
            raise DatasetValidationError("No file source provided.")

        try:
            if isinstance(file_source, (str, Path)):
                path = Path(file_source)

                if not path.exists():
                    raise DatasetValidationError(
                        f"File '{path}' does not exist."
                    )

                if not path.is_file():
                    raise DatasetValidationError(
                        "The provided path is not a file."
                    )

                file_size = path.stat().st_size

                if file_size == 0:
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                if file_size > cls.MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        "The CSV file exceeds the maximum allowed size of 50 MB."
                    )

                df = pd.read_csv(path)

            elif hasattr(file_source, "read"):

                # File-like object (e.g. Streamlit UploadedFile or BytesIO)
                if hasattr(file_source, "size"):
                    if file_source.size == 0:
                        raise DatasetValidationError(
                            "The uploaded CSV file is empty (0 bytes)."
                        )

                    if file_source.size > cls.MAX_FILE_SIZE_BYTES:
                        raise DatasetValidationError(
                            "The CSV file exceeds the maximum allowed size of 50 MB."
                        )

                # Check size when getvalue() is available
                if hasattr(file_source, "getvalue"):
                    content = file_source.getvalue()

                    if len(content) == 0:
                        raise DatasetValidationError(
                            "The uploaded CSV file is empty (0 bytes)."
                        )

                    if len(content) > cls.MAX_FILE_SIZE_BYTES:
                        raise DatasetValidationError(
                            "The CSV file exceeds the maximum allowed size of 50 MB."
                        )

                # Reset file pointer before reading
                if hasattr(file_source, "seek"):
                    file_source.seek(0)

                df = pd.read_csv(file_source)

            else:
                raise DatasetValidationError(
                    "Unsupported file source type."
                )

        except DatasetValidationError:
            raise

        except pd.errors.EmptyDataError:
            raise DatasetValidationError(
                "The CSV file is empty (has no content or headers)."
            )

        except pd.errors.ParserError as e:
            raise DatasetValidationError(
                f"Malformed CSV structure: {str(e)}"
            )

        except Exception as e:
            raise DatasetValidationError(
                f"Unable to parse CSV file: {str(e)}"
            )

        return cls.validate_dataset(df)

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates dataset structure, headers, and usable data.

        Ensures the DataFrame contains:
        - At least two columns.
        - At least one usable row.
        - Valid and unique column names.
        - At least one usable value in every column.

        Individual missing values are preserved for preprocessing.
        """

        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Invalid DataFrame object."
            )

        # Check minimum number of columns
        if len(df.columns) < 2:
            raise DatasetValidationError(
                f"The dataset must have at least 2 columns "
                f"(features and target), found {len(df.columns)}."
            )

        # Check minimum number of rows
        if df.empty or len(df) == 0:
            raise DatasetValidationError(
                "The dataset contains 0 rows."
            )

        # Normalize column names
        df = df.copy()
        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        # Reject blank column names
        if any(not column for column in df.columns):
            raise DatasetValidationError(
                "Some column names are blank. "
                "Please provide valid column names."
            )

        # Reject datasets with only unnamed placeholder headers
        unnamed_cols = [
            column
            for column in df.columns
            if column.startswith("Unnamed:")
        ]

        if len(unnamed_cols) == len(df.columns):
            raise DatasetValidationError(
                "The CSV appears to lack valid header names."
            )

        # Reject duplicate column names
        if not df.columns.is_unique:
            raise DatasetValidationError(
                "Column names must be unique. "
                "Please remove duplicate headers."
            )

        # Reject datasets with no usable data
        if df.dropna(how="all").empty:
            raise DatasetValidationError(
                "The dataset contains no usable data."
            )

        # Remove completely empty rows
        df = df.dropna(how="all").reset_index(drop=True)

        # Reject columns containing only missing values
        empty_columns = [
            column
            for column in df.columns
            if df[column].isna().all()
        ]

        if empty_columns:
            raise DatasetValidationError(
                f"These columns contain no usable values: {empty_columns}"
            )

        # Preserve individual missing values for preprocessing
        return df

    @classmethod
    def get_dataset_profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates basic profiling information: dimensions, columns,
        data types, missing values, and descriptive statistics.
        """

        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Cannot profile a null or invalid DataFrame."
            )

        columns_info = []

        for col in df.columns:
            columns_info.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct": (
                    float((df[col].isnull().sum() / len(df)) * 100)
                    if len(df) > 0 else 0.0
                ),
                "unique_count": int(df[col].nunique()),
            })

        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": [str(c) for c in df.columns],
            "column_details": columns_info,
            "has_missing_values": bool(
                df.isnull().sum().sum() > 0
            ),
            "total_missing_values": int(
                df.isnull().sum().sum()
            ),
            "numeric_columns": [
                str(c)
                for c in df.select_dtypes(include=["number"]).columns
            ],
            "categorical_columns": [
                str(c)
                for c in df.select_dtypes(exclude=["number"]).columns
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
        Ensures target column exists, is not in features,
        and the feature selection is valid.
        """

        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Invalid DataFrame object."
            )

        if target_column not in df.columns:
            raise DatasetValidationError(
                f"Target column '{target_column}' "
                "does not exist in dataset."
            )

        if not feature_columns:
            raise DatasetValidationError(
                "At least one feature column must be selected."
            )

        if target_column in feature_columns:
            raise DatasetValidationError(
                "Target column cannot also be selected "
                "as an input feature."
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise DatasetValidationError(
                "Duplicate feature columns are not allowed."
            )

        missing_features = [
            col for col in feature_columns
            if col not in df.columns
        ]

        if missing_features:
            raise DatasetValidationError(
                f"Selected feature columns not found in dataset: "
                f"{missing_features}"
            )

        return feature_columns, target_column
