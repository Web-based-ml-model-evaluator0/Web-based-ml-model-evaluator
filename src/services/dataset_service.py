
"""
Dataset Service module (ML-F-002, ML-F-003, ML-F-004, ML-F-005,
ML-SR-004, ML-SR-007).

Handles dataset loading, validation, profiling, schema inspection,
and feature/target selection validation.
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when a dataset or feature/target selection is invalid."""
    pass


class DatasetService:
    """
    Encapsulates dataset ingestion, validation, profiling,
    and feature/target selection logic.
    """

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

    @classmethod
    def load_csv(cls, file_source: Any) -> pd.DataFrame:
        """
        Load a CSV from a file path, file-like object, or bytes buffer.
        Validate its structure and contents.
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

                if path.stat().st_size == 0:
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                if path.stat().st_size > cls.MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        "The CSV file exceeds the maximum size limit of 50 MB."
                    )

                df = pd.read_csv(path)

            elif hasattr(file_source, "read"):
                if (
                    hasattr(file_source, "size")
                    and file_source.size == 0
                ):
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                if (
                    hasattr(file_source, "size")
                    and file_source.size > cls.MAX_FILE_SIZE_BYTES
                ):
                    raise DatasetValidationError(
                        "The CSV file exceeds the maximum size limit of 50 MB."
                    )

                df = pd.read_csv(file_source)

            else:
                raise DatasetValidationError(
                    "Unsupported file source type."
                )

        except DatasetValidationError:
            raise
        except pd.errors.EmptyDataError:
            raise DatasetValidationError(
                "The CSV file is empty or has no headers."
            )
        except pd.errors.ParserError as e:
            raise DatasetValidationError(
                f"Malformed CSV structure: {str(e)}"
            )
        except UnicodeDecodeError:
            raise DatasetValidationError(
                "Unable to read the CSV encoding. Please use UTF-8."
            )
        except Exception as e:
            raise DatasetValidationError(
                f"Unable to parse CSV file: {str(e)}"
            )

        return cls.validate_dataset(df)

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate dataset structure, headers, and usable data.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Invalid DataFrame object."
            )

        if len(df.columns) < 2:
            raise DatasetValidationError(
                "The dataset must have at least 2 columns "
                "(features and target)."
            )

        if df.empty:
            raise DatasetValidationError(
                "The dataset contains 0 rows."
            )

        df = df.copy()
        df.columns = [str(column).strip() for column in df.columns]

        if any(not column for column in df.columns):
            raise DatasetValidationError(
                "Some column names are blank. "
                "Please provide valid column names."
            )

        if not df.columns.is_unique:
            raise DatasetValidationError(
                "Column names must be unique. "
                "Please remove duplicate headers."
            )

        if df.dropna(how="all").empty:
            raise DatasetValidationError(
                "The dataset contains no usable data."
            )

        df = df.dropna(how="all").reset_index(drop=True)

        empty_columns = [
            column for column in df.columns
            if df[column].isna().all()
        ]

        if empty_columns:
            raise DatasetValidationError(
                f"These columns contain no usable values: {empty_columns}"
            )

        return df

    @classmethod
    def get_dataset_profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate dataset dimensions, column details, data types,
        missing values, and basic profiling information.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Cannot profile an invalid DataFrame."
            )

        if len(df) == 0:
            raise DatasetValidationError(
                "Cannot profile an empty dataset."
            )

        columns_info = []

        for col in df.columns:
            missing_count = int(df[col].isnull().sum())

            columns_info.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "missing_count": missing_count,
                "missing_pct": float(
                    (missing_count / len(df)) * 100
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
                for c in df.select_dtypes(
                    include=["number"]
                ).columns
            ],
            "categorical_columns": [
                str(c)
                for c in df.select_dtypes(
                    exclude=["number"]
                ).columns
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
        Validate the selected target and feature columns.

        Checks:
        - Dataset is a valid DataFrame.
        - Target column is selected and exists.
        - At least one feature is selected.
        - Every selected feature exists.
        - Duplicate feature selections are rejected.
        - Target cannot also be an input feature.
        """

        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Invalid dataset. Please upload a valid dataset."
            )

        if df.empty:
            raise DatasetValidationError(
                "Cannot select features from an empty dataset."
            )

        if not isinstance(target_column, str) or not target_column.strip():
            raise DatasetValidationError(
                "Please select a target variable."
            )

        if target_column not in df.columns:
            raise DatasetValidationError(
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            )

        if not isinstance(feature_columns, list) or not feature_columns:
            raise DatasetValidationError(
                "Please select at least one feature column."
            )

        if any(
            not isinstance(col, str) or not col.strip()
            for col in feature_columns
        ):
            raise DatasetValidationError(
                "Feature selections must contain valid column names."
            )

        missing_features = [
            col for col in feature_columns
            if col not in df.columns
        ]

        if missing_features:
            raise DatasetValidationError(
                f"Selected feature columns not found: {missing_features}"
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise DatasetValidationError(
                "Duplicate feature columns are not allowed."
            )

        if target_column in feature_columns:
            raise DatasetValidationError(
                "The target column cannot also be selected "
                "as an input feature."
            )

        return feature_columns.copy(), target_column
