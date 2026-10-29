
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
        Load a CSV dataset from a file path, file-like object, or bytes buffer.
        Validate basic tabular readability and file size.
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
                        f"'{path}' is not a file."
                    )

                if path.stat().st_size == 0:
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                if path.stat().st_size > cls.MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        "The uploaded CSV file exceeds the 50 MB size limit."
                    )

                df = pd.read_csv(path)

            elif hasattr(file_source, "read"):
                # File-like objects, such as Streamlit UploadedFile or BytesIO.
                if (
                    hasattr(file_source, "size")
                    and file_source.size == 0
                ):
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                # Check file size without consuming the stream when possible.
                if hasattr(file_source, "getvalue"):
                    content = file_source.getvalue()

                    if len(content) == 0:
                        raise DatasetValidationError(
                            "The uploaded CSV file is empty (0 bytes)."
                        )

                    if len(content) > cls.MAX_FILE_SIZE_BYTES:
                        raise DatasetValidationError(
                            "The uploaded CSV file exceeds the 50 MB size limit."
                        )

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
                "The CSV file is empty or has no headers."
            )

        except pd.errors.ParserError as exc:
            raise DatasetValidationError(
                f"Malformed CSV structure: {exc}"
            )

        except UnicodeDecodeError:
            raise DatasetValidationError(
                "Unable to decode the CSV file. Please use UTF-8 encoding."
            )

        except Exception as exc:
            raise DatasetValidationError(
                f"Unable to parse CSV file: {exc}"
            ) from exc

        return cls.validate_dataset(df)

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that a DataFrame contains rows, at least two columns,
        and valid column names.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Invalid DataFrame object."
            )

        if df.empty or len(df) == 0:
            raise DatasetValidationError(
                "The dataset contains 0 rows."
            )

        if len(df.columns) < 2:
            raise DatasetValidationError(
                "The dataset must have at least 2 columns "
                f"(features and target), found {len(df.columns)}."
            )

        # Reject datasets where all column names are unnamed placeholders.
        unnamed_cols = [
            col
            for col in df.columns
            if str(col).startswith("Unnamed:")
        ]

        if len(unnamed_cols) == len(df.columns):
            raise DatasetValidationError(
                "The CSV appears to lack valid header names."
            )

        # Reject missing or blank column names.
        if any(
            pd.isna(col) or str(col).strip() == ""
            for col in df.columns
        ):
            raise DatasetValidationError(
                "The dataset contains missing or blank column names."
            )

        return df

    @classmethod
    def get_dataset_profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic profiling information:
        dimensions, columns, data types, missing values, and unique counts.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Cannot profile an invalid or null DataFrame."
            )

        if len(df) == 0:
            raise DatasetValidationError(
                "Cannot profile a dataset with 0 rows."
            )

        columns_info = []

        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            unique_count = int(df[col].nunique())

            columns_info.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "missing_count": missing_count,
                "missing_pct": float(
                    (missing_count / len(df)) * 100
                ),
                "unique_count": unique_count,
            })

        total_missing = int(df.isnull().sum().sum())

        return {
            "num_rows": int(len(df)),
            "num_columns": int(len(df.columns)),
            "columns": [str(col) for col in df.columns],
            "column_details": columns_info,
            "has_missing_values": total_missing > 0,
            "total_missing_values": total_missing,
            "numeric_columns": [
                str(col)
                for col in df.select_dtypes(
                    include=["number"]
                ).columns
            ],
            "categorical_columns": [
                str(col)
                for col in df.select_dtypes(
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
        Validate the target column and selected feature columns.
        Ensure that the target exists, is not selected as a feature,
        and at least one valid feature is selected.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Invalid DataFrame object."
            )

        if target_column not in df.columns:
            raise DatasetValidationError(
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            )

        if not feature_columns:
            raise DatasetValidationError(
                "At least one feature column must be selected."
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise DatasetValidationError(
                "Duplicate feature columns are not allowed."
            )

        if target_column in feature_columns:
            raise DatasetValidationError(
                "Target column cannot also be selected "
                "as an input feature."
            )

        missing_features = [
            col
            for col in feature_columns
            if col not in df.columns
        ]

        if missing_features:
            raise DatasetValidationError(
                "Selected feature columns not found in dataset: "
                f"{missing_features}"
            )

        return feature_columns, target_column
