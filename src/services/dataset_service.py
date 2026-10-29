"""
Dataset Service module (ML-F-002, ML-F-003, ML-F-004, ML-SR-004, ML-SR-007).
Handles loading, validation, profiling, and schema inspection for datasets.
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path
import io
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
        Loads and validates a CSV dataset from a file path or file-like object.
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

                if path.suffix.lower() != ".csv":
                    raise DatasetValidationError(
                        "Unsupported file type. Please provide a CSV file."
                    )

                if path.stat().st_size > cls.MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        "The uploaded file exceeds the 50 MB size limit."
                    )

                if path.stat().st_size == 0:
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                df = pd.read_csv(path)

            elif hasattr(file_source, "read"):
                filename = getattr(file_source, "name", "")

                if filename and Path(filename).suffix.lower() != ".csv":
                    raise DatasetValidationError(
                        "Unsupported file type. Please upload a CSV file."
                    )

                if hasattr(file_source, "size"):
                    if file_source.size > cls.MAX_FILE_SIZE_BYTES:
                        raise DatasetValidationError(
                            "The uploaded file exceeds the 50 MB size limit."
                        )

                    if file_source.size == 0:
                        raise DatasetValidationError(
                            "The uploaded CSV file is empty (0 bytes)."
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
        """Validate dataset structure and column names."""
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError("Invalid DataFrame object.")

        if df.empty:
            raise DatasetValidationError("The dataset contains 0 rows.")

        if df.dropna(how="all").empty:
            raise DatasetValidationError(
                "The dataset contains no usable data."
            )
        if len(df.columns) < 2:
            raise DatasetValidationError(
                "The dataset must have at least 2 columns (features and target)."
            )

        # Normalize column names before checking them.
        df.columns = [str(column).strip() for column in df.columns]

        if any(not column for column in df.columns):
            raise DatasetValidationError(
                "Some column names are blank. Please provide valid column names."
            )

        if not df.columns.is_unique:
            raise DatasetValidationError(
                "Column names must be unique. Please remove duplicate headers."
            )

        if all(str(column).startswith("Unnamed:") for column in df.columns):
            raise DatasetValidationError(
                "The CSV appears to lack valid header names."
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
            "numeric_columns": [str(c) for c in df.select_dtypes(include=["number"]).columns],
            "categorical_columns": [str(c) for c in df.select_dtypes(exclude=["number"]).columns],
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
            raise DatasetValidationError(f"Target column '{target_column}' does not exist in dataset.")

        if not feature_columns:
            raise DatasetValidationError("At least one feature column must be selected.")

        if target_column in feature_columns:
            raise DatasetValidationError("Target column cannot also be selected as an input feature.")

        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise DatasetValidationError(f"Selected feature columns not found in dataset: {missing_features}")

        return feature_columns, target_column
