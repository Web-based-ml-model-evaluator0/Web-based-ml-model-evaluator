"""
Dataset Service module (ML-F-002, ML-F-003, ML-F-004, ML-F-005,
ML-SR-004, ML-SR-007).

Handles dataset loading, validation, profiling, schema inspection,
and feature/target selection validation.
"""

from io import BytesIO
from typing import Any, Dict, List, Tuple
from pathlib import Path

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when a dataset or feature/target selection is invalid."""


class DatasetService:
    """
    Encapsulates dataset ingestion, validation, profiling,
    and feature/target selection logic.
    """

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

    @classmethod
    def _get_file_size(cls, file_source: Any) -> int:
        """Determine the size of a supported file source safely."""

        # Streamlit UploadedFile and similar objects expose .size.
        size = getattr(file_source, "size", None)
        if size is not None:
            try:
                size = int(size)
            except (TypeError, ValueError, OverflowError) as exc:
                raise DatasetValidationError(
                    "Unable to determine uploaded file size safely."
                ) from exc
            if size < 0:
                raise DatasetValidationError(
                    "Unable to determine uploaded file size safely."
                )
            return size

        # For seekable streams, measure from the end and restore position.
        if hasattr(file_source, "seek") and hasattr(file_source, "tell"):
            try:
                original_position = file_source.tell()
                file_source.seek(0, 2)
                size = file_source.tell()
                file_source.seek(original_position)
                return int(size)
            except (OSError, ValueError, AttributeError, TypeError) as exc:
                try:
                    file_source.seek(original_position)
                except (OSError, ValueError, AttributeError, TypeError):
                    pass
                raise DatasetValidationError(
                    "Unable to determine uploaded file size safely."
                ) from exc

        raise DatasetValidationError(
            "Unable to determine uploaded file size safely."
        )

    @classmethod
    def load_csv(cls, file_source: Any) -> pd.DataFrame:
        """
        Load a CSV from a path, file-like object, or bytes buffer.
        Enforce file size and validate its structure and contents.
        """
        if file_source is None:
            raise DatasetValidationError("No file source provided.")

        source_to_read = file_source

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
                if path.suffix.lower() != ".csv":
                    raise DatasetValidationError(
                        "Unsupported file type. Please provide a CSV file."
                    )

                file_size = path.stat().st_size
                if file_size > cls.MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        "The uploaded file exceeds the 50 MB size limit."
                    )
                if file_size == 0:
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                source_to_read = path

            elif isinstance(file_source, (bytes, bytearray, memoryview)):
                file_size = len(file_source)
                if file_size > cls.MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        "The uploaded file exceeds the 50 MB size limit."
                    )
                if file_size == 0:
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )
                source_to_read = BytesIO(bytes(file_source))

            elif hasattr(file_source, "read"):
                filename = getattr(file_source, "name", "")
                if filename and Path(str(filename)).suffix.lower() != ".csv":
                    raise DatasetValidationError(
                        "Unsupported file type. Please upload a CSV file."
                    )

                file_size = cls._get_file_size(file_source)
                if file_size > cls.MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        "The uploaded file exceeds the 50 MB size limit."
                    )
                if file_size == 0:
                    raise DatasetValidationError(
                        "The uploaded CSV file is empty (0 bytes)."
                    )

                # Ensure pandas reads from the beginning when the stream is seekable.
                if hasattr(file_source, "seek"):
                    try:
                        file_source.seek(0)
                    except (OSError, ValueError):
                        pass

            else:
                raise DatasetValidationError(
                    "Unsupported file source type."
                )

            df = pd.read_csv(source_to_read)

        except DatasetValidationError:
            raise
        except pd.errors.EmptyDataError as exc:
            raise DatasetValidationError(
                "The CSV file is empty or has no headers."
            ) from exc
        except pd.errors.ParserError as exc:
            raise DatasetValidationError(
                f"Malformed CSV structure: {str(exc)}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise DatasetValidationError(
                "Unable to read the CSV encoding. Please use UTF-8."
            ) from exc
        except Exception as exc:
            raise DatasetValidationError(
                f"Unable to parse CSV file: {str(exc)}"
            ) from exc

        return cls.validate_dataset(df)

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate dataset structure, headers, and usable data.

        Individual missing values are preserved for preprocessing.
        Completely empty rows are removed; completely empty columns
        are rejected because they cannot contribute usable data.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError("Invalid DataFrame object.")

        if df.empty:
            raise DatasetValidationError("The dataset contains 0 rows.")

        df = df.copy()
        df.columns = [str(column).strip() for column in df.columns]

        if len(df.columns) < 2:
            raise DatasetValidationError(
                "The dataset must have at least 2 columns "
                "(features and target)."
            )

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

        # Remove rows that contain no values at all.
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty:
            raise DatasetValidationError(
                "The dataset contains no usable data."
            )

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
                "missing_pct": float((missing_count / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
            })

        total_missing = int(df.isnull().sum().sum())
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": [str(column) for column in df.columns],
            "column_details": columns_info,
            "has_missing_values": bool(total_missing > 0),
            "total_missing_values": total_missing,
            "numeric_columns": [
                str(column)
                for column in df.select_dtypes(include=["number"]).columns
            ],
            "categorical_columns": [
                str(column)
                for column in df.select_dtypes(exclude=["number"]).columns
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

        Checks that the dataset is valid, the target and features exist,
        at least one feature is selected, selections are unique, and the
        target is not also used as an input feature.
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
            not isinstance(column, str) or not column.strip()
            for column in feature_columns
        ):
            raise DatasetValidationError(
                "Feature selections must contain valid column names."
            )

        missing_features = [
            column for column in feature_columns if column not in df.columns
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
