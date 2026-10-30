
    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates dataset structure, headers, and usable data.
        Ensures the DataFrame contains rows, at least two columns,
        valid column names, and usable values.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                "Invalid DataFrame object."
            )

        if len(df.columns) < 2:
            raise DatasetValidationError(
                f"The dataset must have at least 2 columns "
                f"(features and target), found {len(df.columns)}."
            )

        if df.empty or len(df) == 0:
            raise DatasetValidationError(
                "The dataset contains 0 rows."
            )

        # Normalize column names for consistent validation
        df.columns = [str(column).strip() for column in df.columns]

        # Reject missing or blank column names
        if any(not column for column in df.columns):
            raise DatasetValidationError(
                "The dataset contains missing or blank column names."
            )

        # Reject datasets where all column names are unnamed placeholders
        unnamed_cols = [
            col for col in df.columns
            if col.startswith("Unnamed:")
        ]

        if len(unnamed_cols) == len(df.columns):
            raise DatasetValidationError(
                "The CSV appears to lack valid header names."
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
            column for column in df.columns
            if df[column].isna().all()
        ]

        if empty_columns:
            raise DatasetValidationError(
                f"These columns contain no usable values: {empty_columns}"
            )

        return df
