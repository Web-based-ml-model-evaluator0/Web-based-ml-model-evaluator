
import math

import pandas as pd
import streamlit as st

from auth import show_login_gate, show_logout_button

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Web Based ML Model Evaluator",
    page_icon="📊",
    layout="wide",
)

# ==================================================\n# AUTHENTICATION (ML-F-001)\n# ==================================================\nshow_login_gate(st)\nshow_logout_button(st)\n
st.title("Web Based ML Model Evaluator")
st.write(
    "Upload a CSV dataset, explore and validate the data, "
    "train a machine-learning model, and evaluate its performance."
)

MAX_FILE_SIZE_MB = 20


# ==================================================
# 1. DATASET UPLOAD
# ==================================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV dataset",
    type=["csv"],
    help="Maximum file size: 20 MB.",
)

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
    st.stop()

if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
    st.error("File size exceeds the 20 MB limit.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, on_bad_lines="error")

except pd.errors.EmptyDataError:
    st.error("The uploaded CSV is empty.")
    st.stop()

except pd.errors.ParserError:
    st.error(
        "The CSV structure is invalid. Check quotation marks, "
        "delimiters, and row lengths."
    )
    st.stop()

except (UnicodeDecodeError, ValueError):
    st.error("Unable to read this file. Please upload a valid CSV.")
    st.stop()

except Exception as error:
    st.error(f"Could not read the uploaded file: {error}")
    st.stop()


# ==================================================
# 2. DATASET VALIDATION
# ==================================================

st.header("2. Dataset Validation")

if df.empty:
    st.error("The dataset contains no data rows.")
    st.stop()

if len(df.columns) < 2:
    st.error("The dataset must contain at least two columns.")
    st.stop()

df.columns = [str(column).strip() for column in df.columns]

if any(not column for column in df.columns):
    st.error("Some column names are blank. Please correct the CSV.")
    st.stop()

if not df.columns.is_unique:
    st.error("Column names must be unique.")
    st.stop()

if len(df) < 10:
    st.warning(
        "The dataset has fewer than 10 rows. "
        "Model evaluation may not be meaningful."
    )

if df.isna().all(axis=1).any():
    st.warning("Some rows contain only missing values.")

if any(str(col).lower().startswith("unnamed:") for col in df.columns):
    st.warning("Possible unwanted columns such as 'Unnamed: 0' were detected.")

missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

a, b, c, d = st.columns(4)
a.metric("Total Rows", len(df))
b.metric("Total Columns", len(df.columns))
c.metric("Missing Cells", missing_cells)
d.metric("Duplicate Rows", duplicate_rows)

st.success("The CSV file was loaded successfully.")


# ==================================================
# 3. DATA PREVIEW AND PROFILING
# ==================================================

st.header("3. Dataset Preview and Profiling")

st.subheader("First 20 Rows")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [str(dtype) for dtype in df.dtypes],
    "Missing Values": df.isna().sum().values,
    "Missing Percentage": (df.isna().mean().values * 100).round(2),
    "Unique Values": df.nunique(dropna=True).values,
})

st.dataframe(column_info, use_container_width=True)

st.subheader("Descriptive Statistics")
st.dataframe(
    df.describe(include="all").transpose(),
    use_container_width=True,
)

if missing_cells:
    st.warning(
        "Missing values were detected. Missing feature values are imputed "
        "during training. Rows with missing targets are excluded. "
        "Non-numeric text in a regression target is rejected."
    )
else:
    st.info("No missing values were detected.")

if duplicate_rows:
    st.warning(
        f"{duplicate_rows} duplicate rows were detected. "
        "They are reported but not automatically removed."
    )


# ==================================================
# 4. MODEL CONFIGURATION
# ==================================================

st.header("4. Configure Model Training")

target_column = st.selectbox(
    "Select the target column to predict",
    options=list(df.columns),
)

task_type = st.radio(
    "Select prediction task",
    ["Classification", "Regression"],
    horizontal=True,
)

if task_type == "Classification":
    model_name = st.selectbox(
        "Select classification model",
        ["Random Forest", "Logistic Regression", "Decision Tree"],
    )
else:
    model_name = st.selectbox(
        "Select regression model",
        ["Random Forest", "Linear Regression", "Decision Tree"],
    )

test_percentage = st.slider(
    "Test dataset percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)

st.caption(
    "The target is what the model predicts. All other columns are input features."
)


# ==================================================
# 5. TRAINING AND EVALUATION
# ==================================================

if st.button("Train Model and Evaluate", type="primary"):

    with st.spinner("Preparing data and training the model..."):

        try:
            # Exclude rows whose target is genuinely missing.
            working_df = df.dropna(subset=[target_column]).copy()

            if len(working_df) < 10:
                st.error(
                    "At least 10 rows with a non-missing target are required."
                )
                st.stop()

            X = working_df.drop(columns=[target_column])
            original_y = working_df[target_column].copy()

            # ------------------------------------------
            # TARGET VALIDATION
            # ------------------------------------------

            if task_type == "Classification":

                y = original_y.astype(str)

                if y.nunique() < 2:
                    st.error(
                        "Classification requires at least two target classes."
                    )
                    st.stop()

                counts = y.value_counts()
                number_of_classes = len(counts)

                test_rows = math.ceil(
                    len(y) * test_percentage / 100
                )
                train_rows = len(y) - test_rows

                if (
                    counts.min() >= 2
                    and test_rows >= number_of_classes
                    and train_rows >= number_of_classes
                ):
                    stratify_target = y
                else:
                    stratify_target = None
                    st.warning(
                        "Stratified splitting is not possible for this "
                        "class distribution. Using a regular split."
                    )

            else:
                # Reject text such as "unknown" in numeric targets.
                y = pd.to_numeric(original_y, errors="coerce")

                invalid_target = original_y.notna() & y.isna()

                if invalid_target.any():
                    bad_count = int(invalid_target.sum())

                    examples = (
                        original_y.loc[invalid_target]
                        .astype(str)
                        .drop_duplicates()
                        .head(5)
                        .tolist()
                    )

                    st.error(
                        f"The target column '{target_column}' contains "
                        f"{bad_count} non-numeric value(s). "
                        f"Examples: {examples}. "
                        "Correct these values before training."
                    )
                    st.stop()

                if y.isna().any():
                    st.error(
                        "The regression target contains missing or invalid "
                        "values. Please correct the target column."
                    )
                    st.stop()

                if len(y) < 10:
                    st.error(
                        "Too few rows have a valid numerical target."
                    )
                    st.stop()

                if y.nunique() < 2:
                    st.error(
                        "Regression requires at least two different "
                        "numerical target values."
                    )
                    st.stop()

                stratify_target = None

            # ------------------------------------------
            # FEATURE VALIDATION
            # ------------------------------------------

            empty_columns = X.columns[X.isna().all()].tolist()

            if empty_columns:
                X = X.drop(columns=empty_columns)
                st.warning(
                    "Excluded completely empty feature columns: "
                    + ", ".join(empty_columns)
                )

            if X.shape[1] == 0:
                st.error("No usable feature columns remain.")
                st.stop()

            numeric_features = X.select_dtypes(
                include=["number"]
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=["number"]
            ).columns.tolist()

            transformers = []

            if numeric_features:
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])

                transformers.append(
                    ("numeric", numeric_pipeline, numeric_features)
                )

            if categorical_features:
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ])

                transformers.append(
                    ("categorical", categorical_pipeline, categorical_features)
                )

            preprocessor = ColumnTransformer(
                transformers=transformers
            )

            # ------------------------------------------
            # MODEL SELECTION
            # ------------------------------------------

            if task_type == "Classification":
                models = {
                    "Random Forest": RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    "Logistic Regression": LogisticRegression(
                        max_iter=2000,
                    ),
                    "Decision Tree": DecisionTreeClassifier(
                        random_state=42,
                    ),
                }
            else:
                models = {
                    "Random Forest": RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(
                        random_state=42,
                    ),
                }

            selected_model = models[model_name]

            # Split before fitting preprocessing to avoid data leakage.
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_percentage / 100,
                    random_state=42,
                    stratify=stratify_target,
                )
            except ValueError as error:
                st.error(
                    "Unable to create the train/test split. Use more rows, "
                    "adjust the test percentage, or check class counts."
                )
                st.caption(str(error))
                st.stop()

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", selected_model),
            ])

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            # ------------------------------------------
            # EVALUATION RESULTS
            # ------------------------------------------

            st.header("5. Evaluation Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Training Rows", len(X_train))
            col2.metric("Testing Rows", len(X_test))
            col3.metric("Input Features", X.shape[1])

            st.write(f"**Selected Model:** {model_name}")
            st.write(f"**Prediction Task:** {task_type}")
            st.write(f"**Target Column:** {target_column}")

            if task_type == "Classification":

                accuracy = accuracy_score(y_test, predictions)

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy:.4f}")
                m2.metric("Precision", f"{precision:.4f}")
                m3.metric("Recall", f"{recall:.4f}")
                m4.metric("F1 Score", f"{f1:.4f}")

                st.subheader("Confusion Matrix")

                actual_labels = pd.Series(y_test).astype(str)
                predicted_labels = pd.Series(predictions).astype(str)

                labels = sorted(
                    set(actual_labels) | set(predicted_labels)
                )

                matrix = confusion_matrix(
                    actual_labels,
                    predicted_labels,
                    labels=labels,
                )

                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=[f"Actual: {label}" for label in labels],
                        columns=[f"Predicted: {label}" for label in labels],
                    ),
                    use_container_width=True,
                )

                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                )

                st.dataframe(
                    pd.DataFrame(report).transpose(),
                    use_container_width=True,
                )

            else:

                mae = mean_absolute_error(y_test, predictions)

                rmse = math.sqrt(
                    mean_squared_error(y_test, predictions)
                )

                r2 = (
                    r2_score(y_test, predictions)
                    if len(y_test) >= 2
                    else float("nan")
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{mae:.4f}")
                m2.metric("RMSE", f"{rmse:.4f}")
                m3.metric("R² Score", f"{r2:.4f}")

            # ------------------------------------------
            # ACTUAL VS PREDICTED AND DOWNLOAD
            # ------------------------------------------

            results_df = pd.DataFrame({
                "Actual": pd.Series(y_test).reset_index(drop=True),
                "Predicted": pd.Series(predictions),
            })

            st.subheader("Actual vs Predicted Values")

            st.dataframe(
                results_df.head(50),
                use_container_width=True,
            )

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv_data,
                file_name="model_predictions.csv",
                mime="text/csv",
            )

            st.success(
                "Model training and evaluation completed successfully. "
                "Metrics are calculated on the held-out test dataset."
            )

        except ValueError as error:
            st.error(
                "Training failed. Check the target column, data types, "
                "class distribution, and number of rows."
            )
            st.caption(str(error))

        except Exception as error:
            st.error("An unexpected error occurred.")
            st.caption(str(error))
