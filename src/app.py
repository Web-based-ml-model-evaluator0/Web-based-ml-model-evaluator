
"""
Web-Based ML Model Evaluator - Streamlit Presentation Layer
Refactored into a modular architecture decoupling Presentation, Service, and ML layers.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.services.dataset_service import DatasetService, DatasetValidationError
from src.services.preprocessing_service import PreprocessingService, PreprocessingError
from src.services.model_service import ModelService
from src.services.evaluation_service import EvaluationService


def main() -> None:
    st.set_page_config(
        page_title="Web Based ML Model Evaluator",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Web Based ML Model Evaluator")
    st.caption("Sprint 1 Baseline: Dataset Upload, Profiling, Preprocessing, Model Training & Evaluation")

    # Sidebar: Data Source Selection
    st.sidebar.header("📁 1. Dataset Selection")
    data_source = st.sidebar.radio(
        "Choose Data Source:",
        ["Use Sample Classification Dataset", "Upload Custom CSV File"],
    )

    df: Optional[pd.DataFrame] = None

    if data_source == "Use Sample Classification Dataset":
        sample_path = root_dir / "sample_data" / "breast_cancer.csv"
        if sample_path.exists():
            try:
                df = DatasetService.load_csv(sample_path)
                st.sidebar.success("Loaded: Breast Cancer Diagnostic (569 rows, 31 columns)")
            except DatasetValidationError as e:
                st.sidebar.error(f"Error loading sample dataset: {e}")
        else:
            st.sidebar.error(f"Sample dataset not found at {sample_path}")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = DatasetService.load_csv(uploaded_file)
                st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(df)} rows, {len(df.columns)} columns)")
            except DatasetValidationError as e:
                st.sidebar.error(f"Upload validation failed: {e}")
                st.stop()

    if df is None:
        st.info("👈 Please select or upload a dataset using the sidebar to begin.")
        return

    # Section 2: Dataset Preview and Profiling
    st.subheader("📋 Dataset Preview and Profiling")
    profile = DatasetService.get_dataset_profile(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", profile["num_rows"])
    col2.metric("Total Columns", profile["num_columns"])
    col3.metric("Numeric Features", len(profile["numeric_columns"]))
    col4.metric("Missing Values", profile["total_missing_values"])

    with st.expander("🔍 View First 5 Rows of Dataset", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📊 View Column Schema & Data Types"):
        col_df = pd.DataFrame(profile["column_details"])
        st.dataframe(col_df, use_container_width=True)

    st.divider()

    # Section 3: Feature & Target Selection
    st.subheader("🎯 Feature and Target Selection")
    all_columns = df.columns.tolist()

    default_target_idx = len(all_columns) - 1
    if "diagnosis" in all_columns:
        default_target_idx = all_columns.index("diagnosis")

    c1, c2 = st.columns([1, 2])
    with c1:
        target_column = st.selectbox("Select Target Variable (Y):", all_columns, index=default_target_idx)

    available_features = [col for col in all_columns if col != target_column]
    with c2:
        feature_columns = st.multiselect(
            "Select Feature Columns (X):",
            available_features,
            default=available_features,
        )

    try:
        DatasetService.validate_target_and_features(df, target_column, feature_columns)
    except DatasetValidationError as e:
        st.error(f"Selection Error: {e}")
        return

    # Determine Task Type based on Target Column
    target_series = df[target_column]
    is_classification = False
    if target_series.dtype == "object" or target_series.nunique() <= 10 or pd.api.types.is_categorical_dtype(target_series):
        is_classification = True
        task_type = "classification"
    else:
        is_classification = False
        task_type = "regression"

    st.info(f"Detected Task Type: **{task_type.upper()}** (Target '{target_column}' has {target_series.nunique()} unique values)")

    st.divider()

    # Section 4: Preprocessing & Model Configuration
    st.subheader("⚙️ Preprocessing & Model Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

    with cfg_col1:
        test_size_pct = st.slider("Test Set Split Ratio (%)", min_value=10, max_value=40, value=20, step=5)
        test_size = test_size_pct / 100.0

    with cfg_col2:
        missing_strategy = st.selectbox("Missing Value Imputation:", ["median", "mean", "most_frequent"])
        scale_features = st.checkbox("Scale Numeric Features (StandardScaler)", value=True)

    with cfg_col3:
        supported_models = ModelService.get_supported_models(task_type=task_type)
        selected_model_name = st.selectbox("Select ML Model:", supported_models)

    # Model Parameters Configuration
    model_info = ModelService.get_model_info(selected_model_name)
    param_schema = model_info.get("param_schema", {})
    configured_params = {}

    if param_schema:
        with st.expander(f"🛠️ Hyperparameters for {selected_model_name}", expanded=False):
            pcols = st.columns(len(param_schema))
            for i, (param_name, schema) in enumerate(param_schema.items()):
                with pcols[i]:
                    if schema["type"] == "int":
                        configured_params[param_name] = st.slider(
                            param_name,
                            min_value=schema["min"],
                            max_value=schema["max"],
                            value=schema["default"],
                        )
                    elif schema["type"] == "float":
                        configured_params[param_name] = st.slider(
                            param_name,
                            min_value=float(schema["min"]),
                            max_value=float(schema["max"]),
                            value=float(schema["default"]),
                            step=0.1,
                        )

    st.divider()

    # Section 5: Training & Evaluation
    st.subheader("🚀 Model Training and Evaluation")
    if st.button("Train and Evaluate Model", type="primary"):
        with st.spinner("Processing data, training model, and evaluating performance..."):
            try:
                # 1. Split Data
                X_train, X_test, y_train, y_test = PreprocessingService.split_data(
                    df=df,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    test_size=test_size,
                    stratify=is_classification,
                )

                # 2. Build Preprocessor and Fit on Train Data Only (Prevent Data Leakage)
                preprocessor = PreprocessingService.create_preprocessor(
                    X_train=X_train,
                    numeric_strategy=missing_strategy,
                    scale_features=scale_features,
                )
                X_train_proc, X_test_proc = PreprocessingService.fit_and_transform(
                    preprocessor=preprocessor,
                    X_train=X_train,
                    X_test=X_test,
                )

                # 3. Train Model
                train_result = ModelService.train_model(
                    model_name=selected_model_name,
                    X_train=X_train_proc,
                    y_train=y_train,
                    params=configured_params,
                )
                model = train_result["model"]
                training_time = train_result["training_time_seconds"]

                # 4. Predict
                y_pred = model.predict(X_test_proc)

                # 5. Evaluate
                st.success(f"Model '{selected_model_name}' trained successfully in {training_time:.3f} seconds!")

                if is_classification:
                    eval_results = EvaluationService.evaluate_classification(y_true=y_test, y_pred=y_pred)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{eval_results['accuracy']:.4f}")
                    m2.metric("Precision", f"{eval_results['precision']:.4f}")
                    m3.metric("Recall", f"{eval_results['recall']:.4f}")
                    m4.metric("F1-Score", f"{eval_results['f1_score']:.4f}")

                    st.markdown("#### Confusion Matrix")
                    cm_df = pd.DataFrame(
                        eval_results["confusion_matrix"],
                        index=[f"Actual {l}" for l in eval_results["labels"]],
                        columns=[f"Pred {l}" for l in eval_results["labels"]],
                    )
                    st.dataframe(cm_df, use_container_width=True)
                else:
                    eval_results = EvaluationService.evaluate_regression(y_true=y_test, y_pred=y_pred)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MAE", f"{eval_results['mean_absolute_error']:.4f}")
                    m2.metric("MSE", f"{eval_results['mean_squared_error']:.4f}")
                    m3.metric("RMSE", f"{eval_results['root_mean_squared_error']:.4f}")
                    m4.metric("R² Score", f"{eval_results['r2_score']:.4f}")

                # Display Predictions Sample
                st.markdown("#### Sample Predictions vs Actual Values")
                results_df = pd.DataFrame({
                    "Actual": y_test.values[:10],
                    "Predicted": y_pred[:10],
                })
                st.dataframe(results_df, use_container_width=True)

            except (PreprocessingError, DatasetValidationError) as e:
                st.error(f"Unable to evaluate this dataset: {e}")
                st.info(
                    "Check your target and feature selections, missing values, "
                    "and dataset format, then try again."
                )
            except Exception:
                st.error(
                    "Model evaluation failed unexpectedly. "
                    "Check your dataset and model settings, then try again."
                )


if __name__ == "__main__":
    main()
