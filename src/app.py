
"""
Web-Based ML Model Evaluator - Streamlit Presentation Layer

Modular architecture:
Presentation Layer -> Service Layer -> ML Layer

Features:
- User authentication and authorization
- Admin dashboard and activity logs
- Dataset upload and profiling
- Feature and target selection
- Classification and regression detection
- Data preprocessing
- Model training and evaluation
- Training status tracking
- Experiment storage and history with user ownership
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Project Configuration
# --------------------------------------------------

root_dir = Path(__file__).resolve().parent.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# --------------------------------------------------
# Service Imports
# --------------------------------------------------

from src.services.dataset_service import (
    DatasetService,
    DatasetValidationError,
)

from src.services.preprocessing_service import (
    PreprocessingService,
    PreprocessingError,
)

from src.services.model_service import (
    ModelService,
    ModelTrainingError,
    TrainingStatus,
    TrainingStatusTracker,
)

from src.services.evaluation_service import EvaluationService

from src.services.experiment_service import (
    ExperimentStorageService,
)

from src.services.authentication_service import (
    AuthenticationService,
)

from src.services.authorization_service import (
    AuthorizationService,
)

from src.services.auth_service import AuthService

from src.services.audit_service import AuditService


# --------------------------------------------------
# Section 1: Authentication
# --------------------------------------------------

def initialize_authentication():
    """Initialize authentication for development."""

    if "auth_service" not in st.session_state:

        users = {
            "admin": {
                "password_hash": AuthService.hash_password(
                    "Admin@123"
                ),
                "role": "admin",
            },
            "student": {
                "password_hash": AuthService.hash_password(
                    "Student@123"
                ),
                "role": "user",
            },
        }

        st.session_state.auth_service = (
            AuthenticationService(users)
        )

    if "audit_service" not in st.session_state:
        st.session_state.audit_service = AuditService()


def login_screen():
    """Display login form."""

    st.title("ML Model Evaluator - Login")

    with st.form("login_form"):

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password",
        )

        submitted = st.form_submit_button("Login")

        if submitted:

            user = st.session_state.auth_service.authenticate(
                username,
                password,
            )

            if user:

                st.session_state.current_user = user

                st.session_state.audit_service.log_event(
                    event_type="LOGIN_SUCCESS",
                    username=user["username"],
                    details={
                        "role": user["role"],
                    },
                )

                st.rerun()

            else:

                st.session_state.audit_service.log_event(
                    event_type="LOGIN_FAILURE",
                    username=(
                        username.strip()
                        if username
                        else None
                    ),
                )

                st.error("Invalid username or password.")


def check_access():
    """Require authentication before accessing the application."""

    initialize_authentication()

    if "current_user" not in st.session_state:

        login_screen()
        st.stop()

    AuthorizationService.require_authenticated(
        st.session_state.current_user
    )


# --------------------------------------------------
# Section 2: Admin Dashboard
# --------------------------------------------------

def admin_dashboard():

    if not AuthorizationService.has_role(
        st.session_state.current_user,
        "admin",
    ):

        st.error(
            "Access denied. Admin privileges required."
        )

        st.stop()

    st.title("🛠️ Admin Dashboard")

    st.subheader("Model Availability")

    availability = ModelService.get_model_availability()

    for model_name, enabled in availability.items():

        new_status = st.toggle(
            model_name,
            value=enabled,
            key=f"admin_model_{model_name}",
        )

        if new_status != enabled:

            ModelService.set_model_availability(
                model_name,
                new_status,
            )

            st.session_state.audit_service.log_event(
                event_type="MODEL_AVAILABILITY_CHANGED",
                username=(
                    st.session_state.current_user["username"]
                ),
                details={
                    "model": model_name,
                    "enabled": new_status,
                },
            )

            st.rerun()

    st.subheader("📋 Activity Logs")

    events = st.session_state.audit_service.get_events()

    if events:

        st.dataframe(
            pd.DataFrame(events),
            width="stretch",
        )

    else:

        st.info("No activity logs available.")


# --------------------------------------------------
# Section 3: Main Application
# --------------------------------------------------

def main() -> None:

    st.set_page_config(
        page_title="Web Based ML Model Evaluator",
        page_icon="📊",
        layout="wide",
    )

    check_access()

    user = st.session_state.current_user

    # --------------------------------------------------
    # Sidebar and Logout
    # --------------------------------------------------

    st.sidebar.success(
        f"Logged in: {user['username']} ({user['role']})"
    )

    if st.sidebar.button("Logout"):

        st.session_state.audit_service.log_event(
            event_type="LOGOUT",
            username=user["username"],
        )

        del st.session_state.current_user

        st.rerun()

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------

    page = "Evaluator"

    if AuthorizationService.has_role(user, "admin"):

        page = st.sidebar.radio(
            "Navigation",
            [
                "Evaluator",
                "Admin Dashboard",
            ],
            key="app_navigation",
        )

    if page == "Admin Dashboard":

        admin_dashboard()
        return

    # --------------------------------------------------
    # Application Header
    # --------------------------------------------------

    st.title("📊 Web Based ML Model Evaluator")

    st.caption(
        "Sprint 1 Baseline: Dataset Upload, Profiling, "
        "Preprocessing, Model Training & Evaluation"
    )

    # --------------------------------------------------
    # Section 4: Dataset Selection
    # --------------------------------------------------

    st.sidebar.header("📁 1. Dataset Selection")

    data_source = st.sidebar.radio(
        "Choose Data Source:",
        [
            "Use Sample Classification Dataset",
            "Upload Custom CSV File",
        ],
    )

    df: Optional[pd.DataFrame] = None

    if data_source == "Use Sample Classification Dataset":

        sample_path = (
            root_dir
            / "sample_data"
            / "breast_cancer.csv"
        )

        if sample_path.exists():

            try:

                df = DatasetService.load_csv(sample_path)

                st.sidebar.success(
                    "Loaded: Breast Cancer Diagnostic "
                    "(569 rows, 31 columns)"
                )

            except DatasetValidationError as e:

                st.sidebar.error(
                    f"Error loading sample dataset: {e}"
                )

        else:

            st.sidebar.error(
                f"Sample dataset not found at {sample_path}"
            )

    else:

        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=["csv"],
        )

        if uploaded_file is not None:

            try:

                df = DatasetService.load_csv(uploaded_file)

                st.session_state.audit_service.log_event(
                    event_type="DATASET_UPLOAD_SUCCESS",
                    username=user["username"],
                    details={
                        "filename": uploaded_file.name,
                        "rows": len(df),
                        "columns": len(df.columns),
                    },
                )

                st.sidebar.success(
                    f"Uploaded: {uploaded_file.name} "
                    f"({len(df)} rows, {len(df.columns)} columns)"
                )

            except DatasetValidationError as e:

                st.sidebar.error(
                    f"Upload validation failed: {e}"
                )

                st.stop()

    if df is None:

        st.info(
            "👈 Please select or upload a dataset "
            "using the sidebar to begin."
        )

        return

    # --------------------------------------------------
    # Section 5: Dataset Preview and Profiling
    # --------------------------------------------------

    st.subheader("📋 Dataset Preview and Profiling")

    profile = DatasetService.get_dataset_profile(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        profile["num_rows"],
    )

    col2.metric(
        "Total Columns",
        profile["num_columns"],
    )

    col3.metric(
        "Numeric Features",
        len(profile["numeric_columns"]),
    )

    col4.metric(
        "Missing Values",
        profile["total_missing_values"],
    )

    with st.expander(
        "🔍 View First 5 Rows of Dataset",
        expanded=True,
    ):

        st.dataframe(
            df.head(),
            use_container_width=True,
        )

    with st.expander("📊 View Column Schema & Data Types"):

        col_df = pd.DataFrame(
            profile["column_details"]
        )

        st.dataframe(
            col_df,
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------
    # Section 6: Feature and Target Selection
    # ML-F-005 / Issue #41
    # --------------------------------------------------

    st.subheader("🎯 Feature and Target Selection")

    all_columns = df.columns.tolist()

    default_target = (
        "diagnosis"
        if "diagnosis" in all_columns
        else all_columns[-1]
    )

    # Identify the currently loaded dataset.

    dataset_signature = (
        data_source,
        tuple(all_columns),
        len(df),
        tuple(str(dtype) for dtype in df.dtypes),
        int(
            pd.util.hash_pandas_object(
                df,
                index=True,
            ).sum()
        ),
    )

    if (
        st.session_state.get("dataset_signature")
        != dataset_signature
    ):

        st.session_state["dataset_signature"] = (
            dataset_signature
        )

        st.session_state["target_column"] = (
            default_target
        )

        st.session_state["feature_columns"] = [
            col
            for col in all_columns
            if col != default_target
        ]

    # Validate stored target.

    if (
        st.session_state.get("target_column")
        not in all_columns
    ):

        st.session_state["target_column"] = (
            default_target
        )

    c1, c2 = st.columns([1, 2])

    with c1:

        target_column = st.selectbox(
            "Select Target Variable (Y):",
            options=all_columns,
            key="target_column",
            help=(
                "Choose the column that the model "
                "should predict."
            ),
        )

    available_features = [
        col
        for col in all_columns
        if col != target_column
    ]

    # Remove invalid features when target changes.

    st.session_state["feature_columns"] = [
        col
        for col in st.session_state.get(
            "feature_columns",
            [],
        )
        if col in available_features
    ]

    with c2:

        feature_columns = st.multiselect(
            "Select Feature Columns (X):",
            options=available_features,
            key="feature_columns",
            help=(
                "Select one or more columns "
                "to use as model inputs."
            ),
        )

    st.caption(
        f"Selected {len(feature_columns)} feature(s) "
        f"and 1 target variable."
    )

    # Validate feature and target selection.

    try:

        feature_columns, target_column = (
            DatasetService.validate_target_and_features(
                df=df,
                target_column=target_column,
                feature_columns=feature_columns,
            )
        )

        st.success(
            "Feature and target selections are valid."
        )

    except DatasetValidationError as e:

        st.error(f"Selection Error: {e}")

        st.info(
            "Select at least one valid feature column "
            "and one target variable to continue."
        )

        return

    # --------------------------------------------------
    # Section 7: Task Type Detection
    # --------------------------------------------------

    target_series = df[target_column]

    if (
        target_series.dtype == "object"
        or target_series.nunique() <= 10
        or isinstance(
            target_series.dtype,
            pd.CategoricalDtype,
        )
    ):

        is_classification = True
        task_type = "classification"

    else:

        is_classification = False
        task_type = "regression"

    st.info(
        f"Detected Task Type: **{task_type.upper()}** "
        f"(Target '{target_column}' has "
        f"{target_series.nunique()} unique values)"
    )

    st.divider()

    # --------------------------------------------------
    # Section 8: Preprocessing and Model Configuration
    # --------------------------------------------------

    st.subheader(
        "⚙️ Preprocessing & Model Configuration"
    )

    test_size_pct = st.slider(
        "Test Set Split Ratio (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    test_size = test_size_pct / 100.0

    missing_strategy = st.selectbox(
        "Missing Value Imputation:",
        [
            "median",
            "mean",
            "most_frequent",
        ],
    )

    scale_features = st.checkbox(
        "Scale Numeric Features (StandardScaler)",
        value=True,
    )

    supported_models = ModelService.get_supported_models(
        task_type=task_type
    )

    selected_model_name = st.selectbox(
        "Select ML Model:",
        supported_models,
    )

    # --------------------------------------------------
    # Hyperparameter Configuration
    # --------------------------------------------------

    model_info = ModelService.get_model_info(
        selected_model_name
    )

    param_schema = model_info.get(
        "param_schema",
        {},
    )

    configured_params = {}

    if param_schema:

        with st.expander(
            f"🛠️ Hyperparameters for {selected_model_name}",
            expanded=False,
        ):

            pcols = st.columns(
                min(2, len(param_schema))
            )

            for i, (
                param_name,
                schema,
            ) in enumerate(param_schema.items()):

                with pcols[i % len(pcols)]:

                    if schema["type"] == "int":

                        configured_params[param_name] = (
                            st.slider(
                                param_name,
                                min_value=schema["min"],
                                max_value=schema["max"],
                                value=schema["default"],
                            )
                        )

                    elif schema["type"] == "float":

                        configured_params[param_name] = (
                            st.slider(
                                param_name,
                                min_value=float(schema["min"]),
                                max_value=float(schema["max"]),
                                value=float(schema["default"]),
                                step=0.1,
                            )
                        )

    st.divider()

    # --------------------------------------------------
    # Section 9: Model Training and Evaluation
    # --------------------------------------------------

    st.subheader(
        "🚀 Model Training and Evaluation"
    )

    if st.button(
        "Train and Evaluate Model",
        type="primary",
    ):

        try:

            with st.spinner(
                "Processing data, training model, "
                "and evaluating performance..."
            ):

                # ------------------------------------------
                # Step 1: Split Dataset
                # ------------------------------------------

                X_train, X_test, y_train, y_test = (
                    PreprocessingService.split_data(
                        df=df,
                        feature_columns=feature_columns,
                        target_column=target_column,
                        test_size=test_size,
                        stratify=is_classification,
                    )
                )

                # ------------------------------------------
                # Step 2: Build Preprocessor
                # Fit only on training data to prevent leakage.
                # ------------------------------------------

                preprocessor = (
                    PreprocessingService.create_preprocessor(
                        X_train=X_train,
                        numeric_strategy=missing_strategy,
                        scale_features=scale_features,
                    )
                )

                X_train_proc, X_test_proc = (
                    PreprocessingService.fit_and_transform(
                        preprocessor=preprocessor,
                        X_train=X_train,
                        X_test=X_test,
                    )
                )

                # ------------------------------------------
                # Step 3: Initialize Training Status
                # ------------------------------------------

                status_box = st.status(
                    f"Training workflow for "
                    f"'{selected_model_name}'...",
                    expanded=True,
                )

                tracker = TrainingStatusTracker(
                    callback=lambda state, pct, ev: (
                        status_box.write(
                            f"[{state}] "
                            f"{ev.get('message', '')}"
                        )
                    )
                )

                tracker.update(
                    state=TrainingStatus.TRAINING,
                    progress_pct=0.1,
                    message="Step 3/5: Training model...",
                )

                # ------------------------------------------
                # Step 4: Train Model
                # ------------------------------------------

                train_result = ModelService.train_model(
                    model_name=selected_model_name,
                    X_train=X_train_proc,
                    y_train=y_train,
                    params=configured_params,
                    status_tracker=tracker,
                )

                training_time = train_result[
                    "training_time_seconds"
                ]

                model = train_result["model"]

                status_box.update(
                    label=(
                        f"Training complete "
                        f"({training_time:.3f}s)"
                    ),
                    state="complete",
                    expanded=False,
                )

                st.session_state.audit_service.log_event(
                    event_type="MODEL_TRAINING_SUCCESS",
                    username=user["username"],
                    details={
                        "model": selected_model_name,
                        "task_type": task_type,
                        "training_time_seconds": training_time,
                    },
                )

                # ------------------------------------------
                # Step 5: Generate Predictions
                # ------------------------------------------

                y_pred = model.predict(X_test_proc)

                st.success(
                    f"Model '{selected_model_name}' "
                    f"trained successfully in "
                    f"{training_time:.3f} seconds!"
                )

                # ------------------------------------------
                # Step 6: Evaluate Classification
                # ------------------------------------------

                if is_classification:

                    eval_results = (
                        EvaluationService.evaluate_classification(
                            y_true=y_test,
                            y_pred=y_pred,
                        )
                    )

                    m1, m2 = st.columns(2)
                    m3, m4 = st.columns(2)

                    m1.metric(
                        "Accuracy",
                        f"{eval_results['accuracy']:.4f}",
                    )

                    m2.metric(
                        "Precision",
                        f"{eval_results['precision']:.4f}",
                    )

                    m3.metric(
                        "Recall",
                        f"{eval_results['recall']:.4f}",
                    )

                    m4.metric(
                        "F1-Score",
                        f"{eval_results['f1_score']:.4f}",
                    )

                    st.markdown("#### Confusion Matrix")

                    cm_df = pd.DataFrame(
                        eval_results["confusion_matrix"],
                        index=[
                            f"Actual {label}"
                            for label in eval_results["labels"]
                        ],
                        columns=[
                            f"Pred {label}"
                            for label in eval_results["labels"]
                        ],
                    )

                    st.dataframe(
                        cm_df,
                        use_container_width=True,
                    )

                # ------------------------------------------
                # Step 7: Evaluate Regression
                # ------------------------------------------

                else:

                    eval_results = (
                        EvaluationService.evaluate_regression(
                            y_true=y_test,
                            y_pred=y_pred,
                        )
                    )

                    m1, m2 = st.columns(2)
                    m3, m4 = st.columns(2)

                    m1.metric(
                        "MAE",
                        f"{eval_results['mean_absolute_error']:.4f}",
                    )

                    m2.metric(
                        "MSE",
                        f"{eval_results['mean_squared_error']:.4f}",
                    )

                    m3.metric(
                        "RMSE",
                        f"{eval_results['root_mean_squared_error']:.4f}",
                    )

                    m4.metric(
                        "R² Score",
                        f"{eval_results['r2_score']:.4f}",
                    )

                # ------------------------------------------
                # Step 8: Record Evaluation Audit Event
                # ------------------------------------------

                st.session_state.audit_service.log_event(
                    event_type="MODEL_EVALUATION_SUCCESS",
                    username=user["username"],
                    details={
                        "model": selected_model_name,
                        "task_type": task_type,
                    },
                )

                # ------------------------------------------
                # Display Predictions
                # ------------------------------------------

                st.markdown(
                    "#### Sample Predictions vs Actual Values"
                )

                results_df = pd.DataFrame({
                    "Actual": y_test.values[:10],
                    "Predicted": y_pred[:10],
                })

                st.dataframe(
                    results_df,
                    use_container_width=True,
                )

                # ------------------------------------------
                # Step 9: Save Experiment
                # ML-F-016 / TC-HIST-01
                # ------------------------------------------

                storage = ExperimentStorageService()

                exp_config = {
                    "model_name": selected_model_name,
                    "task_type": task_type,
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "feature_count": len(feature_columns),
                    "split_ratio": test_size,
                    "missing_strategy": missing_strategy,
                    "scale_features": scale_features,
                    "parameters": configured_params,
                    "training_time_seconds": training_time,
                }

                exp_id = storage.save_experiment(
                    config=exp_config,
                    metrics=eval_results,
                    username=user["username"],
                )

                st.info(
                    "💾 Experiment configuration and "
                    f"evaluation results saved: `{exp_id}`"
                )

                # ------------------------------------------
                # Step 10: Mark Workflow Complete
                # ------------------------------------------

                tracker.update(
                    state=TrainingStatus.COMPLETED,
                    progress_pct=1.0,
                    message=(
                        "Model training and evaluation completed."
                    ),
                )

                st.success(
                    f"Model '{selected_model_name}' "
                    "trained and evaluated successfully in "
                    f"{training_time:.3f} seconds."
                )

        except (
            PreprocessingError,
            DatasetValidationError,
            ModelTrainingError,
        ) as e:

            st.error(
                f"Unable to evaluate this dataset: {e}"
            )

            st.info(
                "Check your target and feature selections, "
                "missing values, and dataset format, "
                "then try again."
            )

        except Exception as e:

            st.error(
                "Model evaluation failed unexpectedly."
            )

            st.exception(e)

    # --------------------------------------------------
    # Section 10: Experiment History
    # ML-F-016
    # --------------------------------------------------

    st.divider()

    st.subheader("📜 Experiment History")

    storage = ExperimentStorageService()

    history = storage.list_experiments(
        limit=10,
        username=user["username"],
        role=user["role"],
    )

    if history:

        with st.expander(
            f"View Previous Experiment Runs "
            f"({len(history)} stored)",
            expanded=False,
        ):

            history_df = pd.DataFrame(history)

            st.dataframe(
                history_df,
                use_container_width=True,
            )

    else:

        st.caption(
            "No previous experiments stored yet."
        )


# --------------------------------------------------
# Application Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()
