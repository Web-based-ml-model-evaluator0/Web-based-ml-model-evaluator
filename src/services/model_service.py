"""
Model Service module (ML-NF-008, ML-F-008, ML-F-009, ML-F-010, ML-F-011).
Provides a pluggable model registry, training orchestration, and progress tracking
that allows machine learning workflows to report execution status and handle errors safely.
"""

from typing import Any, Callable, Dict, List, Optional
import time
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ModelTrainingError(RuntimeError):
    """Raised when model training encounters an error or invalid input."""
    pass


class TrainingStatus:
    """Standard status states for ML model training lifecycle (ML-F-011)."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PREPROCESSING = "PREPROCESSING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingStatusTracker:
    """
    Tracks and records lifecycle progress and status transitions for training (ML-F-011 / TC-TRAIN-02).
    Provides feedback hooks for UI components and test harnesses.
    """

    def __init__(self, callback: Optional[Callable[[str, float, Dict[str, Any]], None]] = None) -> None:
        self.state: str = TrainingStatus.PENDING
        self.progress_pct: float = 0.0
        self.history: List[Dict[str, Any]] = []
        self.callback = callback
        self.error_message: Optional[str] = None

    def update(
        self,
        state: str,
        progress_pct: float,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Records a new state update and informs listeners."""
        self.state = state
        self.progress_pct = max(0.0, min(1.0, float(progress_pct)))
        event = {
            "state": state,
            "progress_pct": self.progress_pct,
            "message": message,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.history.append(event)
        if state == TrainingStatus.FAILED and message:
            self.error_message = message
        if self.callback:
            try:
                self.callback(state, self.progress_pct, event)
            except Exception:
                pass
        return event

    def is_completed(self) -> bool:
        """Returns True if the training task finished successfully."""
        return self.state == TrainingStatus.COMPLETED

    def is_failed(self) -> bool:
        """Returns True if the training task failed."""
        return self.state == TrainingStatus.FAILED


class ModelService:
    """
    Manages the catalog of supported models, parameter validation, and model fitting.
    Designed according to SAD 3.4 & 4.1 for modular extensibility (ML-NF-008).
    """

    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_model(
        cls,
        model_name: str,
        task_type: str,
        factory: Callable[..., BaseEstimator],
        default_params: Optional[Dict[str, Any]] = None,
        param_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Registers a new machine learning algorithm into the system dynamically.
        Enables adding new models without modifying the rest of the application.
        """
        cls._registry[model_name] = {
            "task_type": task_type,
            "factory": factory,
            "default_params": default_params or {},
            "param_schema": param_schema or {},
        }

    @classmethod
    def get_supported_models(cls, task_type: Optional[str] = None) -> List[str]:
        """
        Returns the list of available model names, optionally filtered by task type.
        """
        if not cls._registry:
            cls._initialize_defaults()

        if task_type:
            return [
                name for name, info in cls._registry.items()
                if info["task_type"].lower() == task_type.lower()
            ]
        return list(cls._registry.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Returns metadata and default parameters for a given model.
        """
        if not cls._registry:
            cls._initialize_defaults()

        if model_name not in cls._registry:
            raise ValueError(f"Model '{model_name}' is not registered in the system.")
        return cls._registry[model_name]

    @classmethod
    def create_model_instance(cls, model_name: str, params: Optional[Dict[str, Any]] = None) -> BaseEstimator:
        """
        Instantiates a scikit-learn model with the provided parameters.
        """
        info = cls.get_model_info(model_name)
        combined_params = {**info["default_params"], **(params or {})}
        return info["factory"](**combined_params)

    @classmethod
    def train_model(
        cls,
        model_name: str,
        X_train: Any,
        y_train: Any,
        params: Optional[Dict[str, Any]] = None,
        status_tracker: Optional[TrainingStatusTracker] = None,
    ) -> Dict[str, Any]:
        """
        Trains the specified model on training data, tracks progress, and records execution metrics.
        Validates inputs and handles training failures safely (ML-F-010 / TC-TRAIN-01, ML-F-011 / TC-TRAIN-02).
        """
        if status_tracker:
            status_tracker.update(
                TrainingStatus.STARTED,
                0.10,
                f"Validating training configuration for '{model_name}'...",
            )

        if X_train is None or len(X_train) == 0:
            msg = "Training dataset X_train cannot be empty or None."
            if status_tracker:
                status_tracker.update(TrainingStatus.FAILED, 0.10, msg)
            raise ModelTrainingError(msg)

        if y_train is None or len(y_train) == 0:
            msg = "Target training labels y_train cannot be empty or None."
            if status_tracker:
                status_tracker.update(TrainingStatus.FAILED, 0.10, msg)
            raise ModelTrainingError(msg)

        if len(X_train) != len(y_train):
            msg = (
                f"Mismatched sample counts: X_train has {len(X_train)} samples, "
                f"but y_train has {len(y_train)} labels."
            )
            if status_tracker:
                status_tracker.update(TrainingStatus.FAILED, 0.10, msg)
            raise ModelTrainingError(msg)

        try:
            model = cls.create_model_instance(model_name, params)
        except Exception as e:
            msg = f"Failed to instantiate model '{model_name}': {str(e)}"
            if status_tracker:
                status_tracker.update(TrainingStatus.FAILED, 0.20, msg)
            raise ModelTrainingError(msg) from e

        if status_tracker:
            status_tracker.update(
                TrainingStatus.TRAINING,
                0.50,
                f"Fitting model '{model_name}' on {len(X_train)} training samples...",
            )

        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
        except Exception as e:
            msg = f"Training failed for '{model_name}': {str(e)}"
            if status_tracker:
                status_tracker.update(TrainingStatus.FAILED, 0.70, msg)
            raise ModelTrainingError(msg) from e

        if status_tracker:
            status_tracker.update(
                TrainingStatus.COMPLETED,
                1.00,
                f"Training completed successfully in {training_time:.3f} seconds.",
                metadata={"training_time": training_time},
            )

        feature_count = None
        if hasattr(X_train, "shape") and len(X_train.shape) > 1:
            feature_count = int(X_train.shape[1])
        elif hasattr(X_train, "columns"):
            feature_count = len(X_train.columns)

        return {
            "model": model,
            "model_name": model_name,
            "training_time_seconds": training_time,
            "params": params or {},
            "sample_count": len(X_train),
            "feature_count": feature_count,
            "status": TrainingStatus.COMPLETED,
        }

    @classmethod
    def _initialize_defaults(cls) -> None:
        """
        Initializes baseline models supported out-of-the-box in Sprint 1.
        """
        # Classification models
        cls.register_model(
            model_name="Logistic Regression",
            task_type="classification",
            factory=LogisticRegression,
            default_params={"max_iter": 1000, "random_state": 42},
            param_schema={"C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0}},
        )
        cls.register_model(
            model_name="Random Forest Classifier",
            task_type="classification",
            factory=RandomForestClassifier,
            default_params={"n_estimators": 100, "random_state": 42},
            param_schema={
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
                "max_depth": {"type": "int", "default": 10, "min": 1, "max": 50},
            },
        )
        cls.register_model(
            model_name="Decision Tree Classifier",
            task_type="classification",
            factory=DecisionTreeClassifier,
            default_params={"random_state": 42},
            param_schema={"max_depth": {"type": "int", "default": 5, "min": 1, "max": 30}},
        )

        # Regression models
        cls.register_model(
            model_name="Linear Regression",
            task_type="regression",
            factory=LinearRegression,
            default_params={},
            param_schema={},
        )
        cls.register_model(
            model_name="Random Forest Regressor",
            task_type="regression",
            factory=RandomForestRegressor,
            default_params={"n_estimators": 100, "random_state": 42},
            param_schema={
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
                "max_depth": {"type": "int", "default": 10, "min": 1, "max": 50},
            },
        )
        cls.register_model(
            model_name="Decision Tree Regressor",
            task_type="regression",
            factory=DecisionTreeRegressor,
            default_params={"random_state": 42},
            param_schema={"max_depth": {"type": "int", "default": 5, "min": 1, "max": 30}},
        )


# Initialize default models on module load
ModelService._initialize_defaults()
