"""
Model Service module (ML-NF-008, ML-F-008, ML-F-009, ML-F-010).
Provides a pluggable model registry and training orchestration that allows new
machine learning algorithms to be registered without altering dataset or evaluation workflows.
"""

from typing import Any, Callable, Dict, List, Optional
import time
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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
    ) -> Dict[str, Any]:
        """
        Trains the specified model on training data and records execution metrics.
        """
        model = cls.create_model_instance(model_name, params)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        return {
            "model": model,
            "model_name": model_name,
            "training_time_seconds": training_time,
            "params": params or {},
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
