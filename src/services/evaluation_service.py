"""
Evaluation Service module (ML-F-012, ML-F-013, ML-F-014).
Computes performance metrics, confusion matrices, and evaluation summaries
for both classification and regression tasks.
"""

from typing import Any, Dict, List, Optional
import numpy as np
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


class EvaluationService:
    """
    Evaluates trained models against ground-truth test labels.
    Decoupled from model fitting and presentation according to SAD 3.4.
    """

    @classmethod
    def evaluate_classification(
        cls,
        y_true: Any,
        y_pred: Any,
        labels: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates classification evaluation metrics: Accuracy, Precision, Recall, F1,
        and Confusion Matrix.
        """
        unique_labels = labels if labels is not None else np.unique(y_true).tolist()
        is_binary = len(unique_labels) == 2
        average_mode = "binary" if is_binary else "weighted"
        pos_label = unique_labels[1] if is_binary else 1

        # Calculate metrics
        acc = float(accuracy_score(y_true, y_pred))
        if is_binary:
            prec = float(precision_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0))
            rec = float(recall_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0))
        else:
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()
        report = classification_report(y_true, y_pred, labels=unique_labels, output_dict=True, zero_division=0)

        return {
            "task_type": "classification",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "labels": [str(l) for l in unique_labels],
            "confusion_matrix": cm,
            "classification_report": report,
        }

    @classmethod
    def evaluate_regression(
        cls,
        y_true: Any,
        y_pred: Any,
    ) -> Dict[str, Any]:
        """
        Calculates regression evaluation metrics: MAE, MSE, RMSE, and R2 score.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        return {
            "task_type": "regression",
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "r2_score": r2,
        }
