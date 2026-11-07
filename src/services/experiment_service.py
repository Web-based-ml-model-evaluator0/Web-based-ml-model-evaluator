"""
Experiment Storage Service module (ML-F-016, TC-HIST-01).
Provides persistence and retrieval of experiment configurations,
hyperparameters, training execution metadata, and evaluation results.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import sqlite3
import json
import uuid
import datetime


class ExperimentStorageService:
    """
    Manages SQLite-backed persistent storage of ML experiments (ML-F-016 / TC-HIST-01).
    Decoupled from presentation and training layers according to SAD 3.4.
    """

    DEFAULT_DB_PATH = Path("experiments") / "experiments.db"

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path is None:
            self.db_path = self.DEFAULT_DB_PATH
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        if self.db_path != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    feature_count INTEGER,
                    split_ratio REAL,
                    training_time_seconds REAL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_experiment(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        experiment_id: Optional[str] = None,
        status: str = "COMPLETED",
    ) -> str:
        """
        Stores experiment configuration and evaluation metrics.
        Returns the unique experiment_id.
        """
        if not experiment_id:
            experiment_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"

        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        model_name = str(config.get("model_name", "Unknown Model"))
        task_type = str(config.get("task_type", "classification"))
        target_column = str(config.get("target_column", ""))
        feature_count = int(config.get("feature_count", 0))
        split_ratio = float(config.get("split_ratio", 0.20))
        training_time = float(metrics.get("training_time_seconds", config.get("training_time_seconds", 0.0)))

        config_json = json.dumps(config, default=str)
        metrics_json = json.dumps(metrics, default=str)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, created_at, model_name, task_type,
                    target_column, feature_count, split_ratio,
                    training_time_seconds, config_json, metrics_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id, created_at, model_name, task_type,
                    target_column, feature_count, split_ratio,
                    training_time, config_json, metrics_json, status
                ),
            )
            conn.commit()

        return experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single experiment and its evaluation results by experiment_id.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "experiment_id": row["experiment_id"],
                "created_at": row["created_at"],
                "model_name": row["model_name"],
                "task_type": row["task_type"],
                "target_column": row["target_column"],
                "feature_count": row["feature_count"],
                "split_ratio": row["split_ratio"],
                "training_time_seconds": row["training_time_seconds"],
                "config": json.loads(row["config_json"]),
                "metrics": json.loads(row["metrics_json"]),
                "status": row["status"],
            }

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Returns a list of stored experiments ordered chronologically descending.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT experiment_id, created_at, model_name, task_type,
                       target_column, feature_count, split_ratio,
                       training_time_seconds, metrics_json, status
                FROM experiments
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                metrics_data = json.loads(r["metrics_json"])
                results.append({
                    "experiment_id": r["experiment_id"],
                    "created_at": r["created_at"],
                    "model_name": r["model_name"],
                    "task_type": r["task_type"],
                    "target_column": r["target_column"],
                    "feature_count": r["feature_count"],
                    "split_ratio": r["split_ratio"],
                    "training_time_seconds": r["training_time_seconds"],
                    "status": r["status"],
                    "accuracy": metrics_data.get("accuracy"),
                    "f1_score": metrics_data.get("f1_score"),
                    "r2_score": metrics_data.get("r2_score"),
                })
            return results

    def delete_experiment(self, experiment_id: str) -> bool:
        """Deletes an experiment from storage. Returns True if found and removed."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def count_experiments(self) -> int:
        """Returns the total number of stored experiments."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
            return int(cursor.fetchone()[0])
