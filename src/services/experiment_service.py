
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
import hashlib


class ExperimentIntegrityError(Exception):
    """Raised when stored experiment data has been modified."""


class ExperimentStorageService:
    """
    Manages SQLite-backed persistent storage of ML experiments.
    Includes integrity verification for stored configurations and results.
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

    @staticmethod
    def _serialize(data: Dict[str, Any]) -> str:
        """Serialize data deterministically for integrity verification."""
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _calculate_hash(data: str) -> str:
        """Calculate SHA-256 hash of serialized experiment data."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

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

            # Add integrity columns to existing databases if missing.
            columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(experiments)"
                ).fetchall()
            }

            if "config_hash" not in columns:
                conn.execute(
                    "ALTER TABLE experiments ADD COLUMN config_hash TEXT"
                )

            if "metrics_hash" not in columns:
                conn.execute(
                    "ALTER TABLE experiments ADD COLUMN metrics_hash TEXT"
                )

            conn.commit()

    def save_experiment(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        experiment_id: Optional[str] = None,
        status: str = "COMPLETED",
    ) -> str:
        """Stores experiment configuration and metrics with integrity hashes."""

        if not experiment_id:
            experiment_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"

        created_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        model_name = str(config.get("model_name", "Unknown Model"))
        task_type = str(config.get("task_type", "classification"))
        target_column = str(config.get("target_column", ""))
        feature_count = int(config.get("feature_count", 0))
        split_ratio = float(config.get("split_ratio", 0.20))

        training_time = float(
            metrics.get(
                "training_time_seconds",
                config.get("training_time_seconds", 0.0),
            )
        )

        config_json = self._serialize(config)
        metrics_json = self._serialize(metrics)

        config_hash = self._calculate_hash(config_json)
        metrics_hash = self._calculate_hash(metrics_json)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, created_at, model_name, task_type,
                    target_column, feature_count, split_ratio,
                    training_time_seconds, config_json, metrics_json,
                    status, config_hash, metrics_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    created_at,
                    model_name,
                    task_type,
                    target_column,
                    feature_count,
                    split_ratio,
                    training_time,
                    config_json,
                    metrics_json,
                    status,
                    config_hash,
                    metrics_hash,
                ),
            )
            conn.commit()

        return experiment_id

    def get_experiment(
        self,
        experiment_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves an experiment and verifies stored data integrity."""

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()

            if not row:
                return None

            config_json = row["config_json"]
            metrics_json = row["metrics_json"]

            # Verify configuration integrity.
            if row["config_hash"] and (
                self._calculate_hash(config_json) != row["config_hash"]
            ):
                raise ExperimentIntegrityError(
                    f"Configuration integrity check failed for "
                    f"experiment {experiment_id}"
                )

            # Verify evaluation result integrity.
            if row["metrics_hash"] and (
                self._calculate_hash(metrics_json) != row["metrics_hash"]
            ):
                raise ExperimentIntegrityError(
                    f"Metrics integrity check failed for "
                    f"experiment {experiment_id}"
                )

            return {
                "experiment_id": row["experiment_id"],
                "created_at": row["created_at"],
                "model_name": row["model_name"],
                "task_type": row["task_type"],
                "target_column": row["target_column"],
                "feature_count": row["feature_count"],
                "split_ratio": row["split_ratio"],
                "training_time_seconds": row["training_time_seconds"],
                "config": json.loads(config_json),
                "metrics": json.loads(metrics_json),
                "status": row["status"],
            }

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns stored experiments after integrity verification."""

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT experiment_id FROM experiments
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

        results = []

        for row in rows:
            experiment = self.get_experiment(row["experiment_id"])

            if experiment is None:
                continue

            metrics_data = experiment["metrics"]

            results.append({
                "experiment_id": experiment["experiment_id"],
                "created_at": experiment["created_at"],
                "model_name": experiment["model_name"],
                "task_type": experiment["task_type"],
                "target_column": experiment["target_column"],
                "feature_count": experiment["feature_count"],
                "split_ratio": experiment["split_ratio"],
                "training_time_seconds": experiment[
                    "training_time_seconds"
                ],
                "status": experiment["status"],
                "accuracy": metrics_data.get("accuracy"),
                "f1_score": metrics_data.get("f1_score"),
                "r2_score": metrics_data.get("r2_score"),
            })

        return results

    def delete_experiment(self, experiment_id: str) -> bool:
        """Deletes an experiment. Returns True if it existed."""
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
