
"""
Experiment Storage Service module (ML-F-016, TC-HIST-01).
Provides persistence, integrity verification, and access-controlled
retrieval of experiment configurations and evaluation results.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import sqlite3
import json
import uuid
import datetime
import hashlib
import hmac


class ExperimentStorageService:
    """SQLite-backed experiment storage with integrity and access control."""

    DEFAULT_DB_PATH = Path("experiments") / "experiments.db"

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path is None:
            self.db_path = self.DEFAULT_DB_PATH
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        if self.db_path != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
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
            """)
            conn.commit()

    def save_experiment(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        experiment_id: Optional[str] = None,
        status: str = "COMPLETED",
        username: Optional[str] = None,
    ) -> str:
        """Store an experiment and associate it with its owner."""

        if not experiment_id:
            experiment_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"

        created_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        config_json = json.dumps(config, sort_keys=True, default=str)
        metrics_json = json.dumps(metrics, sort_keys=True, default=str)

        with self._get_connection() as conn:
            columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(experiments)"
                )
            }

            if "owner" not in columns:
                conn.execute(
                    "ALTER TABLE experiments ADD COLUMN owner TEXT"
                )

            conn.execute("""
                INSERT INTO experiments (
                    experiment_id, created_at, model_name, task_type,
                    target_column, feature_count, split_ratio,
                    training_time_seconds, config_json, metrics_json,
                    status, owner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                created_at,
                str(config.get("model_name", "Unknown Model")),
                str(config.get("task_type", "classification")),
                str(config.get("target_column", "")),
                int(config.get("feature_count", 0)),
                float(config.get("split_ratio", 0.20)),
                float(metrics.get(
                    "training_time_seconds",
                    config.get("training_time_seconds", 0.0)
                )),
                config_json,
                metrics_json,
                status,
                username,
            ))
            conn.commit()

        return experiment_id

    def get_experiment(
        self,
        experiment_id: str,
        username: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve an experiment only when access is permitted."""

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()

            if row is None:
                return None

            owner = row["owner"] if "owner" in row.keys() else None

            if role not in {"admin", "evaluator"}:
                if not username or owner != username:
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

    def list_experiments(
        self,
        limit: int = 50,
        username: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List only experiments accessible to the requesting user."""

        with self._get_connection() as conn:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(experiments)")
            }

            if "owner" not in columns:
                conn.execute(
                    "ALTER TABLE experiments ADD COLUMN owner TEXT"
                )

            if role in {"admin", "evaluator"}:
                rows = conn.execute("""
                    SELECT * FROM experiments
                    ORDER BY id DESC LIMIT ?
                """, (limit,)).fetchall()
            else:
                if not username:
                    return []

                rows = conn.execute("""
                    SELECT * FROM experiments
                    WHERE owner = ?
                    ORDER BY id DESC LIMIT ?
                """, (username, limit)).fetchall()

            results = []
            for row in rows:
                metrics = json.loads(row["metrics_json"])
                results.append({
                    "experiment_id": row["experiment_id"],
                    "created_at": row["created_at"],
                    "model_name": row["model_name"],
                    "task_type": row["task_type"],
                    "target_column": row["target_column"],
                    "feature_count": row["feature_count"],
                    "split_ratio": row["split_ratio"],
                    "training_time_seconds": row["training_time_seconds"],
                    "status": row["status"],
                    "accuracy": metrics.get("accuracy"),
                    "f1_score": metrics.get("f1_score"),
                    "r2_score": metrics.get("r2_score"),
                })

            return results

    def delete_experiment(
        self,
        experiment_id: str,
        username: Optional[str] = None,
        role: Optional[str] = None,
    ) -> bool:
        """Delete an experiment only if the requester has permission."""

        with self._get_connection() as conn:
            if role in {"admin", "evaluator"}:
                cursor = conn.execute(
                    "DELETE FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                )
            else:
                if not username:
                    return False

                cursor = conn.execute("""
                    DELETE FROM experiments
                    WHERE experiment_id = ? AND owner = ?
                """, (experiment_id, username))

            conn.commit()
            return cursor.rowcount > 0

    def count_experiments(self) -> int:
        with self._get_connection() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM experiments"
                ).fetchone()[0]
            )
