"""
Utility script to inspect the SQLite ML experiments database.
Usage:
    python scripts/show_db.py
    python scripts/show_db.py --detail
"""

import sqlite3
import json
import sys
from pathlib import Path

# Safe stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_PATH = Path("experiments") / "experiments.db"

def inspect_db(show_detail: bool = False):
    if not DB_PATH.exists():
        print(f"[!] Database file not found at: {DB_PATH.resolve()}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n" + "=" * 70)
    print("      WEB-BASED ML MODEL EVALUATOR - DATABASE INSPECTOR")
    print(f"      Database File: {DB_PATH.resolve()}")
    print("=" * 70)

    # 1. Table Schema
    print("\n[1] TABLE SCHEMA: 'experiments'")
    print("-" * 55)
    print(f"{'Column Name':<26} {'Data Type':<12} {'Nullable':<10}")
    print("-" * 55)
    columns = cursor.execute("PRAGMA table_info(experiments)").fetchall()
    for col in columns:
        col_name = col["name"]
        col_type = col["type"]
        not_null = "NOT NULL" if col["notnull"] else "NULL"
        pk = " (PRIMARY KEY)" if col["pk"] else ""
        print(f"{col_name:<26} {col_type:<12} {not_null}{pk}")

    # 2. Record Summary
    rows = cursor.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    print("\n" + "-" * 70)
    print(f"[2] STORED EXPERIMENTS ({len(rows)} Total Records)")
    print("-" * 70)

    if not rows:
        print("  (No experiment records found yet. Train a model in the app to save one!)")
        return

    print(f"{'ID':<14} {'Model':<20} {'Target':<12} {'Split':<7} {'Time(s)':<8} {'Status':<10}")
    print("-" * 70)
    for r in rows:
        exp_id = r["experiment_id"]
        model = r["model_name"][:18]
        target = r["target_column"][:10]
        split = f"{r['split_ratio']:.2f}"
        t = f"{r['training_time_seconds']:.2f}"
        status = r["status"]
        print(f"{exp_id:<14} {model:<20} {target:<12} {split:<7} {t:<8} {status:<10}")

    # 3. Metrics Details
    print("\n" + "-" * 70)
    print("[3] DETAILED METRICS FOR LATEST EXPERIMENT")
    print("-" * 70)
    latest = rows[0]
    metrics = json.loads(latest["metrics_json"])
    print(f"Experiment ID: {latest['experiment_id']}")
    print(f"Timestamp    : {latest['created_at']}")
    print(f"Model Name   : {latest['model_name']}")
    if "accuracy" in metrics:
        print(f"Accuracy     : {metrics.get('accuracy', 0):.4f}")
        print(f"Precision    : {metrics.get('precision', 0):.4f}")
        print(f"Recall       : {metrics.get('recall', 0):.4f}")
        print(f"F1-Score     : {metrics.get('f1_score', 0):.4f}")
    if "confusion_matrix" in metrics:
        print(f"Confusion Matrix: {metrics.get('confusion_matrix')}")

    if show_detail:
        print("\n[4] HYPERPARAMETERS & CONFIG (JSON):")
        print(json.dumps(json.loads(latest["config_json"]), indent=2))
        print("\n[5] FULL CLASSIFICATION REPORT:")
        print(json.dumps(metrics.get("classification_report", {}), indent=2))

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    detail_flag = "--detail" in sys.argv or "-d" in sys.argv
    inspect_db(show_detail=detail_flag)
