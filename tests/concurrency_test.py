#!/usr/bin/env python3
"""Concurrent HTTP smoke test for ML-NF-002 (standard library only).

Example:
  python tests/concurrency_test.py --url http://127.0.0.1:8501 --users 5
Use --users with the baseline required by the approved SRS.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request_once(url: str, timeout: float, worker: int) -> dict:
    started = time.perf_counter()
    status = None
    error = ""
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ML-NF-002-concurrency-check/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "worker": worker,
        "url": url,
        "status_code": status if status is not None else "",
        "response_time_ms": elapsed_ms,
        "success": status is not None and 200 <= status < 400,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--users", type=int, required=True,
                        help="Concurrent users to simulate; use the SRS baseline")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out-dir", default="performance-results")
    args = parser.parse_args()
    if args.users < 1:
        parser.error("--users must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    started_utc = datetime.now(timezone.utc).isoformat()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = [
            pool.submit(request_once, args.url, args.timeout, worker)
            for worker in range(1, args.users + 1)
        ]
        results = [future.result() for future in futures]

    successful = sum(bool(item["success"]) for item in results)
    failed = len(results) - successful
    decision = "PASS" if failed == 0 else "FAIL"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "tc-perf-02-results.csv"
    json_path = out / "tc-perf-02-summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "requirement_id": "ML-NF-002",
        "test_case_id": "TC-PERF-02",
        "started_at_utc": started_utc,
        "url": args.url,
        "requested_concurrent_users": args.users,
        "successful_requests": successful,
        "failed_requests": failed,
        "result": decision,
        "scope_note": (
            "Concurrent HTTP GET smoke test only. It does not simulate browser "
            "sessions, authenticated workflows, or model-training workloads."
        ),
        "requests": results,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}, indent=2))
    print(f"Detailed CSV: {csv_path}\nSummary JSON: {json_path}")
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
