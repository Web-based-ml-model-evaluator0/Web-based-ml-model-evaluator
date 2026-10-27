#!/usr/bin/env python3
"""Measure response time for a non-training page of the Streamlit app.

Example:
  streamlit run app.py
  python tests/performance_test.py --url http://127.0.0.1:8501 --runs 10

The SRS target is 3000 ms at the 95th percentile for non-training page/API responses.
"""
from __future__ import annotations
import argparse, csv, json, statistics, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((p / 100) * len(ordered) + 0.999999) - 1))
    return ordered[index]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501", help="Page URL to measure")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--target-ms", type=float, default=3000.0, help="SRS target in milliseconds (default: 3000)")
    parser.add_argument("--out-dir", default="performance-results")
    args = parser.parse_args()
    if args.runs < 1: parser.error("--runs must be at least 1")
    if args.timeout <= 0: parser.error("--timeout must be positive")
    if args.target_ms <= 0: parser.error("--target-ms must be positive")
    results = []
    for number in range(1, args.runs + 1):
        request = urllib.request.Request(args.url, headers={"User-Agent": "ML-NF-001-performance-check/1.0"})
        started = time.perf_counter(); status = None; error = ""
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response.read(); status = response.status
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = str(exc)
        elapsed = (time.perf_counter() - started) * 1000
        results.append({"run": number, "url": args.url, "status_code": status if status is not None else "",
                        "response_time_ms": round(elapsed, 3), "success": status is not None and 200 <= status < 400,
                        "error": error})
    successful = [r["response_time_ms"] for r in results if r["success"]]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "tc-perf-01-results.csv"; json_path = out / "tc-perf-01-summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys())); writer.writeheader(); writer.writerows(results)
    p95 = percentile(successful, 95) if successful else None
    decision = ("PASS" if successful and len(successful) == args.runs and p95 <= args.target_ms
                else "FAIL")
    summary = {"requirement_id": "ML-NF-001", "test_case_id": "TC-PERF-01",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "url": args.url, "runs_requested": args.runs,
        "successful_runs": len(successful), "failed_runs": args.runs - len(successful),
        "average_response_time_ms": round(statistics.mean(successful), 3) if successful else None,
        "median_response_time_ms": round(statistics.median(successful), 3) if successful else None,
        "p95_response_time_ms": p95, "srs_target_ms": args.target_ms, "performance_target_result": decision,
        "scope_note": "Measures selected page HTTP response only; not model training or unimplemented API endpoints.",
        "requests": results}
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "requests"}, indent=2))
    print(f"Detailed CSV: {csv_path}\nSummary JSON: {json_path}")
    return 0 if len(successful) == args.runs else 1

if __name__ == "__main__": raise SystemExit(main())
