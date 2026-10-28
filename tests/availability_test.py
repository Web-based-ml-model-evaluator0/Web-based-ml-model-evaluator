#!/usr/bin/env python3
"""Measure HTTP availability for ML-NF-003 (standard library only).

Example:
  python tests/availability_test.py --url http://127.0.0.1:8501 \
      --duration-seconds 300 --interval-seconds 10 --target-percent 99.0

The target must be taken from the approved SRS. An optional maintenance CSV
must have columns start_utc,end_utc with ISO-8601 timestamps (UTC recommended).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone: {value}")
    return parsed.astimezone(timezone.utc)


def read_maintenance(path: str | None) -> list[tuple[datetime, datetime]]:
    if not path:
        return []
    windows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = parse_time(row["start_utc"])
            end = parse_time(row["end_utc"])
            if end < start:
                raise ValueError("Maintenance end_utc must not precede start_utc")
            windows.append((start, end))
    return windows


def in_maintenance(at: datetime, windows: list[tuple[datetime, datetime]]) -> bool:
    return any(start <= at <= end for start, end in windows)


def probe(url: str, timeout: float) -> tuple[bool, int | None, str, float]:
    started = time.perf_counter()
    status = None
    error = ""
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ML-NF-003-availability-check/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return status is not None and 200 <= status < 400, status, error, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--duration-seconds", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--target-percent", type=float, required=True,
                        help="Availability target from the approved SRS")
    parser.add_argument("--maintenance-csv", default=None,
                        help="Optional CSV with start_utc,end_utc columns")
    parser.add_argument("--out-dir", default="availability-results")
    args = parser.parse_args()
    if args.duration_seconds < 1:
        parser.error("--duration-seconds must be at least 1")
    if args.interval_seconds <= 0 or args.timeout <= 0:
        parser.error("--interval-seconds and --timeout must be positive")
    if not 0 < args.target_percent <= 100:
        parser.error("--target-percent must be greater than 0 and at most 100")

    try:
        maintenance = read_maintenance(args.maintenance_csv)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(f"Invalid maintenance CSV: {exc}")

    started = time.monotonic()
    deadline = started + args.duration_seconds
    rows = []
    while True:
        now = datetime.now(timezone.utc)
        excluded = in_maintenance(now, maintenance)
        ok, status, error, elapsed_ms = probe(args.url, args.timeout)
        rows.append({
            "timestamp_utc": now.isoformat(),
            "url": args.url,
            "status_code": status if status is not None else "",
            "response_time_ms": elapsed_ms,
            "available": ok,
            "planned_maintenance_excluded": excluded,
            "error": error,
        })
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.interval_seconds, remaining))

    eligible = [row for row in rows if not row["planned_maintenance_excluded"]]
    successes = sum(bool(row["available"]) for row in eligible)
    failures = len(eligible) - successes
    measured = (successes / len(eligible) * 100) if eligible else None
    decision = (
        "PASS" if measured is not None and measured >= args.target_percent else "FAIL"
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "tc-ops-01-results.csv"
    json_path = out / "tc-ops-01-summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "requirement_id": "ML-NF-003",
        "test_case_id": "TC-OPS-01",
        "started_at_utc": rows[0]["timestamp_utc"],
        "url": args.url,
        "duration_seconds_requested": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "planned_maintenance_windows": len(maintenance),
        "total_probes": len(rows),
        "excluded_maintenance_probes": len(rows) - len(eligible),
        "eligible_probes": len(eligible),
        "successful_probes": successes,
        "failed_probes": failures,
        "availability_percent": round(measured, 4) if measured is not None else None,
        "target_percent": args.target_percent,
        "result": decision,
        "scope_note": (
            "HTTP endpoint probe availability over this test window; not a "
            "claim about long-term production uptime or browser-level usability."
        ),
        "probes": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=2))
    print(f"Detailed CSV: {csv_path}\nSummary JSON: {json_path}")
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
