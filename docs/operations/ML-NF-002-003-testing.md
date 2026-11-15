# ML-NF-002 and ML-NF-003 — Test Procedures

These procedures use Python's standard library and target the Streamlit HTTP page. They do not require third-party load-testing packages.

## Before running

1. Check the approved SRS and record the required concurrent-user baseline for ML-NF-002.
2. Check the approved SRS and record the availability percentage and measurement window for ML-NF-003.
3. Do not substitute guessed targets. The scripts require the concurrency count, availability target, and monitoring duration as command-line arguments.
4. Install application dependencies and start the app from the repository root:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Run tests from a second terminal. Replace example values with the approved SRS values.

## ML-NF-002 — Concurrent User Scalability (TC-PERF-02)

Example for a five-user test only; use the SRS baseline if different:

```bash
python tests/concurrency_test.py --url http://127.0.0.1:8501 --users 5
```

The script starts the requested number of concurrent HTTP GET requests and records status, latency, errors, and the pass/fail result in:

- `performance-results/tc-perf-02-results.csv`
- `performance-results/tc-perf-02-summary.json`

Pass condition: all requests return an HTTP status from 200 through 399 with no request errors. A passing HTTP smoke test does not prove that simultaneous browser sessions or model training are safe; those need workflow-level tests.

## ML-NF-003 — System Availability (TC-OPS-01)

Example only; replace target and duration with approved SRS values:

```bash
python tests/availability_test.py --url http://127.0.0.1:8501 --duration-seconds 300 --interval-seconds 10 --target-percent 99.0
```

The script counts successful HTTP probes divided by eligible probes. If planned maintenance needs to be excluded, create a CSV such as:

```csv
start_utc,end_utc
2026-10-28T11:00:00+00:00,2026-10-28T11:15:00+00:00
```

Pass `--maintenance-csv planned-maintenance.csv` to exclude probes whose timestamps fall within those windows. Timestamps must include a timezone. Keep the maintenance CSV with the test evidence.

Output files:

- `availability-results/tc-ops-01-results.csv`
- `availability-results/tc-ops-01-summary.json`

Pass condition: measured availability meets or exceeds the approved SRS target after planned-maintenance probes are excluded. A short local test is only a smoke test; it cannot establish long-term production availability.

## Evidence and acceptance checklist

- [ ] Record the exact SRS target and test environment.
- [ ] Run TC-PERF-02 at the required concurrency baseline.
- [ ] Confirm no critical failures and review latency/errors.
- [ ] Save CSV and JSON results and attach them to Issue #22.
- [ ] Run TC-OPS-01 for the approved measurement window.
- [ ] Exclude only documented planned-maintenance windows.
- [ ] Save CSV and JSON results and attach them to Issue #23.
- [ ] Have the team review the evidence before marking either issue Done.

## Limitations

Both tests probe the Streamlit page's HTTP response. The current app is a Streamlit application; these scripts do not claim that a separate REST API exists. Availability measured over a short test window is not equivalent to monthly or yearly production uptime. Results are generated only when the scripts are actually run against the application.
