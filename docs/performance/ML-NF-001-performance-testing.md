# ML-NF-001 — Page/API Performance (TC-PERF-01)

## Purpose
Measure response time for normal, non-training requests and compare with the approved SRS target.

## Repository finding
The application entry point is a Streamlit app (`app.py`). This initial check measures the app page HTTP response. Do not claim API coverage unless API endpoints are identified. Model-training time is outside this test scope.

## Prerequisites
- Python 3.9 or newer.
- Install dependencies with `pip install -r requirements.txt`.
- Start the app: `streamlit run app.py`.
- In another terminal, run the test from the repository root.

## Run the test
```bash
python tests/performance_test.py --url http://127.0.0.1:8501 --runs 10
```

Once the approved SRS target is known, pass it explicitly using `--target-ms <SRS_TARGET_IN_MS>`. Do not use an example value as the project requirement.

The script writes `performance-results/tc-perf-01-results.csv` and `performance-results/tc-perf-01-summary.json`. The CSV records every request; the JSON reports average, median, P95, failures, target and decision.

If `--target-ms` is omitted, the result is `NOT_VERIFIED_TARGET_NOT_PROVIDED`; the script does not invent an SRS target.

## Test protocol
1. Record OS, Python version, app version, test date and relevant test conditions.
2. Start the app and wait until ready.
3. Run 10 requests as a baseline; repeat under agreed normal-load conditions if required.
4. Compare P95 with the approved SRS target and investigate errors/timeouts.
5. Attach results to Issue #21 or the sprint evidence.
6. Do not mark complete until the target and results are reviewed.

## Acceptance criteria
- [ ] Response time measured and CSV saved.
- [ ] Approved SRS target identified and recorded.
- [ ] P95 compared against target with evidence-based pass/fail.
- [ ] Results attached to Issue #21.
- [ ] Relevant API endpoints separately tested if the app exposes them.

## Limitations
This is a lightweight HTTP page-response test, not a full load/stress test. It does not measure client-side rendering, end-to-end interaction time, model training, or unidentified APIs. P95 uses the nearest-rank method. Record conditions for reproducibility.
