"""Concurrent-user load testing (ML-NF-002).

The SRS requires the baseline deployment to support at least 20 concurrent
active users without functional failure. This service runs a worker function
in parallel and records successes, failures and timing so the result can be
attached to the requirement evidence.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional


DEFAULT_CONCURRENT_USERS = 20


class LoadTestService:
    """Runs a callable concurrently and summarises the outcome."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers

    def run(
        self,
        worker: Callable[[int], Any],
        concurrent_users: int = DEFAULT_CONCURRENT_USERS,
        timeout: Optional[float] = 60.0,
    ) -> Dict[str, Any]:
        """Execute ``worker(index)`` for each simulated user.

        A run is successful when no worker raises an exception.
        """
        if concurrent_users < 1:
            raise ValueError("concurrent_users must be at least 1.")
        if not callable(worker):
            raise TypeError("worker must be callable.")

        started = time.perf_counter()
        successes = 0
        failures: List[Dict[str, Any]] = []
        max_workers = self.max_workers or concurrent_users

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker, index): index for index in range(concurrent_users)
            }
            for future in as_completed(futures, timeout=timeout):
                index = futures[future]
                try:
                    future.result()
                    successes += 1
                except Exception as error:  # noqa: BLE001 - record any failure
                    failures.append({"user": index, "error": repr(error)})

        elapsed = time.perf_counter() - started
        return {
            "concurrent_users": concurrent_users,
            "total_requests": concurrent_users,
            "successes": successes,
            "failures": failures,
            "failure_count": len(failures),
            "duration_seconds": round(elapsed, 4),
            "passed": len(failures) == 0,
        }


def default_worker(index: int) -> Dict[str, Any]:
    """A lightweight worker used for baseline concurrency smoke tests."""
    from src.services.evaluation_service import EvaluationService

    metrics = EvaluationService.evaluate_classification(
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 1, 1],
    )
    return {"user": index, "accuracy": metrics["accuracy"]}
