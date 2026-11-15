"""Tests for concurrent-user scalability (ML-NF-002 / issue #22)."""

import threading
import unittest

from src.services.load_test_service import (
    DEFAULT_CONCURRENT_USERS,
    LoadTestService,
    default_worker,
)


class TestConcurrentUsers(unittest.TestCase):
    def test_default_concurrency_matches_srs_baseline(self):
        self.assertEqual(DEFAULT_CONCURRENT_USERS, 20)

    def test_twenty_concurrent_users_supported_without_failure(self):
        service = LoadTestService()
        result = service.run(default_worker, concurrent_users=20)

        self.assertEqual(result["concurrent_users"], 20)
        self.assertEqual(result["total_requests"], 20)
        self.assertEqual(result["successes"], 20)
        self.assertEqual(result["failure_count"], 0)
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["duration_seconds"], 0.0)

    def test_more_than_baseline_users_supported(self):
        service = LoadTestService()
        result = service.run(default_worker, concurrent_users=25)
        self.assertTrue(result["passed"])
        self.assertEqual(result["successes"], 25)

    def test_worker_failures_are_recorded(self):
        def failing_worker(index):
            if index % 2 == 0:
                raise RuntimeError("simulated failure")
            return index

        service = LoadTestService()
        result = service.run(failing_worker, concurrent_users=20)

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_count"], 10)
        self.assertEqual(result["successes"], 10)
        self.assertEqual(len(result["failures"]), 10)
        self.assertIn("simulated failure", result["failures"][0]["error"])

    def test_workers_run_in_parallel(self):
        barrier = threading.Barrier(4, timeout=10)

        def synchronising_worker(index):
            # All four workers must reach the barrier, proving concurrency.
            barrier.wait()
            return index

        service = LoadTestService()
        result = service.run(synchronising_worker, concurrent_users=4)
        self.assertTrue(result["passed"])

    def test_invalid_concurrency_rejected(self):
        service = LoadTestService()
        with self.assertRaises(ValueError):
            service.run(default_worker, concurrent_users=0)

    def test_non_callable_worker_rejected(self):
        service = LoadTestService()
        with self.assertRaises(TypeError):
            service.run(None, concurrent_users=5)

    def test_default_worker_computes_metrics(self):
        result = default_worker(1)
        self.assertEqual(result["user"], 1)
        self.assertIn("accuracy", result)


if __name__ == "__main__":
    unittest.main()
