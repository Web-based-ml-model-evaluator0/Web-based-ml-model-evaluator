"""Tests for system availability measurement (ML-NF-003 / issue #23)."""

import unittest
from datetime import datetime, timedelta

from src.services.availability_service import AvailabilityService


BASE = datetime(2026, 11, 1, 0, 0, 0)


class TestAvailabilityService(unittest.TestCase):
    def setUp(self):
        self.service = AvailabilityService(target=0.99)

    def test_default_target_is_99_percent(self):
        self.assertEqual(AvailabilityService().target, 0.99)

    def test_no_checks_reports_full_availability(self):
        self.assertEqual(self.service.compute_availability(), 1.0)

    def test_all_up_checks_meet_target(self):
        for hours in range(0, 24, 1):
            self.service.record_check(True, BASE + timedelta(hours=hours))

        self.assertAlmostEqual(self.service.compute_availability(), 1.0)
        self.assertTrue(self.service.meets_target())

    def test_downtime_lowers_availability(self):
        # 23 hours up, 1 hour down across a 24 hour window.
        self.service.record_check(True, BASE)
        self.service.record_check(False, BASE + timedelta(hours=23))

        availability = self.service.compute_availability(
            BASE, BASE + timedelta(hours=24)
        )
        self.assertAlmostEqual(availability, 23 / 24, places=4)
        self.assertFalse(self.service.meets_target(BASE, BASE + timedelta(hours=24)))

    def test_small_downtime_still_meets_target(self):
        # 24 hours window with 10 minutes of downtime = 99.3% availability.
        self.service.record_check(True, BASE)
        self.service.record_check(False, BASE + timedelta(hours=12))
        self.service.record_check(True, BASE + timedelta(hours=12, minutes=10))

        availability = self.service.compute_availability(
            BASE, BASE + timedelta(hours=24)
        )
        self.assertGreaterEqual(availability, 0.99)
        self.assertTrue(self.service.meets_target(BASE, BASE + timedelta(hours=24)))

    def test_planned_maintenance_is_excluded(self):
        # 1 hour downtime that falls entirely inside planned maintenance.
        self.service.record_check(True, BASE)
        self.service.record_check(False, BASE + timedelta(hours=2))
        self.service.record_check(True, BASE + timedelta(hours=3))
        self.service.add_maintenance(
            BASE + timedelta(hours=2),
            BASE + timedelta(hours=3),
        )

        availability = self.service.compute_availability(
            BASE, BASE + timedelta(hours=4)
        )
        self.assertAlmostEqual(availability, 1.0, places=4)
        self.assertTrue(self.service.meets_target(BASE, BASE + timedelta(hours=4)))

    def test_maintenance_does_not_hide_unplanned_downtime(self):
        self.service.record_check(True, BASE)
        self.service.record_check(False, BASE + timedelta(hours=1))
        self.service.add_maintenance(
            BASE + timedelta(hours=10),
            BASE + timedelta(hours=11),
        )

        availability = self.service.compute_availability(
            BASE, BASE + timedelta(hours=2)
        )
        self.assertAlmostEqual(availability, 0.5, places=4)

    def test_invalid_maintenance_window_rejected(self):
        with self.assertRaises(ValueError):
            self.service.add_maintenance(BASE, BASE - timedelta(hours=1))

    def test_get_status_reports_target_details(self):
        self.service.record_check(True, BASE)
        self.service.record_check(True, BASE + timedelta(hours=1))

        status = self.service.get_status(BASE, BASE + timedelta(hours=1))
        self.assertTrue(status["meets_target"])
        self.assertEqual(status["target_percent"], 99.0)
        self.assertAlmostEqual(status["availability_percent"], 100.0)
        self.assertEqual(status["checks"], 2)

    def test_unordered_checks_are_sorted(self):
        self.service.record_check(True, BASE + timedelta(hours=2))
        self.service.record_check(False, BASE)
        availability = self.service.compute_availability(
            BASE, BASE + timedelta(hours=3)
        )
        self.assertAlmostEqual(availability, 1 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
