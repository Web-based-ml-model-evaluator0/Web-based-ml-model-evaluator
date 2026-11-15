"""System availability measurement (ML-NF-003).

Tracks health-check samples and planned-maintenance windows, then computes
time-weighted availability excluding planned maintenance. The SRS target is
at least 99% monthly availability.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


class AvailabilityService:
    """Measures uptime from health-check samples."""

    DEFAULT_TARGET = 0.99

    def __init__(self, target: float = DEFAULT_TARGET) -> None:
        self.target = float(target)
        self._checks: List[Tuple[datetime, bool]] = []
        self._maintenance: List[Tuple[datetime, datetime]] = []

    @staticmethod
    def _as_datetime(value: Optional[datetime]) -> datetime:
        return value if value is not None else datetime.now()

    def record_check(self, is_up: bool, timestamp: Optional[datetime] = None) -> None:
        """Record a health-check result at a point in time."""
        self._checks.append((self._as_datetime(timestamp), bool(is_up)))
        self._checks.sort(key=lambda item: item[0])

    def add_maintenance(
        self,
        start: datetime,
        end: datetime,
    ) -> None:
        """Register a planned-maintenance window to exclude from availability."""
        if end < start:
            raise ValueError("Maintenance end must be after start.")
        self._maintenance.append((start, end))

    def _maintenance_overlap(self, start: datetime, end: datetime) -> float:
        """Return seconds of [start, end] covered by planned maintenance."""
        if end <= start:
            return 0.0
        total = 0.0
        for m_start, m_end in self._maintenance:
            overlap_start = max(start, m_start)
            overlap_end = min(end, m_end)
            if overlap_end > overlap_start:
                total += (overlap_end - overlap_start).total_seconds()
        return total

    def compute_availability(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> float:
        """Compute time-weighted availability in [0, 1], excluding maintenance."""
        if not self._checks:
            return 1.0

        window_start = start or self._checks[0][0]
        window_end = end or self._checks[-1][0]

        if window_end <= window_start:
            return 1.0 if self._checks[-1][1] else 0.0

        # Each check describes the state until the next check (or window end).
        intervals = []
        for index, (timestamp, is_up) in enumerate(self._checks):
            interval_start = max(timestamp, window_start)
            if index + 1 < len(self._checks):
                interval_end = min(self._checks[index + 1][0], window_end)
            else:
                interval_end = window_end
            if interval_end > interval_start:
                intervals.append((interval_start, interval_end, is_up))

        total_seconds = (window_end - window_start).total_seconds()
        maintenance_seconds = self._maintenance_overlap(window_start, window_end)
        monitored_seconds = total_seconds - maintenance_seconds
        if monitored_seconds <= 0:
            return 1.0

        up_seconds = 0.0
        for interval_start, interval_end, is_up in intervals:
            if not is_up:
                continue
            duration = (interval_end - interval_start).total_seconds()
            duration -= self._maintenance_overlap(interval_start, interval_end)
            up_seconds += max(duration, 0.0)

        return max(0.0, min(1.0, up_seconds / monitored_seconds))

    def meets_target(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> bool:
        """Return True when measured availability meets the configured target."""
        return self.compute_availability(start, end) >= self.target

    def get_status(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Return a summary suitable for dashboards and reports."""
        availability = self.compute_availability(start, end)
        return {
            "availability": availability,
            "availability_percent": round(availability * 100.0, 4),
            "target": self.target,
            "target_percent": round(self.target * 100.0, 2),
            "meets_target": availability >= self.target,
            "checks": len(self._checks),
            "maintenance_windows": len(self._maintenance),
        }
