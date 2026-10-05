#!/usr/bin/env python3
"""Regression coverage for scheduler-log vs wrapper-window time zones."""

from __future__ import annotations

import datetime as dt
import unittest

from analyze_r12 import parse_time


class SchedulerTimestampTest(unittest.TestCase):
    def test_pdt_scheduler_timestamp_matches_utc_wrapper_window(self) -> None:
        scheduler = parse_time("2026-10-04 20:28:21.374", "America/Los_Angeles")
        wrapper = parse_time("2026-10-05T03:28:21.374Z")
        self.assertEqual(scheduler.astimezone(dt.timezone.utc), wrapper)


if __name__ == "__main__":
    unittest.main()
