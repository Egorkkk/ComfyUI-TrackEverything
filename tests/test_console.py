#!/usr/bin/env python
"""Unit tests for the console reporting of the workers.

The workers print a stage-by-stage report (model loading, geometry
estimation, tracking, saving) with a progress bar and a rate. These tests
pin the format down and check that the output really reaches the ComfyUI
console.

Run with:
    pixi run python tests/test_console.py
"""

from __future__ import annotations

import io
import logging
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "worker") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "worker"))

from console import (  # noqa: E402
    Console,
    UpstreamLogHandler,
    human_bytes,
    human_time,
    install_upstream_logging,
)


def make_console(tty: bool = False, **kwargs) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(stream, tty=tty, **kwargs), stream


class TestConsole(unittest.TestCase):
    def test_banner_lists_the_parameters(self):
        console, out = make_console()
        console.banner("TrackEverything", {"device": "RTX 4060 Ti", "frames": "16"})
        text = out.getvalue()
        self.assertIn("TrackEverything", text)
        self.assertIn("device", text)
        self.assertIn("RTX 4060 Ti", text)
        self.assertIn("frames", text)
        self.assertIn("16", text)

    def test_stage_shows_the_index_and_the_title(self):
        console, out = make_console()
        console.stage(2, 4, "loading the geometry backend")
        text = out.getvalue()
        self.assertIn("[2/4]", text)
        self.assertIn("loading the geometry backend", text)

    def test_substage_is_indented(self):
        console, out = make_console()
        console.substage("geometry estimation")
        text = out.getvalue()
        self.assertIn("-> geometry estimation", text)
        self.assertTrue(text.startswith(" "))

    def test_detail_lines_are_indented(self):
        console, out = make_console()
        console.detail("demo_hf: Input video: 8 frames")
        self.assertIn("       demo_hf: Input video: 8 frames", out.getvalue())

    def test_stage_done_reports_the_duration(self):
        console, out = make_console()
        console.stage_done(2.5, "520.8 MiB on disk")
        text = out.getvalue()
        self.assertIn("done in 2.50 s", text)
        self.assertIn("520.8 MiB on disk", text)

    def test_bar_shows_percentage_and_rate(self):
        console, out = make_console(tty=True)
        console.bar("geometry chunk", 3, 12, unit=" chunks", rate=0.42,
                    elapsed=7.1, eta=21.4)
        text = out.getvalue()
        self.assertIn("geometry chunk", text)
        self.assertIn("25.0%", text)
        self.assertIn("3/12 chunks", text)
        self.assertIn("0.42 chunks/s", text)
        self.assertIn("elapsed 7s", text)
        self.assertIn("ETA 21s", text)

    def test_bar_is_clamped(self):
        console, out = make_console(tty=True)
        console.bar("x", 99, 12)
        self.assertIn("100.0%", out.getvalue())

    def test_live_rewrites_in_place_on_a_tty(self):
        console, out = make_console(tty=True)
        console.live("first")
        console.live("second")
        text = out.getvalue()
        self.assertIn("\r", text)
        self.assertIn("first", text)
        self.assertIn("second", text)

    def test_live_is_throttled_when_not_a_tty(self):
        console, out = make_console(tty=False, min_interval=100.0)
        console.live("one")
        console.live("two")
        console.live("three")
        text = out.getvalue()
        # only the first line is written, the others are throttled away
        self.assertEqual(text.count("\n"), 1)
        self.assertIn("one", text)
        self.assertNotIn("two", text)

    def test_live_writes_lines_when_not_a_tty(self):
        console, out = make_console(tty=False, min_interval=0.0)
        console.live("one")
        console.live("two")
        text = out.getvalue()
        self.assertIn("one", text)
        self.assertIn("two", text)
        self.assertNotIn("\r", text)

    def test_summary_lists_the_results(self):
        console, out = make_console()
        console.summary({"tracks": 2415, "total": "13.5 s"})
        text = out.getvalue()
        self.assertIn("tracks", text)
        self.assertIn("2415", text)
        self.assertIn("13.5 s", text)

    def test_done_clears_a_live_line(self):
        console, out = make_console(tty=True)
        console.live("running")
        console.done("finished")
        text = out.getvalue()
        self.assertIn("finished", text)
        # the live line was cleared with a carriage return
        self.assertIn("\r", text)


class TestFormatting(unittest.TestCase):
    def test_human_bytes(self):
        self.assertEqual(human_bytes(1024), "1.0 KiB")
        self.assertEqual(human_bytes(1024 ** 2), "1.0 MiB")
        self.assertEqual(human_bytes(1024 ** 3), "1.0 GiB")
        self.assertEqual(human_bytes(512), "512.0 B")

    def test_human_time(self):
        self.assertEqual(human_time(5), " 5s")
        self.assertEqual(human_time(65), " 1m05s")
        self.assertEqual(human_time(3725), " 1h02m")
        self.assertEqual(human_time(-1), "--:--")


class TestUpstreamLogging(unittest.TestCase):
    def test_records_become_detail_lines(self):
        console, out = make_console()
        handler = UpstreamLogHandler(console)
        record = logging.LogRecord(
            name="trackeverything.demo_hf",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Geometry estimation complete.",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        text = out.getvalue()
        self.assertIn("demo_hf: Geometry estimation complete.", text)

    def test_the_callback_sees_every_message(self):
        console, out = make_console()
        seen = []
        handler = UpstreamLogHandler(console, on_message=seen.append)
        for msg in ("Estimating geometry with pi3 (8 frames)...",
                    "Running model inference (8 frames, mode=dense)..."):
            handler.emit(logging.LogRecord(
                name="trackeverything.demo_hf", level=logging.INFO,
                pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None,
            ))
        self.assertEqual(len(seen), 2)
        self.assertIn("Estimating geometry", seen[0])

    def test_install_takes_over_the_root_logger(self):
        console, out = make_console()
        root = logging.getLogger()
        original = list(root.handlers)
        try:
            install_upstream_logging(console)
            self.assertEqual(len(root.handlers), 1)
            self.assertIsInstance(root.handlers[0], UpstreamLogHandler)
            logging.getLogger("trackeverything.demo_hf").info("hello")
            self.assertIn("hello", out.getvalue())
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for handler in original:
                root.addHandler(handler)

    def test_the_chatty_loggers_are_demoted(self):
        console, out = make_console()
        root = logging.getLogger()
        original = list(root.handlers)
        original_levels = {
            name: logging.getLogger(name).level for name in
            ("httpx", "httpx2", "dinov3")
        }
        try:
            install_upstream_logging(console)
            for name in original_levels:
                self.assertGreaterEqual(
                    logging.getLogger(name).level, logging.WARNING,
                    f"{name} should not log at INFO level",
                )
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for handler in original:
                root.addHandler(handler)
            for name, level in original_levels.items():
                logging.getLogger(name).setLevel(level)


if __name__ == "__main__":
    unittest.main(verbosity=2)
