"""Console reporting for the TrackEverything workers.

A small, dependency-free progress reporter used by ``worker/runner.py`` and
``worker/visualize.py``. It prints a readable, stage-by-stage report:

* a banner with the run parameters,
* one line per stage (``[3/5] estimating geometry (pi3)``),
* a live progress bar with a rate whenever the total is known,
* a heartbeat line (elapsed + rate) for stages whose total is not known,
* detail lines for the upstream log records,
* a summary with the total time and the peak VRAM.

On a TTY the live lines are rewritten in place; when the output is a pipe
(which is the case when ComfyUI runs the worker) they are emitted as normal
lines at a throttled interval, so the log file stays readable.
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from typing import Optional, TextIO

BAR_WIDTH = 24


def _human_bytes(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num) < 1024.0 or unit == "GiB":
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} GiB"


def _human_time(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:      # negative or NaN
        return "--:--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds:2d}s"
    if seconds < 3600:
        return f"{seconds // 60:2d}m{seconds % 60:02d}s"
    return f" {seconds // 3600:d}h{(seconds % 3600) // 60:02d}m"


class Console:
    """Stage-aware console reporter."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        *,
        tty: Optional[bool] = None,
        min_interval: float = 2.0,
        prefix: str = "",
    ):
        self.stream = stream if stream is not None else sys.stdout
        self.is_tty = self.stream.isatty() if tty is None else tty
        self.min_interval = min_interval
        self.prefix = prefix
        self._live_open = False
        self._live_last = 0.0
        self._live_text = ""
        self._stage_started: Optional[float] = None

    # -- low level -------------------------------------------------------
    def _write(self, text: str) -> None:
        self._clear_live()
        self.stream.write(text)
        self.stream.flush()

    def _clear_live(self) -> None:
        if self._live_open:
            width = shutil.get_terminal_size((100, 24)).columns
            self.stream.write("\r" + " " * max(0, width - 1) + "\r")
            self.stream.flush()
            self._live_open = False

    # -- structure -------------------------------------------------------
    def banner(self, title: str, rows: dict) -> None:
        """A boxed header with the run parameters."""
        width = 74
        self._write("\n" + "=" * width + "\n")
        self._write(f" {title}\n")
        self._write("=" * width + "\n")
        for key, value in rows.items():
            self._write(f"   {key:<16s}: {value}\n")
        self._write("=" * width + "\n")

    def rule(self, title: str = "") -> None:
        width = 74
        self._write("-" * width + "\n")
        if title:
            self._write(f" {title}\n")

    def stage(self, index: int, total: int, title: str) -> None:
        """Announce a new stage."""
        self._stage_started = time.time()
        self._write(f"\n[{index}/{total}] {title}\n")

    def substage(self, title: str) -> None:
        """A sub-step inside the current stage."""
        self._write(f"       -> {title}\n")

    def detail(self, message: str) -> None:
        """An indented sub-line."""
        for line in str(message).rstrip("\n").splitlines() or [""]:
            self._write(f"       {line}\n")

    def note(self, message: str) -> None:
        self._write(f"       {message}\n")

    # -- progress --------------------------------------------------------
    def live(self, text: str) -> None:
        """Update a live line.

        On a TTY the line is rewritten in place; otherwise it is emitted as a
        normal line, at most once every ``min_interval`` seconds.
        """
        now = time.time()
        if self.is_tty:
            self._clear_live()
            self.stream.write("\r" + text.ljust(shutil.get_terminal_size((100, 24)).columns - 1))
            self.stream.flush()
            self._live_open = True
            self._live_text = text
            return
        if now - self._live_last >= self.min_interval:
            self._live_last = now
            self._write(f"       {text}\n")

    def bar(
        self,
        label: str,
        done: float,
        total: float,
        *,
        unit: str = "",
        rate: Optional[float] = None,
        elapsed: Optional[float] = None,
        eta: Optional[float] = None,
    ) -> None:
        """A progress bar with a rate."""
        total = max(float(total), 1e-9)
        done = max(0.0, min(float(done), total))
        fraction = done / total
        filled = int(round(BAR_WIDTH * fraction))
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)

        parts = [f"[{bar}] {fraction * 100:5.1f}%"]
        unit = unit.strip()
        if total >= 1:
            shown_done = int(done)
            shown_total = int(total)
            parts.append(f"{shown_done}/{shown_total}{(' ' + unit) if unit else ''}")
        if rate is not None:
            parts.append(f"{rate:.2f} {unit}/s" if unit else f"{rate:.2f}/s")
        if elapsed is not None:
            parts.append(f"elapsed {_human_time(elapsed).strip()}")
        if eta is not None and fraction < 1.0:
            parts.append(f"ETA {_human_time(eta).strip()}")

        self.live(f"{label:<26s} " + "  ".join(parts))

    def done(self, message: str) -> None:
        """Finish a live line with a final, permanent message."""
        self._clear_live()
        self._write(f"       {message}\n")

    def stage_done(self, elapsed: float, extra: str = "") -> None:
        """Close a stage with its duration."""
        text = f"done in {elapsed:.2f} s"
        if extra:
            text += f"   {extra}"
        self.done(text)

    def summary(self, rows: dict) -> None:
        self._clear_live()
        width = 74
        self._write("\n" + "=" * width + "\n")
        for key, value in rows.items():
            self._write(f"   {key:<16s}: {value}\n")
        self._write("=" * width + "\n")


class UpstreamLogHandler(logging.Handler):
    """Routes the upstream log records into the console as detail lines."""

    LEVELS = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warn",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, console: Console, level: int = logging.INFO,
                 on_message=None):
        super().__init__(level=level)
        self.console = console
        self.on_message = on_message
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - never break the run
            return
        if not message:
            return
        name = record.name.replace("trackeverything.", "")
        self.console.detail(f"{name}: {message}")
        if self.on_message is not None:
            try:
                self.on_message(message)
            except Exception:  # pragma: no cover
                pass


def install_upstream_logging(console: Console, level: int = logging.INFO,
                             on_message=None) -> None:
    """Route every log record into ``console`` (and nowhere else).

    The upstream code logs through the standard :mod:`logging` module. By
    taking over the root logger we keep its output in our own format instead
    of the interleaved default handlers.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(UpstreamLogHandler(console, level=level,
                                       on_message=on_message))
    root.setLevel(level)

    # the HTTP libraries are extremely chatty at INFO level and would bury
    # the actual progress
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "filelock",
                  "huggingface_hub", "transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # dinov3 dumps its whole rope configuration for every variant - pure
    # repetition; the useful parts (embed dim, interaction indexes, patch
    # size) are logged by the dinov3_adapter / dinov3_encoder loggers
    logging.getLogger("dinov3").setLevel(logging.WARNING)


def human_bytes(num: float) -> str:
    return _human_bytes(num)


def human_time(seconds: float) -> str:
    return _human_time(seconds)
