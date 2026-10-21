#!/usr/bin/env python
"""Stage D - run the ORIGINAL upstream CLI and measure time / peak VRAM.

This is a thin wrapper around

    python -m trackeverything.demo_hf ...

It does not modify the upstream code in any way: the upstream module is
executed as a subprocess with the exact command line given on the command
line of this script. The wrapper only

  * records the wall-clock duration,
  * samples device-level VRAM usage while the run is in progress,
  * forwards stdout/stderr unchanged,
  * writes a small JSON report next to the outputs.

Usage:
    pixi run python scripts/run_inference.py \
        --video test_data/synthetic_16f.mp4 \
        --output_dir outputs/standalone \
        --geometry_backend pi3 --mode dense --num_iters 3 \
        --voxel_size 0.02 --mem_efficient --max_frames 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class VramMonitor(threading.Thread):
    """Polls the device-level VRAM usage (includes other processes)."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_used = 0
        self.peak_free = None
        self.total = None
        self.samples = 0
        # NOTE: do not name this `_stop` - it would shadow threading.Thread._stop.
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            self.total = total
            while not self._stop_event.is_set():
                free, total = torch.cuda.mem_get_info()
                used = total - free
                self.peak_used = max(self.peak_used, used)
                self.peak_free = min(self.peak_free, free) if self.peak_free is not None else free
                self.samples += 1
                time.sleep(self.interval)
        except Exception as exc:  # pragma: no cover
            print(f"[vram-monitor] disabled: {exc!r}")

    def stop(self) -> None:
        self._stop_event.set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output_dir", default="outputs/standalone")
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--mode", default="dense")
    ap.add_argument("--geometry_backend", default="pi3")
    ap.add_argument("--num_iters", type=int, default=3)
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--mem_efficient", action="store_true", default=True,
                    help="upstream uses BooleanOptionalAction; --no-mem_efficient disables it")
    ap.add_argument("--no_mem_efficient", action="store_true",
                    help="pass --no-mem_efficient to the upstream CLI")
    ap.add_argument("--save_dense", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", nargs="*", default=[], help="extra upstream CLI args")
    args = ap.parse_args()

    env = dict(os.environ)
    # The upstream repo IS the `trackeverything` package -> its PARENT must be importable.
    pp = str(REPO_ROOT / "external")
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [
        sys.executable, "-m", "trackeverything.demo_hf",
        "--video", args.video,
        "--output_dir", args.output_dir,
        "--geometry_backend", args.geometry_backend,
        "--mode", args.mode,
        "--num_iters", str(args.num_iters),
        "--voxel_size", str(args.voxel_size),
        "--frame_stride", str(args.frame_stride),
        "--start_frame", str(args.start_frame),
        "--device", args.device,
    ]
    # upstream: --mem_efficient / --no-mem_efficient (BooleanOptionalAction, no value)
    cmd += ["--mem_efficient" if not args.no_mem_efficient else "--no-mem_efficient"]
    if args.max_frames:
        cmd += ["--max_frames", str(args.max_frames)]
    if args.save_dense:
        cmd += ["--save_dense"]
    cmd += list(args.extra)

    print("=" * 78)
    print("COMMAND:")
    print("  " + " ".join(cmd))
    print("  PYTHONPATH=" + env["PYTHONPATH"])
    print("=" * 78, flush=True)

    monitor = VramMonitor()
    monitor.start()

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    monitor.stop()
    monitor.join(timeout=2.0)

    gib = 1024 ** 3
    report = {
        "command": cmd,
        "video": args.video,
        "output_dir": args.output_dir,
        "returncode": proc.returncode,
        "wall_clock_seconds": round(elapsed, 2),
        "vram_samples": monitor.samples,
        "vram_total_gib": round(monitor.total / gib, 2) if monitor.total else None,
        "vram_peak_used_gib": round(monitor.peak_used / gib, 2) if monitor.total else None,
        "vram_peak_used_mib": round(monitor.peak_used / 1024 ** 2) if monitor.total else None,
        "vram_min_free_mib": round(monitor.peak_free / 1024 ** 2) if monitor.peak_free else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Also write a metadata.json so the result can be consumed directly by
    # the ComfyUI nodes (the same layout the Track node produces).
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from comfy_nodes.bridge import synthesize_metadata

        metadata = synthesize_metadata(out_dir)
        metadata["inference"] = {
            "mode": args.mode,
            "geometry_backend": args.geometry_backend,
            "num_iters": args.num_iters,
            "voxel_size": args.voxel_size,
            "mem_efficient": True,
            "device": args.device,
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "start_frame": args.start_frame,
        }
        metadata["worker"] = {
            "command": cmd,
            "returncode": proc.returncode,
            "wall_clock_seconds": round(elapsed, 2),
            "log": "run_report.json",
            "pid_isolated": False,
        }
        metadata["files"]["metadata"] = "metadata.json"
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"  metadata written to       : {out_dir / 'metadata.json'}")
    except Exception as exc:  # noqa: BLE001 - never fail the run for this
        print(f"  (could not write metadata.json: {exc!r})")

    print()
    print("=" * 78)
    print("RUN REPORT")
    print("=" * 78)
    for k, v in report.items():
        if k != "command":
            print(f"  {k:24s}: {v}")
    print(f"  report written to        : {report_path}")
    print("=" * 78)

    if proc.returncode != 0:
        print(f"FAILED with exit code {proc.returncode}")
        sys.exit(proc.returncode)
    print("SUCCEEDED")


if __name__ == "__main__":
    main()
