#!/usr/bin/env python
"""Fake TrackEverything worker used by the automated tests.

Accepts the same command line as ``worker/runner.py`` but does **not** load
any model: it writes small but *valid* ``tracks.pt`` / ``geometry.pt`` files
plus a ``worker_report.json``, so the bridge can be exercised end to end
without a GPU or a real inference run.

Behaviour switches (for the error-path tests):

    --fail            exit with code 1 and print a traceback-ish message
    --missing-outputs exit with code 0 but write no result files
    --hang            sleep forever (used for the timeout test)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="katefgroup/TrackEverything")
    ap.add_argument("--pi3-checkpoint", default="yyfz233/Pi3")
    ap.add_argument("--geometry-backend", default="pi3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="dense")
    ap.add_argument("--num-iters", type=int, default=3)
    ap.add_argument("--voxel-size", type=float, default=0.02)
    ap.add_argument("--mem-efficient", dest="mem_efficient", action="store_true", default=True)
    ap.add_argument("--no-mem-efficient", dest="mem_efficient", action="store_false")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fail", action="store_true")
    ap.add_argument("--missing-outputs", action="store_true")
    ap.add_argument("--hang", action="store_true")
    args = ap.parse_args()

    if args.hang:
        time.sleep(3600)

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fail:
        print("ERROR: simulated worker failure (CUDA out of memory)", file=sys.stderr)
        print("Traceback (most recent call last):", file=sys.stderr)
        print("  RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
              file=sys.stderr)
        return 1

    num_frames = len(sorted(frames_dir.glob("*.png")))
    if args.max_frames:
        num_frames = min(num_frames, args.max_frames)
    num_frames = max(num_frames, 1)

    if not args.missing_outputs:
        import torch

        n_tracks = 7
        h, w = 8, 12
        tracks = {
            "coords": torch.full((n_tracks, num_frames, 3), 0.5, dtype=torch.float32),
            "visibs": torch.full((n_tracks, num_frames), 3.0, dtype=torch.float32),
            "dyn_logits": torch.full((n_tracks, num_frames), -1.0, dtype=torch.float32),
            "src_time": torch.zeros(n_tracks, dtype=torch.int64),
            "src_y": torch.zeros(n_tracks, dtype=torch.int64),
            "src_x": torch.zeros(n_tracks, dtype=torch.int64),
            "cell_to_token": torch.zeros(num_frames, h, w, dtype=torch.int64),
        }
        torch.save(tracks, output_dir / "tracks.pt")

        geometry = {
            "norm_factor": 1.5,
            "depths": torch.ones(num_frames, 384, 512, dtype=torch.float32),
            "intrinsics": torch.eye(3).expand(num_frames, 3, 3).contiguous(),
            "extrinsics": torch.eye(4).expand(num_frames, 4, 4).contiguous(),
        }
        torch.save(geometry, output_dir / "geometry.pt")

    report = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "versions": {
            "python": sys.version.split()[0],
            "torch": "fake",
            "upstream_commit": "faketest",
            "cuda_device": "fake",
        },
        "parameters": {
            "model": args.model,
            "geometry_backend": args.geometry_backend,
            "mode": args.mode,
            "num_iters": args.num_iters,
            "voxel_size": args.voxel_size,
            "mem_efficient": args.mem_efficient,
        },
        "timings": {"inference_seconds": 0.01},
        "results": {
            "num_tracks": 7,
            "num_frames": num_frames,
            "feature_grid_hw": [96, 128],
            "working_resolution_hw": [384, 512],
            "norm_factor": 1.5,
            "voxel_size": args.voxel_size,
            "dynamic_threshold": 0.5,
        },
        "cuda": {"max_memory_allocated_bytes": 0},
        "wall_clock_seconds": 0.01,
    }
    (output_dir / "worker_report.json").write_text(json.dumps(report, indent=2))
    print(f"[fake-worker] wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
