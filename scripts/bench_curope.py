#!/usr/bin/env python
"""Compare the CUDA cuRoPE2D implementation with the pure-pytorch fallback.

Runs the **same** upstream pipeline twice - once with the compiled CUDA
kernel and once with the fallback (forced through
``TRACKEVERYTHING_FORCE_ROPE_FALLBACK=1``, which only hides ``models.curope``
from the import system; the upstream source is not touched) - and reports

* the wall clock of the geometry stage and of the whole run,
* the peak VRAM of both runs,
* the numerical difference of the geometry (depths / intrinsics / extrinsics)
  and of the tracks (coords / visibs / dyn_logits).

Run with:
    pixi run python scripts/bench_curope.py --frames-dir outputs/_frames_test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER = REPO_ROOT / "worker" / "runner.py"
MIB = 1024 ** 2


def run_worker(frames_dir: Path, out_dir: Path, args, force_fallback: bool) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "external") + os.pathsep + env.get("PYTHONPATH", "")
    if force_fallback:
        env["TRACKEVERYTHING_FORCE_ROPE_FALLBACK"] = "1"
    else:
        env.pop("TRACKEVERYTHING_FORCE_ROPE_FALLBACK", None)

    cmd = [
        sys.executable, str(WORKER),
        "--frames-dir", str(frames_dir),
        "--output-dir", str(out_dir),
        "--geometry-backend", args.geometry_backend,
        "--mode", args.mode,
        "--num-iters", str(args.num_iters),
        "--voxel-size", str(args.voxel_size),
        "--mem-efficient",
        "--max-frames", str(args.max_frames),
    ]
    started = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    wall = time.time() - started
    if proc.returncode != 0:
        raise SystemExit(f"the worker failed (exit {proc.returncode})")

    report = json.loads((out_dir / "worker_report.json").read_text())
    report["_wall"] = wall
    return report


def load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def compare_tensor(name: str, a, b) -> dict:
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return {"name": name, "comparable": False}
    if a.shape != b.shape:
        return {
            "name": name, "comparable": False,
            "shape_a": list(a.shape), "shape_b": list(b.shape),
        }
    if a.dtype != b.dtype:
        return {"name": name, "comparable": False, "dtype_a": str(a.dtype),
                "dtype_b": str(b.dtype)}
    if a.numel() == 0:
        return {"name": name, "comparable": True, "max_abs_diff": 0.0, "rel": 0.0}

    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = float(diff.max())
    scale = float(af.abs().max())
    rel = max_abs / scale if scale > 0 else max_abs
    return {
        "name": name,
        "comparable": True,
        "shape": list(a.shape),
        "max_abs_diff": max_abs,
        "rel": rel,
    }


def compare_dirs(dir_a: Path, dir_b: Path) -> list[dict]:
    import torch

    rows: list[dict] = []
    for fname in ("geometry.pt", "tracks.pt"):
        pa, pb = dir_a / fname, dir_b / fname
        if not (pa.exists() and pb.exists()):
            rows.append({"name": fname, "comparable": False, "reason": "missing"})
            continue
        ta, tb = load(pa), load(pb)
        for key in sorted(set(ta) | set(tb)):
            if key not in ta or key not in tb:
                rows.append({"name": f"{fname}.{key}", "comparable": False,
                             "reason": "missing key"})
                continue
            rows.append(compare_tensor(f"{fname}.{key}", ta[key], tb[key]))
    return rows


def photometric_error(geometry: dict, frames_dir: Path) -> float:
    """Multi-view photometric consistency of the estimated geometry.

    The depth maps are projected into the next frame with the estimated
    cameras and compared with the real pixels. A lower error means a more
    self-consistent (better) geometry - this is the objective quality measure
    that tells whether the CUDA rope degrades the result or not.
    """
    import cv2
    import numpy as np
    import torch

    depths = geometry["depths"].float()          # (T, H, W)
    intr = geometry["intrinsics"].float()        # (T, 3, 3)
    extr = geometry["extrinsics"].float()        # (T, 4, 4)  world -> camera
    T, H, W = depths.shape

    # the source frames, resized to the geometry resolution
    paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    frames = []
    for path in paths[:T]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR))
    if len(frames) < 2:
        return float("nan")
    frames_np = np.stack(frames).astype(np.float32) / 255.0

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")

    errors = []
    for t in range(T - 1):
        depth = depths[t].numpy()
        valid = np.isfinite(depth) & (depth > 0.05)
        if not valid.any():
            continue
        K = intr[t].numpy()
        E = extr[t].numpy()
        K2 = intr[t + 1].numpy()
        E2 = extr[t + 1].numpy()

        # unproject frame t into the world
        x = (xx - K[0, 2]) / K[0, 0] * depth
        y = (yy - K[1, 2]) / K[1, 1] * depth
        ones = np.ones_like(x)
        rays_cam = np.stack([x, y, ones], axis=-1)          # (H, W, 3)
        world = rays_cam @ E[:3, :3].T + E[:3, 3]            # world -> cam is E

        # project into frame t+1
        cam2 = world @ E2[:3, :3].T + E2[:3, 3]
        z = cam2[..., 2]
        ok = valid & (z > 1e-3)
        if not ok.any():
            continue
        u = (K2[0, 0] * cam2[..., 0] / z + K2[0, 2])
        v = (K2[1, 1] * cam2[..., 1] / z + K2[1, 2])
        inside = ok & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        if inside.sum() < 100:
            continue
        ui = u[inside].astype(np.int32)
        vi = v[inside].astype(np.int32)
        source = frames_np[t][yy[inside].astype(np.int32),
                              xx[inside].astype(np.int32)]
        target = frames_np[t + 1][vi, ui]
        errors.append(float(np.abs(source - target).mean()))

    return float(np.mean(errors)) if errors else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", default=str(REPO_ROOT / "outputs" / "_frames_test"))
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--num-iters", type=int, default=3)
    ap.add_argument("--voxel-size", type=float, default=0.02)
    ap.add_argument("--mode", default="dense")
    ap.add_argument("--geometry_backend", default="pi3")
    ap.add_argument("--repeat", action="store_true", default=True,
                    help="run a warm-up pair first (the page cache would "
                         "otherwise bias the first measured run)")
    ap.add_argument("--no-repeat", dest="repeat", action="store_false")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "curope-benchmark.json"))
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"frames directory {frames_dir} not found")

    tmp = Path(tempfile.mkdtemp(prefix="curope_bench_"))
    cuda_dir = tmp / "cuda"
    fallback_dir = tmp / "fallback"

    print("=" * 78)
    print("cuRoPE2D benchmark: CUDA kernel vs pure-pytorch fallback")
    print("=" * 78)
    print(f"frames   : {frames_dir}")
    print(f"max frames: {args.max_frames}")

    print("\n--- run 1/2: the CUDA kernel")
    report_cuda = run_worker(frames_dir, cuda_dir, args, force_fallback=False)
    print(f"    rope implementation : {report_cuda.get('rope_implementation')}")

    print("\n--- run 2/2: the pure-pytorch fallback")
    report_fb = run_worker(frames_dir, fallback_dir, args, force_fallback=True)
    print(f"    rope implementation : {report_fb.get('rope_implementation')}")

    # A second pair of runs: the first pair pays for the cold page cache of
    # the 3.8 GB Pi3 weights and for the first CUDA context of the process,
    # which would bias the comparison.
    if args.repeat:
        print("\n--- warm-up pair (the page cache is hot now)")
        run_worker(frames_dir, tmp / "cuda_warm", args, force_fallback=False)
        run_worker(frames_dir, tmp / "fallback_warm", args, force_fallback=True)

        print("\n--- run 3/4: the CUDA kernel (measured)")
        report_cuda = run_worker(frames_dir, cuda_dir, args, force_fallback=False)
        print("\n--- run 4/4: the pure-pytorch fallback (measured)")
        report_fb = run_worker(frames_dir, fallback_dir, args, force_fallback=True)

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    def timing(report: dict, key: str) -> float:
        return float(report.get("timings", {}).get(key, 0.0))

    def vram(report: dict) -> float:
        return float(report.get("cuda", {}).get("max_memory_allocated_bytes", 0)) / MIB

    rows = [
        ("geometry estimation", timing(report_cuda, "inference_seconds"),
         timing(report_fb, "inference_seconds")),
        ("total worker time", report_cuda["wall_clock_seconds"],
         report_fb["wall_clock_seconds"]),
        ("peak VRAM (MiB)", vram(report_cuda), vram(report_fb)),
    ]
    print(f"{'metric':<26}{'CUDA':>16}{'fallback':>16}{'speedup':>12}")
    print("-" * 78)
    for name, cuda_value, fb_value in rows:
        speed = (fb_value / cuda_value) if cuda_value else 0.0
        print(f"{name:<26}{cuda_value:>16.2f}{fb_value:>16.2f}{speed:>11.2f}x")
    print("-" * 78)

    print()
    print("numerical agreement (CUDA vs fallback):")
    comparisons = compare_dirs(cuda_dir, fallback_dir)
    for row in comparisons:
        if not row.get("comparable"):
            print(f"    {row['name']:<28} not comparable "
                  f"({row.get('reason', '')}{row.get('shape_a', '')})")
            continue
        print(f"    {row['name']:<28} max|diff| {row['max_abs_diff']:.3e}"
              f"   relative {row['rel']:.3e}")

    # ------------------------------------------------------------------
    # objective quality: the multi-view photometric consistency
    # ------------------------------------------------------------------
    print()
    print("geometry quality (multi-view photometric error, lower is better):")
    geo_cuda = load(cuda_dir / "geometry.pt")
    geo_fb = load(fallback_dir / "geometry.pt")
    photo_cuda = photometric_error(geo_cuda, frames_dir)
    photo_fb = photometric_error(geo_fb, frames_dir)
    print(f"    CUDA kernel : {photo_cuda:.6f}")
    print(f"    fallback    : {photo_fb:.6f}")
    quality_delta = photo_cuda - photo_fb
    print(f"    difference  : {quality_delta:+.6f}"
          f"   ({100.0 * quality_delta / max(photo_fb, 1e-9):+.2f} %)")

    # ------------------------------------------------------------------
    result = {
        "frames_dir": str(frames_dir),
        "max_frames": args.max_frames,
        "cuda": {
            "rope_implementation": report_cuda.get("rope_implementation"),
            "timings": report_cuda.get("timings"),
            "peak_vram_mib": vram(report_cuda),
            "wall_clock_seconds": report_cuda["wall_clock_seconds"],
            "num_tracks": report_cuda.get("results", {}).get("num_tracks"),
            "photometric_error": photo_cuda,
        },
        "fallback": {
            "rope_implementation": report_fb.get("rope_implementation"),
            "timings": report_fb.get("timings"),
            "peak_vram_mib": vram(report_fb),
            "wall_clock_seconds": report_fb["wall_clock_seconds"],
            "num_tracks": report_fb.get("results", {}).get("num_tracks"),
            "photometric_error": photo_fb,
        },
        "photometric_error_delta": quality_delta,
        "comparisons": comparisons,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print()
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
