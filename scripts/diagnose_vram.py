#!/usr/bin/env python
"""Stage 1 (ТЗ №2) - precise CUDA memory diagnostics for the standalone run.

Measures, **in-process**, the memory the TrackEverything pipeline actually
uses, split by stage:

    1. load the TrackEverything checkpoint
    2. load the Pi3 geometry backend
    3. run the inference itself
    4. save the outputs

Reported per stage:

    * ``torch.cuda.max_memory_allocated()``  - bytes handed out by the caching
      allocator to tensors (the "real" tensor footprint)
    * ``torch.cuda.max_memory_reserved()``   - bytes reserved from the driver
      by the caching allocator (what the process actually holds)
    * ``torch.cuda.memory_stats()`` highlights - active bytes, segment count
    * device-level VRAM (via ``torch.cuda.mem_get_info()``) sampled by a
      background thread - the only *device* level figure available on WSL2.

NOTE on NVML: ``pynvml.nvmlDeviceGetComputeRunningProcesses()`` reports
``usedGpuMemory = None`` for every process under WSL2 - the platform does not
expose per-process GPU memory attribution. The script still calls NVML so the
situation is visible in the report, but the authoritative per-process numbers
are the torch allocator ones.

Run with:
    pixi run python scripts/diagnose_vram.py --max_frames 16
    pixi run python scripts/diagnose_vram.py --noise --max_frames 16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_PARENT = str(REPO_ROOT / "external")
if UPSTREAM_PARENT not in sys.path:
    sys.path.insert(0, UPSTREAM_PARENT)

MIB = 1024 ** 2


# ---------------------------------------------------------------------------
# device level sampling
# ---------------------------------------------------------------------------
class DeviceSampler(threading.Thread):
    """Samples device level VRAM usage while the pipeline runs."""

    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_used = 0
        self.min_free = None
        self.total = 0
        self.samples = 0
        self._stop_event = threading.Event()

    def run(self) -> None:
        import torch

        while not self._stop_event.is_set():
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
                self.total = total
                self.peak_used = max(self.peak_used, used)
                self.min_free = free if self.min_free is None else min(self.min_free, free)
                self.samples += 1
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def allocator_snapshot() -> dict:
    import torch

    stats = torch.cuda.memory_stats()
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "peak_active_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "num_ooms": int(stats.get("num_ooms", 0)),
        "segment_count": int(stats.get("segment.all.current", 0)),
        "largest_allocation_bytes": int(stats.get("allocated_bytes.all.peak", 0)),
    }


def reset_peaks() -> None:
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def nvml_snapshot() -> dict:
    """Best-effort NVML report (per-process attribution is unavailable on WSL2)."""
    out: dict = {"available": False}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        procs = []
        for pr in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            procs.append({"pid": int(pr.pid), "used_gpu_memory_mb": pr.usedGpuMemory})
        out = {
            "available": True,
            "device_name": pynvml.nvmlDeviceGetName(handle),
            "device_total_mib": int(info.total // MIB),
            "device_used_mib": int(info.used // MIB),
            "device_free_mib": int(info.free // MIB),
            "compute_processes": procs,
            "per_process_memory_supported": any(
                p["used_gpu_memory_mb"] is not None for p in procs
            ),
            "note": (
                "WSL2 does not expose per-process GPU memory; "
                "usedGpuMemory is None for every process."
            ),
        }
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def print_table(rows: list[tuple[str, dict]], baseline_device_used: int) -> None:
    print()
    print("=" * 112)
    print("CUDA MEMORY BY STAGE")
    print("=" * 112)
    print(f"{'stage':<36}{'max_alloc (MiB)':>18}{'max_reserved (MiB)':>20}"
          f"{'peak_active (MiB)':>20}{'segments':>10}")
    print("-" * 112)
    for name, snap in rows:
        print(f"{name:<36}"
              f"{snap['max_memory_allocated_bytes'] / MIB:>18.2f}"
              f"{snap['max_memory_reserved_bytes'] / MIB:>20.2f}"
              f"{snap['peak_active_bytes'] / MIB:>20.2f}"
              f"{snap['segment_count']:>10}")
    print("-" * 112)
    print(f"{'baseline device used (MiB)':<36}{baseline_device_used / MIB:>18.2f}")
    print()


# ---------------------------------------------------------------------------
# input preparation
# ---------------------------------------------------------------------------
def make_frames_dir(args) -> tuple[Path, bool]:
    """Return (frames_dir, is_temporary)."""
    if args.frames_dir:
        return Path(args.frames_dir), False

    tmp = Path(REPO_ROOT) / "outputs" / "_vram_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("*.png"):
        old.unlink()

    if args.noise:
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(0)
        for i in range(args.num_frames):
            img = rng.integers(0, 255, size=(args.height, args.width, 3), dtype=np.uint8)
            Image.fromarray(img).save(tmp / f"frame_{i:06d}.png")
        return tmp, True

    src = Path(args.source_video)
    if not src.exists():
        raise FileNotFoundError(
            f"source video {src} not found - pass --source_video or --noise"
        )
    import av
    from PIL import Image

    container = av.open(str(src))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    for i, frame in enumerate(frames[: args.num_frames]):
        Image.fromarray(frame).save(tmp / f"frame_{i:06d}.png")
    return tmp, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames_dir", default=None,
                    help="directory with input frames; generated when omitted")
    ap.add_argument("--source_video",
                    default=str(REPO_ROOT / "test_data" / "synthetic_16f.mp4"),
                    help="video converted to a frame directory when --frames_dir is omitted")
    ap.add_argument("--noise", action="store_true",
                    help="use pure-noise frames (worst case for the tracker)")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--num_iters", type=int, default=3)
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--geometry_backend", default="pi3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="katefgroup/TrackEverything")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "vram_diagnostics" / "run.json"))
    ap.add_argument("--keep_frames", action="store_true")
    args = ap.parse_args()

    import torch

    print("=" * 112)
    print("TrackEverything CUDA memory diagnostics")
    print("=" * 112)
    print(f"  python           : {platform.python_version()}  ({sys.executable})")
    print(f"  torch            : {torch.__version__}  (CUDA build {torch.version.cuda})")
    print(f"  device           : {torch.cuda.get_device_name(0)}")
    print(f"  upstream parent  : {UPSTREAM_PARENT}")

    frames_dir, is_tmp = make_frames_dir(args)
    print(f"  frames dir       : {frames_dir}" + ("  (pure noise)" if args.noise else ""))

    sampler = DeviceSampler()
    sampler.start()
    time.sleep(0.3)

    # --- baseline -------------------------------------------------------
    reset_peaks()
    baseline = allocator_snapshot()
    free_now, total = torch.cuda.mem_get_info()
    baseline_device_used = total - free_now
    print(f"  baseline device  : {baseline_device_used / MIB:.2f} MiB used "
          f"of {total / MIB:.2f} MiB")

    rows: list[tuple[str, dict]] = []
    timings: dict[str, float] = {}

    from trackeverything.demo_hf import TrackEverything

    # --- stage 1: tracker checkpoint ------------------------------------
    t0 = time.time()
    model = TrackEverything.from_pretrained(
        args.model, device=args.device, geometry_backend=args.geometry_backend
    )
    torch.cuda.synchronize()
    timings["load_tracker_seconds"] = round(time.time() - t0, 3)
    rows.append(("1. load TrackEverything", allocator_snapshot()))
    print(f"  [1] tracker loaded in {timings['load_tracker_seconds']:.2f}s")

    # --- stage 2: Pi3 geometry weights ----------------------------------
    # The upstream annotator loads Pi3 lazily inside `_load_model()`, i.e.
    # during the first `process_video()` call. Force it here so the tracker
    # and the geometry backend are measured separately.
    reset_peaks()
    t0 = time.time()
    annotator = model.geometry
    loader = getattr(annotator, "_load_model", None)
    if loader is not None:
        loader()
    torch.cuda.synchronize()
    timings["load_geometry_seconds"] = round(time.time() - t0, 3)
    rows.append(("2. load Pi3 geometry", allocator_snapshot()))
    print(f"  [2] Pi3 loaded in {timings['load_geometry_seconds']:.2f}s")

    # --- stage 3: inference (geometry estimation + tracking) -------------
    reset_peaks()
    t0 = time.time()
    output = model.infer(
        video=str(frames_dir),
        mode="dense",
        num_iters=args.num_iters,
        voxel_size=args.voxel_size,
        mem_efficient=True,
        max_frames=args.max_frames,
    )
    torch.cuda.synchronize()
    timings["inference_seconds"] = round(time.time() - t0, 3)
    rows.append(("3. inference (dense)", allocator_snapshot()))
    print(f"  [3] inference done in {timings['inference_seconds']:.2f}s")

    # --- stage 4: saving ------------------------------------------------
    reset_peaks()
    from trackeverything.demo_hf import save_outputs

    out_dir = Path(REPO_ROOT) / "outputs" / "_vram_out"
    t0 = time.time()
    save_outputs(output, out_dir, save_dense=False)
    timings["save_seconds"] = round(time.time() - t0, 3)
    rows.append(("4. save outputs", allocator_snapshot()))

    sampler.stop()
    sampler.join(timeout=2.0)

    tracks = output.tracks()
    n_tracks = int(tracks.coords.shape[0])
    n_frames = int(tracks.coords.shape[1])

    print_table(rows, baseline_device_used)
    print(f"  device-level peak used : {sampler.peak_used / MIB:.2f} MiB "
          f"(of {sampler.total / MIB:.2f} MiB total)")
    print(f"  device-level min free  : {(sampler.min_free or 0) / MIB:.2f} MiB")
    print(f"  samples                : {sampler.samples}")
    print()
    print(f"  tracks                 : N={n_tracks}  T={n_frames}")
    print(f"  working resolution     : {tuple(output.depths.shape[-2:])} (H, W)")
    print(f"  norm_factor            : {output.norm_factor}")
    print()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_total_mib": int(total // MIB),
        "pid": os.getpid(),
        "parameters": {
            "model": args.model,
            "geometry_backend": args.geometry_backend,
            "num_iters": args.num_iters,
            "voxel_size": args.voxel_size,
            "mode": "dense",
            "mem_efficient": True,
            "max_frames": args.max_frames,
            "frames_dir": str(frames_dir),
            "input_size_wh": [args.width, args.height],
            "input_kind": "noise" if args.noise else "synthetic_clip",
        },
        "stages": {name: snap for name, snap in rows},
        "baseline": baseline,
        "device_level": {
            "baseline_used_mib": round(baseline_device_used / MIB, 2),
            "peak_used_mib": round(sampler.peak_used / MIB, 2),
            "min_free_mib": round((sampler.min_free or 0) / MIB, 2),
            "samples": sampler.samples,
        },
        "nvml": nvml_snapshot(),
        "timings": timings,
        "results": {
            "num_tracks": n_tracks,
            "num_frames": n_frames,
            "working_resolution_hw": list(output.depths.shape[-2:]),
            "norm_factor": output.norm_factor,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  report written to {out_path}")

    if is_tmp and not args.keep_frames:
        for f in frames_dir.glob("*.png"):
            f.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
