#!/usr/bin/env python
"""TrackEverything worker - runs inside the **isolated** environment.

This process is launched by the ComfyUI node (see ``comfy_nodes/bridge.py``)
using the python interpreter of the isolated Pixi environment. It is the only
place where ``trackeverything`` and ``torch_scatter`` are imported.

It calls the **original upstream** implementation
(``trackeverything.demo_hf``) without modifying it:

    from trackeverything.demo_hf import TrackEverything, save_outputs
    model = TrackEverything.from_pretrained(...)
    output = model.infer(video=<frames dir>, ...)
    save_outputs(output, output_dir, save_dense=False)

``save_outputs`` is the upstream function, so ``tracks.pt`` and
``geometry.pt`` are byte-for-byte in the upstream format and the numeric
values are untouched.

On top of the upstream files the worker writes ``worker_report.json`` with
runtime facts (versions, timings, peak CUDA memory, tensor shapes). That
report is what the node turns into ``metadata.json``.

The worker never talks to ComfyUI: everything is exchanged through files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKER_DIR.parent
UPSTREAM_PARENT = PROJECT_ROOT / "external"


def _bootstrap_path() -> None:
    """Make the upstream package importable.

    The upstream repository *is* the ``trackeverything`` package, so its
    parent directory has to be on ``sys.path`` (upstream README, "Installation").
    """
    parent = str(UPSTREAM_PARENT)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _upstream_commit() -> str:
    """Best-effort read of the pinned upstream commit.

    Handles both a plain checkout and a Git submodule (where ``.git`` is a
    file containing a *relative* ``gitdir:`` pointer).
    """
    repo = UPSTREAM_PARENT / "trackeverything"
    git_dir = repo / ".git"
    try:
        if git_dir.is_file():          # submodule: .git is a pointer file
            pointer = git_dir.read_text().strip()
            if pointer.startswith("gitdir:"):
                target = pointer.split("gitdir:", 1)[1].strip()
                git_dir = Path(target)
                if not git_dir.is_absolute():
                    git_dir = (repo / git_dir).resolve()
        head = git_dir / "HEAD"
        if head.is_file():
            ref = head.read_text().strip()
            if ref.startswith("ref:"):
                ref_name = ref.split(" ", 1)[1].strip()
                ref_file = git_dir / ref_name
                if ref_file.is_file():
                    return ref_file.read_text().strip()
                return ref_name
            return ref
    except Exception:
        pass
    return "unknown"


def _versions() -> dict:
    import torch

    versions = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "upstream_commit": _upstream_commit(),
    }
    try:
        versions["cuda_device"] = torch.cuda.get_device_name(0)
        versions["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        versions["cuda_available"] = False

    for name in ("numpy", "transformers", "huggingface_hub", "torch_scatter"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def _cuda_report() -> dict:
    import torch

    report: dict = {}
    try:
        if torch.cuda.is_available():
            report["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            report["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            report["device_total_bytes"] = int(total)
            report["device_free_bytes"] = int(free)
    except Exception as exc:
        report["error"] = repr(exc)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TrackEverything inference worker (isolated environment)"
    )
    parser.add_argument("--frames-dir", required=True,
                        help="directory with input frames (PNG/JPG, sorted by name)")
    parser.add_argument("--output-dir", required=True,
                        help="directory that receives tracks.pt and geometry.pt")
    parser.add_argument("--model", default="katefgroup/TrackEverything")
    parser.add_argument("--pi3-checkpoint", default="yyfz233/Pi3")
    parser.add_argument("--geometry-backend", default="pi3",
                        choices=["pi3", "vggt_omega"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="dense", choices=["dense", "sparse"])
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--mem-efficient", dest="mem_efficient",
                        action="store_true", default=True)
    parser.add_argument("--no-mem-efficient", dest="mem_efficient",
                        action="store_false")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    _bootstrap_path()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    if not frames_dir.is_dir():
        print(f"ERROR: frames directory {frames_dir} does not exist", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    report: dict = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "versions": _versions(),
        "parameters": {
            "model": args.model,
            "pi3_checkpoint": args.pi3_checkpoint,
            "geometry_backend": args.geometry_backend,
            "device": args.device,
            "mode": args.mode,
            "num_iters": args.num_iters,
            "voxel_size": args.voxel_size,
            "mem_efficient": args.mem_efficient,
            "frame_stride": args.frame_stride,
            "start_frame": args.start_frame,
            "max_frames": args.max_frames,
            "frames_dir": str(frames_dir),
            "output_dir": str(output_dir),
        },
    }

    try:
        from trackeverything.demo_hf import TrackEverything, save_outputs
    except Exception as exc:
        print(f"ERROR: cannot import the upstream 'trackeverything' package: {exc!r}",
              file=sys.stderr)
        print(f"       PYTHONPATH={os.environ.get('PYTHONPATH')}", file=sys.stderr)
        return 3

    try:
        t0 = time.time()
        model = TrackEverything.from_pretrained(
            args.model,
            device=args.device,
            geometry_backend=args.geometry_backend,
            pi3_checkpoint=args.pi3_checkpoint,
        )
        report["timings"] = {"load_model_seconds": round(time.time() - t0, 3)}

        t0 = time.time()
        output = model.infer(
            video=str(frames_dir),
            mode=args.mode,
            num_iters=args.num_iters,
            voxel_size=args.voxel_size,
            mem_efficient=args.mem_efficient,
            frame_stride=args.frame_stride,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
        )
        report.setdefault("timings", {})["inference_seconds"] = round(time.time() - t0, 3)

        t0 = time.time()
        save_outputs(output, output_dir, save_dense=False)
        report.setdefault("timings", {})["save_seconds"] = round(time.time() - t0, 3)

    except Exception as exc:  # noqa: BLE001 - reported to the node
        import traceback

        traceback.print_exc()
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        report["cuda"] = _cuda_report()
        (output_dir / "worker_report.json").write_text(json.dumps(report, indent=2))
        print(f"ERROR: TrackEverything inference failed: {exc!r}", file=sys.stderr)
        return 1

    # -- collect the facts the node needs --------------------------------
    try:
        import torch

        tracks_path = output_dir / "tracks.pt"
        tracks = torch.load(tracks_path, map_location="cpu", weights_only=True)
        coords = tracks.get("coords")
        cell_to_token = tracks.get("cell_to_token")
        report["results"] = {
            "num_tracks": int(coords.shape[0]) if coords is not None else None,
            "num_frames": int(coords.shape[1]) if coords is not None else None,
            "feature_grid_hw": (
                [int(cell_to_token.shape[1]), int(cell_to_token.shape[2])]
                if cell_to_token is not None else None
            ),
            "working_resolution_hw": (
                [int(output.depths.shape[1]), int(output.depths.shape[2])]
                if getattr(output, "depths", None) is not None else None
            ),
            "norm_factor": float(getattr(output, "norm_factor", 1.0)),
            "voxel_size": float(getattr(output, "voxel_size", args.voxel_size)),
            "dynamic_threshold": float(getattr(output, "dynamic_threshold", 0.5)),
        }
    except Exception as exc:  # noqa: BLE001
        report["results_error"] = repr(exc)

    report["cuda"] = _cuda_report()
    report["wall_clock_seconds"] = round(time.time() - started, 3)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    (output_dir / "worker_report.json").write_text(json.dumps(report, indent=2))
    print(f"[worker] done in {report['wall_clock_seconds']:.1f}s -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
