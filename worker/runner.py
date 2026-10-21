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
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Optional

WORKER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKER_DIR.parent
UPSTREAM_PARENT = PROJECT_ROOT / "external"

if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from console import Console, human_bytes, install_upstream_logging  # noqa: E402


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


# ---------------------------------------------------------------------------
# console helpers
# ---------------------------------------------------------------------------
def _synchronize() -> None:
    """Wait for the CUDA queue so the measured times are real."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _device_name(device: str) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return device


def _count_frames(frames_dir: Path) -> int:
    try:
        return sum(
            1 for p in frames_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
    except OSError:
        return 0


def _effective_frames(total: int, start_frame: int, frame_stride: int,
                      max_frames: Optional[int]) -> int:
    """How many frames the upstream will actually process.

    Mirrors the upstream slicing: ``slice(start_frame, None, frame_stride)``
    followed by ``[:max_frames]``.
    """
    if total <= 0:
        return 0
    count = len(range(start_frame, total, max(1, frame_stride)))
    if max_frames is not None:
        count = min(count, max_frames)
    return count


def _frames_size(frames_dir: Path) -> str:
    try:
        from PIL import Image

        for path in sorted(frames_dir.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                with Image.open(path) as img:
                    return f"{img.width}x{img.height}"
    except Exception:
        pass
    return "?"


def _checkpoint_size(model_id: str) -> str:
    """How big the tracker checkpoint is on disk (cache hit or download)."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_id, filename="model.pth", local_files_only=True
        )
        return f"{human_bytes(os.path.getsize(path))} on disk"
    except Exception:
        return ""


def _load_geometry_weights(annotator) -> None:
    """Force the lazily loaded geometry weights to be read.

    The upstream annotator loads Pi3 inside ``_load_model()``, i.e. during the
    first ``process_video()`` call. Loading it here gives the geometry stage
    its own timing and keeps the inference stage free of model loading.
    """
    loader = getattr(annotator, "_load_model", None)
    if loader is not None:
        loader()


class _ChunkProgress:
    """Counts the geometry chunks so the geometry phase gets a real bar.

    The upstream annotator processes the video chunk by chunk. Wrapping the
    instance method (the upstream *source* is not modified) makes the number
    of processed chunks observable from the outside. If the upstream ever
    renames the method, the wrapper simply is not installed and the phase
    falls back to the heartbeat line.
    """

    def __init__(self, annotator, console, total_frames: int):
        self.annotator = annotator
        self.console = console
        self.total_frames = total_frames
        self.done = 0
        self.total: Optional[int] = None
        self.started = time.time()
        self._install()

    def _install(self) -> None:
        method = getattr(self.annotator, "_run_model_chunk", None)
        if method is None:
            return
        compute = getattr(self.annotator, "_compute_chunk_indices", None)
        if compute is not None:
            try:
                self.total = int(len(compute(self.total_frames)))
            except Exception:
                self.total = None
        console = self.console

        def wrapper(*args, **kwargs):
            result = method(*args, **kwargs)
            self.done += 1
            self._report()
            return result

        self.annotator._run_model_chunk = wrapper

    def _report(self) -> None:
        elapsed = time.time() - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        if self.total:
            eta = ((self.total - self.done) / rate) if rate > 0 else None
            # a permanent line per chunk: it stays readable in the log file
            # and in the ComfyUI console
            self.console.detail(
                f"chunk {self.done}/{self.total}   "
                f"elapsed {elapsed:6.1f}s   {rate:.2f} chunks/s"
                + (f"   ETA {eta:.0f}s" if eta else "")
            )
        else:
            self.console.live(
                f"{'geometry':<26s} chunk {self.done}   "
                f"elapsed {elapsed:6.1f}s"
            )


def _heartbeat(console: Console, stop: threading.Event, phase: dict,
               num_frames: int):
    """Print an elapsed-time / rate line while a long call is running."""
    while not stop.wait(1.0):
        elapsed = time.time() - phase["started"]
        rate = num_frames / elapsed if elapsed > 0 and num_frames else 0.0
        console.live(
            f"{phase['label']:<26s} elapsed {elapsed:6.1f}s"
            + (f"   {rate:5.2f} frames/s" if rate else "")
        )



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

    # -- console reporting ------------------------------------------------
    console = Console()

    phase = {"label": "inference", "started": time.time()}

    def _on_upstream_message(message: str) -> None:
        """Turn the upstream log messages into visible sub-stages."""
        if "Estimating geometry with" in message:
            console.substage("geometry estimation")
            phase["label"] = "geometry"
            phase["started"] = time.time()
        elif "Running model inference" in message:
            console.substage(f"tracking ({args.mode})")
            phase["label"] = "tracking"
            phase["started"] = time.time()

    install_upstream_logging(console, on_message=_on_upstream_message)

    total_frames = _count_frames(frames_dir)
    num_frames = _effective_frames(
        total_frames, args.start_frame, args.frame_stride, args.max_frames
    )
    console.banner(
        "TrackEverything - standalone inference",
        {
            "device": _device_name(args.device),
            "model": args.model,
            "geometry": args.geometry_backend,
            "pi3": args.pi3_checkpoint,
            "frames": (
                f"{num_frames} of {total_frames} @ {_frames_size(frames_dir)}"
                if num_frames != total_frames
                else f"{num_frames} @ {_frames_size(frames_dir)}"
            ),
            "mode": f"{args.mode}, {args.num_iters} iterations",
            "voxel size": args.voxel_size,
            "mem efficient": args.mem_efficient,
            "output": str(output_dir),
        },
    )

    total_stages = 4

    try:
        # -- stage 1: the tracker checkpoint -----------------------------
        console.stage(1, total_stages, "loading the TrackEverything checkpoint")
        t0 = time.time()
        model = TrackEverything.from_pretrained(
            args.model,
            device=args.device,
            geometry_backend=args.geometry_backend,
            pi3_checkpoint=args.pi3_checkpoint,
        )
        _synchronize()
        load_seconds = time.time() - t0
        report["timings"] = {"load_model_seconds": round(load_seconds, 3)}
        console.stage_done(load_seconds, _checkpoint_size(args.model))

        # -- stage 2: the geometry backend --------------------------------
        console.stage(2, total_stages, f"loading the {args.geometry_backend} geometry backend")
        t0 = time.time()
        annotator = model.geometry
        _load_geometry_weights(annotator)
        _synchronize()
        geometry_seconds = time.time() - t0
        report["timings"]["load_geometry_seconds"] = round(geometry_seconds, 3)
        console.stage_done(geometry_seconds)

        # -- stage 3: geometry estimation and tracking --------------------
        console.stage(3, total_stages, "estimating geometry and tracking points")
        t0 = time.time()
        # real progress for the geometry phase (chunk by chunk)
        chunk_progress = _ChunkProgress(annotator, console, num_frames)
        phase["label"] = "inference"
        phase["started"] = t0
        stop = threading.Event()
        beat = threading.Thread(
            target=_heartbeat,
            args=(console, stop, phase, num_frames),
            daemon=True,
        )
        beat.start()
        try:
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
        finally:
            stop.set()
            beat.join(timeout=2.0)
        _synchronize()
        infer_seconds = time.time() - t0
        report.setdefault("timings", {})["inference_seconds"] = round(infer_seconds, 3)
        rate = (num_frames / infer_seconds) if infer_seconds > 0 else 0.0
        console.stage_done(
            infer_seconds,
            f"{num_frames} frames, {rate:.2f} frames/s"
            + (f", {chunk_progress.total} geometry chunk(s)"
               if chunk_progress.total else ""),
        )

        # -- stage 4: saving ------------------------------------------------
        console.stage(4, total_stages, "saving tracks.pt and geometry.pt")
        t0 = time.time()
        save_outputs(output, output_dir, save_dense=False)
        save_seconds = time.time() - t0
        report.setdefault("timings", {})["save_seconds"] = round(save_seconds, 3)
        written = sum(
            p.stat().st_size for p in output_dir.glob("*.pt") if p.is_file()
        )
        console.stage_done(save_seconds, f"{human_bytes(written)} written")

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

    cuda = report.get("cuda", {})
    allocated = cuda.get("max_memory_allocated_bytes")
    reserved = cuda.get("max_memory_reserved_bytes")
    console.summary(
        {
            "result": str(output_dir),
            "tracks": report.get("results", {}).get("num_tracks"),
            "frames": report.get("results", {}).get("num_frames"),
            "total": f"{report['wall_clock_seconds']:.2f} s",
            "peak VRAM": (
                f"{human_bytes(allocated)} allocated / {human_bytes(reserved)} reserved"
                if allocated is not None else "n/a"
            ),
        }
    )
    print(f"[worker] done in {report['wall_clock_seconds']:.1f}s -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
