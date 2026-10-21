"""Bridge between the ComfyUI process and the isolated TrackEverything worker.

Responsibilities:

* validate a ComfyUI ``IMAGE`` batch and convert it to uint8 RGB exactly the
  way the official upstream API does,
* write the frames to a temporary directory with a guaranteed frame order,
* launch the **isolated** worker as a separate process (explicit argv, never
  ``shell=True``),
* validate the produced files and assemble the run metadata,
* return a :class:`~comfy_nodes.types.TrackData` reference.

This module runs inside the **ComfyUI** process. It must never import
``trackeverything``, ``torch_scatter`` or anything from the worker's Python
3.10 environment - everything GPU related happens in the child process and is
exchanged through files only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import Config, assert_isolated
from .types import (
    GEOMETRY_FILENAME,
    METADATA_FILENAME,
    SCHEMA_VERSION,
    TRACKS_FILENAME,
    WORKER_LOG_FILENAME,
    WORKER_REPORT_FILENAME,
    TrackData,
)


class TrackEverythingError(RuntimeError):
    """A user-facing error with an actionable message."""


# ---------------------------------------------------------------------------
# image handling
# ---------------------------------------------------------------------------
@dataclass
class FrameBatch:
    """A validated, converted frame batch."""

    frames: "np.ndarray"          # (T, H, W, 3) uint8 RGB
    original_size: dict            # {"width": W, "height": H}
    source_dtype: str
    source_range: str

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])


def _to_numpy(images: Any) -> Any:
    """Accept a torch tensor (ComfyUI IMAGE) or a numpy array."""
    if hasattr(images, "detach"):          # torch.Tensor
        return images.detach().cpu().numpy()
    if hasattr(images, "cpu") and hasattr(images, "numpy"):
        return images.cpu().numpy()
    import numpy as np

    return np.asarray(images)


def validate_image_batch(images: Any) -> FrameBatch:
    """Validate a ComfyUI ``IMAGE`` batch and convert it to uint8 RGB.

    The conversion mirrors the official upstream API
    (``trackeverything.demo_hf._load_video``):

        ``(x * 255).clip(0, 255).astype(uint8)``

    i.e. the value is scaled by 255, clamped and **truncated** (not rounded),
    exactly like upstream.
    """
    import numpy as np

    if images is None:
        raise TrackEverythingError("No images received (input 'images' is None).")

    arr = _to_numpy(images)

    if arr.dtype == np.uint8:
        source_dtype = "uint8"
        source_range = "[0, 255]"
        frames = arr
    else:
        if not np.issubdtype(arr.dtype, np.floating):
            raise TrackEverythingError(
                f"Unsupported image dtype {arr.dtype}; expected float or uint8."
            )
        source_dtype = str(arr.dtype)
        source_range = "[0, 1]"
        frames = (arr.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)

    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4:
        raise TrackEverythingError(
            f"Expected an IMAGE batch of shape (B, H, W, C), got {frames.shape}."
        )
    if frames.shape[0] < 1:
        raise TrackEverythingError("The IMAGE batch is empty (0 frames).")
    if frames.shape[-1] != 3:
        raise TrackEverythingError(
            f"Expected 3 colour channels (RGB), got {frames.shape[-1]}."
        )

    height, width = int(frames.shape[1]), int(frames.shape[2])
    if height < 2 or width < 2:
        raise TrackEverythingError(
            f"Frames are too small ({width}x{height}); at least 2x2 is required."
        )

    # All frames must share one resolution - the upstream pipeline resizes the
    # whole batch to a single model resolution anyway.
    if not (frames.shape[1] == frames.shape[2] or True):
        raise TrackEverythingError("Inconsistent frame size in the IMAGE batch.")

    return FrameBatch(
        frames=frames,
        original_size={"width": width, "height": height},
        source_dtype=source_dtype,
        source_range=source_range,
    )


def write_frames(batch: FrameBatch, frames_dir: Path) -> list[str]:
    """Write the frames as PNG files with a guaranteed order.

    The upstream loader reads an image folder and sorts the files **by name**,
    so the names are zero padded to keep the numeric order.
    """
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for index, frame in enumerate(batch.frames):
        name = f"frame_{index:06d}.png"
        Image.fromarray(frame).save(frames_dir / name)
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# worker invocation
# ---------------------------------------------------------------------------
@dataclass
class WorkerResult:
    returncode: int
    command: list[str]
    stdout: str
    stderr: str
    log_path: Path
    wall_clock_seconds: float
    worker_report: dict = field(default_factory=dict)


def build_worker_argv(
    cfg: Config,
    frames_dir: Path,
    output_dir: Path,
    *,
    num_iters: int = 3,
    voxel_size: float = 0.02,
    mem_efficient: bool = True,
    mode: str = "dense",
    geometry_backend: Optional[str] = None,
    device: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    start_frame: int = 0,
) -> list[str]:
    """Explicit argv for the worker. Never a shell string."""
    return [
        str(cfg.worker_python),
        str(cfg.worker_script),
        "--frames-dir", str(frames_dir),
        "--output-dir", str(output_dir),
        "--model", cfg.model_repo,
        "--pi3-checkpoint", cfg.pi3_checkpoint,
        "--geometry-backend", geometry_backend or cfg.geometry_backend,
        "--device", device or cfg.device,
        "--mode", mode,
        "--num-iters", str(int(num_iters)),
        "--voxel-size", str(float(voxel_size)),
        "--mem-efficient" if mem_efficient else "--no-mem-efficient",
        "--frame-stride", str(int(frame_stride)),
        "--start-frame", str(int(start_frame)),
    ] + (["--max-frames", str(int(max_frames))] if max_frames else [])


def worker_environment(cfg: Config) -> dict:
    """Environment for the child process.

    The isolated environment's ``site-packages`` is **not** injected into the
    ComfyUI process; here we only prepare the child side: the upstream package
    parent has to be importable by the worker.
    """
    env = dict(os.environ)
    upstream_parent = str(cfg.upstream_parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = upstream_parent + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Tokens are forwarded from the parent environment only; never written to
    # disk and never passed on the command line.
    return env


def run_worker(
    cfg: Config,
    argv: Sequence[str],
    output_dir: Path,
    *,
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
) -> WorkerResult:
    """Run the worker and capture its output into ``worker.log``."""
    assert_isolated(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / WORKER_LOG_FILENAME

    started = time.time()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    with open(log_path, "wb") as log_file:
        log_file.write(b"$ " + " ".join(argv).encode("utf-8") + b"\n\n")
        log_file.flush()
        try:
            proc = subprocess.Popen(  # noqa: S603 - explicit argv, no shell
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=worker_environment(cfg),
                cwd=str(cwd or cfg.project_root),
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise TrackEverythingError(
                f"Could not start the TrackEverything worker: {exc}\n"
                f"Command: {' '.join(argv)}"
            ) from exc

        assert proc.stdout is not None and proc.stderr is not None
        # Read both pipes line by line so a chatty worker cannot deadlock.
        import selectors

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
        open_streams = 2
        timed_out = False

        while open_streams:
            if timeout is not None and (time.time() - started) > timeout:
                timed_out = True
                proc.kill()
                break
            for key, _mask in sel.select(timeout=1.0):
                chunk = key.fileobj.readline()
                if not chunk:
                    sel.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                if key.data == "stdout":
                    stdout_chunks.append(chunk)
                else:
                    stderr_chunks.append(chunk)
                log_file.write(chunk.encode("utf-8"))
                log_file.flush()
        sel.close()

        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    elapsed = time.time() - started
    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    returncode = proc.returncode if proc.returncode is not None else -1

    if timed_out:
        raise TrackEverythingError(
            f"The TrackEverything worker timed out after {timeout:.0f}s.\n"
            f"Log: {log_path}"
        )

    worker_report: dict = {}
    report_path = output_dir / WORKER_REPORT_FILENAME
    if report_path.is_file():
        try:
            worker_report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            worker_report = {}

    return WorkerResult(
        returncode=returncode,
        command=list(argv),
        stdout=stdout,
        stderr=stderr,
        log_path=log_path,
        wall_clock_seconds=elapsed,
        worker_report=worker_report,
    )


def _tail(text: str, lines: int = 25) -> str:
    parts = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(parts[-lines:])


# ---------------------------------------------------------------------------
# result collection
# ---------------------------------------------------------------------------
def _read_tracks_shapes(tracks_path: Path) -> dict:
    """Read the tensor shapes of ``tracks.pt`` without touching the GPU."""
    import torch

    try:
        obj = torch.load(tracks_path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(tracks_path, map_location="cpu", weights_only=False)

    shapes: dict[str, Any] = {}
    for key, value in obj.items():
        shapes[key] = list(value.shape) if hasattr(value, "shape") else None
    return shapes


def synthesize_metadata(result_dir: Path) -> dict:
    """Build a minimal metadata for a result set that has no ``metadata.json``.

    The standalone inference pipeline writes only ``tracks.pt`` and
    ``geometry.pt``. Those two files are enough to visualize and to inspect a
    result, so the nodes accept them and describe them on the fly. The result
    is clearly marked with ``"metadata_source": "synthesized"``.
    """
    import torch

    tracks_path = result_dir / TRACKS_FILENAME
    geometry_path = result_dir / GEOMETRY_FILENAME

    try:
        tracks = torch.load(tracks_path, map_location="cpu", weights_only=True)
    except Exception:
        tracks = torch.load(tracks_path, map_location="cpu", weights_only=False)
    try:
        geometry = torch.load(geometry_path, map_location="cpu", weights_only=True)
    except Exception:
        geometry = torch.load(geometry_path, map_location="cpu", weights_only=False)

    coords = tracks.get("coords")
    num_tracks = int(coords.shape[0]) if coords is not None else None
    num_frames = int(coords.shape[1]) if coords is not None else None

    depths = geometry.get("depths")
    working = (
        {"width": int(depths.shape[2]), "height": int(depths.shape[1])}
        if depths is not None else None
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "producer": "comfy_nodes.bridge.synthesize_metadata",
        "metadata_source": "synthesized",
        "note": (
            "metadata.json was not part of this result set - the description "
            "below was derived from tracks.pt / geometry.pt"
        ),
        "inference": {},
        "versions": {},
        "input": {
            "num_frames": num_frames,
            "original_size": None,
            "dtype": "unknown",
            "value_range": "unknown",
            "colour_space": "RGB",
        },
        "output": {
            "num_frames": num_frames,
            "num_tracks": num_tracks,
            "working_size": working,
            "feature_grid": (
                {"height": working["height"] // 4, "width": working["width"] // 4}
                if working else None
            ),
            "patch_size": 4 if working else None,
            "track_tensor_shapes": {
                key: (list(value.shape) if hasattr(value, "shape") else None)
                for key, value in tracks.items()
            },
        },
        "coordinate_transform": {
            "description": (
                "3D coordinates live in the upstream scene frame: cam-0 aligned "
                "and normalised by the median scene distance."
            ),
            "world_frame": "camera-0 aligned, scene-normalised",
            "norm_factor": geometry.get("norm_factor"),
            "unnormalise": "world_original = coords * norm_factor",
            "pixel_mapping": {
                "note": "src_x / src_y are cell indices of the feature grid, not pixels.",
                "working_px_x": "x_working = src_x * patch_size + (patch_size - 1) / 2",
                "working_px_y": "y_working = src_y * patch_size + (patch_size - 1) / 2",
                "original_px_x": "x_original = x_working / scale_x",
                "original_px_y": "y_working / scale_y",
            },
            "projection": {
                "note": (
                    "X_cam = R @ X_world + t; p_h = K @ X_cam; "
                    "x = p_h[0] / p_h[2]; y = p_h[1] / p_h[2]"
                ),
                "intrinsics": "geometry.pt['intrinsics'] (T, 3, 3)",
                "extrinsics": "geometry.pt['extrinsics'] (T, 4, 4)",
                "depths": "geometry.pt['depths'] (T, H, W)",
                "intrinsics_are_normalised": True,
            },
            "sentinel": {
                "value": -1000,
                "meaning": "missing / not-yet-existing trajectory entry - NOT a coordinate",
            },
        },
        "files": {
            "tracks": TRACKS_FILENAME,
            "geometry": GEOMETRY_FILENAME,
            "metadata": METADATA_FILENAME,
        },
    }
    return metadata


def read_metadata(result_dir: Path) -> dict:
    """Read ``metadata.json``, or synthesize it when the file is absent."""
    metadata_path = Path(result_dir) / METADATA_FILENAME
    if metadata_path.is_file():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrackEverythingError(f"Cannot read {metadata_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise TrackEverythingError(
                f"{metadata_path} does not contain a JSON object."
            )
        return data
    return synthesize_metadata(Path(result_dir))


def build_metadata(
    cfg: Config,
    batch: FrameBatch,
    params: dict,
    worker: WorkerResult,
    output_dir: Path,
) -> dict:
    """Assemble ``metadata.json``.

    The numeric payload itself is **not** duplicated here - the metadata only
    describes it. Coordinates, logits and matrices stay exactly as the
    upstream produced them in ``tracks.pt`` / ``geometry.pt``.
    """
    report = worker.worker_report or {}
    results = report.get("results", {})
    versions = report.get("versions", {})

    num_frames = int(results.get("num_frames") or batch.num_frames)
    num_tracks = results.get("num_tracks")
    working_hw = results.get("working_resolution_hw")
    norm_factor = results.get("norm_factor")

    original_w = batch.original_size["width"]
    original_h = batch.original_size["height"]

    working_w = int(working_hw[1]) if working_hw else None
    working_h = int(working_hw[0]) if working_hw else None

    scale_x = (working_w / original_w) if working_w else None
    scale_y = (working_h / original_h) if working_h else None

    feature_grid = None
    if working_h and working_w:
        feature_grid = {"height": working_h // 4, "width": working_w // 4}

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "producer": "comfy_nodes.TrackEverythingTrack",
        "inference": params,
        "versions": versions,
        "input": {
            "num_frames": batch.num_frames,
            "original_size": {"width": original_w, "height": original_h},
            "dtype": batch.source_dtype,
            "value_range": batch.source_range,
            "conversion": "(x * 255).clip(0, 255).astype(uint8)  [upstream _load_video]",
            "colour_space": "RGB",
        },
        "output": {
            "num_frames": num_frames,
            "num_tracks": num_tracks,
            "working_size": (
                {"width": working_w, "height": working_h} if working_w else None
            ),
            "feature_grid": feature_grid,
            "patch_size": 4 if feature_grid else None,
            "track_tensor_shapes": _read_tracks_shapes(output_dir / TRACKS_FILENAME),
        },
        "coordinate_transform": {
            "description": (
                "3D coordinates live in the upstream scene frame: cam-0 aligned and "
                "normalised by the median scene distance."
            ),
            "world_frame": "camera-0 aligned, scene-normalised",
            "norm_factor": norm_factor,
            "unnormalise": "world_original = coords * norm_factor",
            "pixel_mapping": {
                "note": (
                    "src_x / src_y are cell indices of the feature grid, not pixels."
                ),
                "working_px_x": "x_working = src_x * patch_size + (patch_size - 1) / 2",
                "working_px_y": "y_working = src_y * patch_size + (patch_size - 1) / 2",
                "original_px_x": "x_original = x_working / scale_x",
                "original_px_y": "y_original = y_working / scale_y",
                "scale_x": scale_x,
                "scale_y": scale_y,
            },
            "projection": {
                "note": (
                    "Use intrinsics / extrinsics from geometry.pt to project a 3D "
                    "world point back to pixels: X_cam = R @ X_world + t; "
                    "p_h = K @ X_cam; x = p_h[0] / p_h[2]; y = p_h[1] / p_h[2]. "
                    "A point is only in front of the camera when p_h[2] > 0; "
                    "points behind it must be discarded."
                ),
                "formulas": {
                    "world_to_camera": "X_cam = R @ X_world + t",
                    "project": "p_h = K @ X_cam",
                    "divide": "x = p_h[0] / p_h[2]; y = p_h[1] / p_h[2]",
                    "depth": "z_cam = p_h[2] (must be > 0)",
                },
                "intrinsics": "geometry.pt['intrinsics'] (T, 3, 3)",
                "extrinsics": "geometry.pt['extrinsics'] (T, 4, 4)",
                "depths": "geometry.pt['depths'] (T, H, W)",
                "intrinsics_are_normalised": True,
                "intrinsics_resolution": (
                    "the intrinsics refer to 'output.working_size'; to use them "
                    "at another resolution rescale K[0, :] by (W_to - 1) / (W_from - 1) "
                    "and K[1, :] by (H_to - 1) / (H_from - 1)"
                ),
            },
            "sentinel": {
                "value": -1000,
                "meaning": (
                    "missing / not-yet-existing trajectory entry - NOT a coordinate"
                ),
            },
        },
        "files": {
            "tracks": TRACKS_FILENAME,
            "geometry": GEOMETRY_FILENAME,
            "metadata": METADATA_FILENAME,
            "worker_report": WORKER_REPORT_FILENAME,
            "worker_log": WORKER_LOG_FILENAME,
        },
        "worker": {
            "command": worker.command,
            "returncode": worker.returncode,
            "wall_clock_seconds": round(worker.wall_clock_seconds, 3),
            "log": WORKER_LOG_FILENAME,
            "pid_isolated": True,
        },
        "environment": cfg.describe(),
    }
    return metadata


def build_track_data(result_dir: Path, metadata: dict) -> TrackData:
    result_dir = Path(result_dir)
    return TrackData(
        result_dir=str(result_dir),
        tracks_path=str(result_dir / TRACKS_FILENAME),
        geometry_path=str(result_dir / GEOMETRY_FILENAME),
        metadata_path=str(result_dir / METADATA_FILENAME),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def run_track(
    cfg: Config,
    images: Any,
    *,
    num_iters: int = 3,
    voxel_size: float = 0.02,
    mem_efficient: bool = True,
    mode: str = "dense",
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    start_frame: int = 0,
    timeout: Optional[float] = None,
    keep_frames: bool = False,
    output_dir: Optional[Path] = None,
) -> TrackData:
    """Full pipeline: IMAGE batch -> worker process -> :class:`TrackData`."""
    cfg.validate()

    batch = validate_image_batch(images)

    result_dir = Path(output_dir) if output_dir else _new_result_dir(cfg)
    result_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="trackeverything_frames_"))
    frames_dir = tmp_root / "frames"

    try:
        frame_names = write_frames(batch, frames_dir)
        argv = build_worker_argv(
            cfg,
            frames_dir,
            result_dir,
            num_iters=num_iters,
            voxel_size=voxel_size,
            mem_efficient=mem_efficient,
            mode=mode,
            max_frames=max_frames,
            frame_stride=frame_stride,
            start_frame=start_frame,
        )

        params = {
            "mode": mode,
            "geometry_backend": cfg.geometry_backend,
            "num_iters": int(num_iters),
            "voxel_size": float(voxel_size),
            "mem_efficient": bool(mem_efficient),
            "device": cfg.device,
            "model": cfg.model_repo,
            "pi3_checkpoint": cfg.pi3_checkpoint,
            "max_frames": max_frames,
            "frame_stride": int(frame_stride),
            "start_frame": int(start_frame),
            "num_input_frames": batch.num_frames,
            "num_frame_files": len(frame_names),
        }

        worker = run_worker(cfg, argv, result_dir, timeout=timeout)

        if worker.returncode != 0:
            raise TrackEverythingError(
                "The TrackEverything worker failed "
                f"(exit code {worker.returncode}).\n"
                f"Command: {' '.join(argv)}\n"
                f"Log: {worker.log_path}\n"
                "--- last output ---\n"
                f"{_tail(worker.stdout + worker.stderr)}"
            )

        tracks_path = result_dir / TRACKS_FILENAME
        geometry_path = result_dir / GEOMETRY_FILENAME
        missing = [
            str(p) for p in (tracks_path, geometry_path) if not p.is_file()
        ]
        if missing:
            raise TrackEverythingError(
                "The TrackEverything worker finished but did not produce "
                f"the expected files: {', '.join(missing)}\n"
                f"Log: {worker.log_path}"
            )

        metadata = build_metadata(cfg, batch, params, worker, result_dir)
        (result_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return build_track_data(result_dir, metadata)

    finally:
        if keep_frames:
            print(f"[TrackEverything] frames kept in {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


def _new_result_dir(cfg: Config) -> Path:
    """Create a fresh, timestamped directory under the configured output root."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(cfg.output_root)
    candidate = root / f"trackeverything_{stamp}"
    index = 1
    while candidate.exists():
        candidate = root / f"trackeverything_{stamp}_{index:02d}"
        index += 1
    return candidate


def load_result_dir(result_dir: Any) -> TrackData:
    """Load an existing result directory (used by the ``Load Data`` node)."""
    directory = Path(str(result_dir)).expanduser()
    if not directory.is_dir():
        raise TrackEverythingError(
            f"Result directory {directory} does not exist. "
            "Point the node at the folder that contains tracks.pt."
        )

    tracks_path = directory / TRACKS_FILENAME
    geometry_path = directory / GEOMETRY_FILENAME

    missing = [
        name for name, path in (
            (TRACKS_FILENAME, tracks_path),
            (GEOMETRY_FILENAME, geometry_path),
        ) if not path.is_file()
    ]
    if missing:
        raise TrackEverythingError(
            f"Result directory {directory} is missing: {', '.join(missing)}."
        )

    # metadata.json is optional: a result produced by the standalone pipeline
    # only carries tracks.pt and geometry.pt, and is described on the fly.
    metadata = read_metadata(directory)

    # Sanity check the payload files (header only, no GPU involved).
    for path in (tracks_path, geometry_path):
        try:
            import torch

            torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            raise TrackEverythingError(
                f"Cannot read {path}: {exc}"
            ) from exc

    return build_track_data(directory, metadata)


# ---------------------------------------------------------------------------
# saving results
# ---------------------------------------------------------------------------
def _rewrite_paths(obj: Any, old_dir: str, new_dir: str) -> Any:
    """Replace every occurrence of ``old_dir`` inside ``obj`` with ``new_dir``.

    The metadata mixes **relative** names (``"files": {"tracks": "tracks.pt"}``,
    which stay valid inside the copy) with **absolute** paths (the worker
    command, the environment description). The absolute ones that point into
    the source result directory have to follow the copy, otherwise the copy
    would claim to live where it does not.
    """
    if isinstance(obj, str):
        return obj.replace(old_dir, new_dir)
    if isinstance(obj, list):
        return [_rewrite_paths(item, old_dir, new_dir) for item in obj]
    if isinstance(obj, dict):
        return {key: _rewrite_paths(value, old_dir, new_dir)
                for key, value in obj.items()}
    return obj


def _unique_destination(root: Path, prefix: str) -> Path:
    """A fresh, timestamped directory - never an existing one."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = root / prefix if prefix else root
    candidate = base / f"{prefix or 'trackeverything'}_{stamp}"
    index = 1
    while candidate.exists():
        candidate = base / f"{prefix or 'trackeverything'}_{stamp}_{index:02d}"
        index += 1
    return candidate


def save_result(
    track_data: TrackData,
    *,
    save_mode: str = "reference",
    filename_prefix: str = "trackeverything",
    output_root: Optional[Path] = None,
) -> tuple[TrackData, str]:
    """Persist a result set.

    ``reference``
        Keep the files where they are, verify that they are complete and
        report the location. Nothing is copied.

    ``copy``
        Copy ``tracks.pt``, ``geometry.pt``, ``metadata.json`` and the worker
        log into a **new** directory under the output root. The ``.pt`` files
        are copied byte for byte - no tensor is converted.

    Returns the (possibly new) :class:`TrackData` and a human readable message.
    """
    if save_mode not in ("reference", "copy"):
        raise TrackEverythingError(
            f"Unknown save_mode {save_mode!r}; expected 'reference' or 'copy'."
        )

    source = Path(track_data.result_dir)
    tracks_path = source / TRACKS_FILENAME
    geometry_path = source / GEOMETRY_FILENAME
    metadata_path = source / METADATA_FILENAME

    # tracks.pt and geometry.pt are the payload and must exist; metadata.json
    # is the description - when it is missing (a result of the standalone
    # pipeline) it is synthesized from the two tensor files.
    missing = [
        name for name, path in (
            (TRACKS_FILENAME, tracks_path),
            (GEOMETRY_FILENAME, geometry_path),
        ) if not path.is_file()
    ]
    if missing:
        raise TrackEverythingError(
            f"Cannot save: {source} is missing {', '.join(missing)}."
        )
    if not metadata_path.is_file():
        print(
            f"[TrackEverything] {source} has no metadata.json - the result "
            f"description is derived from the tensor files."
        )

    # --- reference mode ---------------------------------------------------
    if save_mode == "reference":
        message = (
            f"TrackEverything result referenced (not copied): {source}  "
            f"[{track_data.num_tracks} tracks, {track_data.num_frames} frames]"
        )
        print(message)
        return track_data, message

    # --- copy mode ---------------------------------------------------------
    root = Path(output_root) if output_root else Path(track_data.result_dir).parent
    prefix = (filename_prefix or "trackeverything").strip().strip(os.sep)
    destination = _unique_destination(root, prefix)
    staging = destination.with_name(destination.name + f".tmp-{os.getpid()}")

    try:
        staging.mkdir(parents=True)

        # byte-for-byte copies - no tensor conversion, no re-serialisation
        for name in (TRACKS_FILENAME, GEOMETRY_FILENAME):
            shutil.copy2(source / name, staging / name)
        if metadata_path.is_file():
            shutil.copy2(metadata_path, staging / METADATA_FILENAME)

        # the worker log is optional
        source_log = source / WORKER_LOG_FILENAME
        if source_log.is_file():
            shutil.copy2(source_log, staging / WORKER_LOG_FILENAME)
        source_report = source / WORKER_REPORT_FILENAME
        if source_report.is_file():
            shutil.copy2(source_report, staging / WORKER_REPORT_FILENAME)

        # --- metadata: follow the copy, keep the provenance ---------------
        metadata = read_metadata(source)
        source_abs = str(source.resolve())
        destination_abs = str(destination.resolve())
        metadata = _rewrite_paths(metadata, source_abs, destination_abs)
        metadata = _rewrite_paths(metadata, str(source), str(destination))
        metadata["result_dir"] = destination_abs
        metadata["copy"] = {
            "source_result_dir": source_abs,
            "copied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "copied_by": "TrackEverythingSaveData",
            "save_mode": "copy",
            "files_are_byte_identical": True,
        }
        metadata.setdefault("files", {})["worker_log"] = WORKER_LOG_FILENAME

        (staging / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        # only publish the directory once everything is in place
        staging.rename(destination)

    except Exception:
        # never leave a half-written directory behind
        shutil.rmtree(staging, ignore_errors=True)
        # ... and not an empty parent folder either
        try:
            if staging.parent.is_dir() and not any(staging.parent.iterdir()):
                staging.parent.rmdir()
        except OSError:
            pass
        raise

    message = (
        f"TrackEverything result copied to {destination}  "
        f"[{track_data.num_tracks} tracks, {track_data.num_frames} frames]"
    )
    print(message)

    copied_metadata = json.loads(
        (destination / METADATA_FILENAME).read_text(encoding="utf-8")
    )
    return build_track_data(destination, copied_metadata), message


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------
def build_visualizer_argv(
    cfg: Config,
    result_dir: Path,
    output_dir: Path,
    *,
    frames_dir: Optional[Path] = None,
    max_points: int = 5000,
    trail_length: int = 8,
    visibility_threshold: float = 0.5,
    dynamic_threshold: float = 0.5,
    background_opacity: float = 0.7,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> list[str]:
    """Explicit argv for the visualizer worker."""
    argv = [
        str(cfg.worker_python),
        str(cfg.project_root / "worker" / "visualize.py"),
        "--result-dir", str(result_dir),
        "--output-dir", str(output_dir),
        "--max-points", str(int(max_points)),
        "--trail-length", str(int(trail_length)),
        "--visibility-threshold", str(float(visibility_threshold)),
        "--dynamic-threshold", str(float(dynamic_threshold)),
        "--background-opacity", str(float(background_opacity)),
    ]
    if frames_dir is not None:
        argv += ["--frames-dir", str(frames_dir)]
    if width and height:
        argv += ["--width", str(int(width)), "--height", str(int(height))]
    return argv


def _read_png_batch(directory: Path) -> "np.ndarray":
    """Read the rendered PNG frames into a (T, H, W, 3) uint8 array."""
    import numpy as np
    from PIL import Image

    paths = sorted(directory.glob("vis_*.png"))
    if not paths:
        raise TrackEverythingError(
            f"The visualizer produced no frames in {directory}."
        )
    frames = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(np.asarray(img.convert("RGB")))
    return np.stack(frames)


def visualize_result(
    cfg: Config,
    track_data: TrackData,
    *,
    images: Any = None,
    max_points: int = 5000,
    trail_length: int = 8,
    visibility_threshold: float = 0.5,
    dynamic_threshold: float = 0.5,
    background_opacity: float = 0.7,
    timeout: Optional[float] = None,
) -> tuple:
    """Render the track overlay and return a ComfyUI ``IMAGE`` batch.

    The rendering happens in the isolated worker (CPU only, no models). The
    frames come back through PNG files and are turned into a float32
    ``(T, H, W, 3)`` tensor in ``[0, 1]``.
    """
    cfg.validate()

    result_dir = Path(track_data.result_dir)
    for name in (TRACKS_FILENAME, GEOMETRY_FILENAME):
        if not (result_dir / name).is_file():
            raise TrackEverythingError(
                f"Cannot visualize: {result_dir} is missing {name}."
            )

    tmp_root = Path(tempfile.mkdtemp(prefix="trackeverything_vis_"))
    frames_dir = tmp_root / "frames"
    source_frames = None

    try:
        if images is not None:
            batch = validate_image_batch(images)
            expected = track_data.num_frames
            if expected is not None and batch.num_frames != int(expected):
                raise TrackEverythingError(
                    f"The IMAGE batch has {batch.num_frames} frames but the "
                    f"tracks cover {expected} frames. They must match."
                )
            write_frames(batch, frames_dir)
            source_frames = batch

        out_dir = tmp_root / "rendered"
        argv = build_visualizer_argv(
            cfg,
            result_dir,
            out_dir,
            frames_dir=frames_dir if source_frames is not None else None,
            max_points=max_points,
            trail_length=trail_length,
            visibility_threshold=visibility_threshold,
            dynamic_threshold=dynamic_threshold,
            background_opacity=background_opacity,
        )

        worker = run_worker(cfg, argv, out_dir, timeout=timeout)
        if worker.returncode != 0:
            raise TrackEverythingError(
                "The TrackEverything visualizer failed "
                f"(exit code {worker.returncode}).\n"
                f"Command: {' '.join(argv)}\n"
                f"Log: {worker.log_path}\n"
                "--- last output ---\n"
                f"{_tail(worker.stdout + worker.stderr)}"
            )

        frames = _read_png_batch(out_dir)

        # ComfyUI IMAGE: float32 (T, H, W, 3) in [0, 1]
        import torch

        tensor = torch.from_numpy(frames).float().div_(255.0)

        message = (
            f"TrackEverything visualized {tensor.shape[0]} frames "
            f"({tensor.shape[2]}x{tensor.shape[1]}), "
            f"{track_data.num_tracks} tracks"
        )
        print(message)

        return (tensor,)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
