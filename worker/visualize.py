#!/usr/bin/env python
"""TrackEverything **visualizer** worker - runs inside the isolated environment.

Reads a saved result set (``tracks.pt`` + ``geometry.pt``) and renders the
2D track overlay with the **original upstream renderer**
(``trackeverything.utils.vis_utils._render_flat_2d_tracks_frames``).

This worker deliberately

* does **not** load the TrackEverything or Pi3 weights,
* does **not** run any inference,
* runs entirely on the **CPU** - no CUDA context is created,
* does **not** modify the input ``.pt`` files.

The upstream renderer expects a ``CompressedTracks`` named tuple, while the
saved ``tracks.pt`` holds the ``SceneTokens`` fields. The two share every
field, so the adapter below is a pure field copy - no numbers are changed.

Geometry handling
-----------------
``geometry.pt`` stores the cameras at the **working** resolution (the fixed
model resolution), while the frames the user supplies are at the **original**
resolution. The intrinsics are rescaled with the exact inverse of the
upstream resize rule, so the projected points land on the right pixels of the
original frames - no silent shift.

The projection itself is the upstream one::

    X_cam = R @ X_world + t
    p_h   = K @ X_cam
    x     = p_h[0] / p_h[2]
    y     = p_h[1] / p_h[2]

Points that fall behind the camera (``p_h[2] <= 0``) would be projected with a
mirrored sign, so they are masked out (visibility logits are set to the
upstream sentinel) before rendering. The saved files are not touched - only
the in-memory copy handed to the renderer.

The rendered frames are written as PNG files; the ComfyUI side reads them
back and turns them into an ``IMAGE`` batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKER_DIR.parent
UPSTREAM_PARENT = PROJECT_ROOT / "external"

#: The upstream sentinel for "missing / not-yet-existing" trajectory entries.
SENTINEL = -1000.0


def _bootstrap_path() -> None:
    """Make the upstream package importable (parent directory on sys.path)."""
    parent = str(UPSTREAM_PARENT)
    if parent not in sys.path:
        sys.path.insert(0, parent)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_torch_dict(path: Path) -> dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def scale_intrinsics(intrinsics, from_hw: tuple[int, int], to_hw: tuple[int, int]):
    """Rescale the intrinsics between two resolutions.

    This is the exact inverse of the rule the upstream ``_preprocess`` applies
    when it resizes the video to the model resolution::

        K[0, :] *= (W_to - 1) / (W_from - 1)
        K[1, :] *= (H_to - 1) / (H_from - 1)

    ``from_hw`` is the resolution the intrinsics currently refer to,
    ``to_hw`` the resolution they should refer to.
    """
    import torch

    k = intrinsics.clone().float()
    h_from, w_from = from_hw
    h_to, w_to = to_hw
    if (h_from, w_from) == (h_to, w_to):
        return k
    if w_from > 1 and w_to > 1:
        k[:, 0, :] *= (w_to - 1) / (w_from - 1)
    if h_from > 1 and h_to > 1:
        k[:, 1, :] *= (h_to - 1) / (h_from - 1)
    return k


def behind_camera_mask(coords, intrinsics, extrinsics):
    """(T, N) bool mask: True where the point is behind the camera.

    ``coords`` is (N, T, 3) in world space, ``extrinsics`` is (T, 4, 4)
    world->camera.
    """
    import torch

    world = coords.permute(1, 0, 2).double()          # (T, N, 3)
    t = world.shape[1]
    rot = extrinsics[:, :3, :3].double()              # (T, 3, 3)
    trans = extrinsics[:, :3, 3].double()             # (T, 3)
    cam = torch.einsum("tij,tnj->tni", rot, world) + trans[:, None, :]
    z = cam[..., 2]
    return (z <= 0).to(torch.bool)


def read_frames_rgb(frames_dir: Path, size_hw: tuple[int, int] | None = None):
    """Read the source frames as a (T, 3, H, W) float tensor in [0, 1].

    The frames are used **as they are** - no resizing - so the overlay lines
    up with the original footage pixel by pixel.
    """
    import numpy as np
    import torch
    from PIL import Image

    paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not paths:
        raise ValueError(f"no frames found in {frames_dir}")

    frames = []
    for path in paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))
        frames.append(arr)

    stack = np.stack(frames)                       # (T, H, W, 3) uint8
    tensor = torch.from_numpy(stack).permute(0, 3, 1, 2).float() / 255.0
    if size_hw is not None and tuple(tensor.shape[-2:]) != tuple(size_hw):
        raise ValueError(
            f"source frames are {tuple(tensor.shape[-2:])} (H, W) but the "
            f"metadata says {tuple(size_hw)}"
        )
    return tensor


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TrackEverything visualizer worker (CPU only, no models)"
    )
    parser.add_argument("--result-dir", required=True,
                        help="directory holding tracks.pt / geometry.pt / metadata.json")
    parser.add_argument("--output-dir", required=True,
                        help="directory that receives the rendered PNG frames")
    parser.add_argument("--frames-dir", default=None,
                        help="optional directory with the source frames")
    parser.add_argument("--width", type=int, default=None,
                        help="render width (defaults to the frames / metadata)")
    parser.add_argument("--height", type=int, default=None,
                        help="render height (defaults to the frames / metadata)")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--trail-length", type=int, default=8)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--dynamic-threshold", type=float, default=0.5)
    parser.add_argument("--background-opacity", type=float, default=0.7)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--linewidth", type=int, default=1)
    args = parser.parse_args(argv)

    _bootstrap_path()

    import numpy as np
    import torch

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = result_dir / "tracks.pt"
    geometry_path = result_dir / "geometry.pt"
    for path in (tracks_path, geometry_path):
        if not path.is_file():
            print(f"ERROR: {path} is missing", file=sys.stderr)
            return 2

    metadata: dict = {}
    metadata_path = result_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    started = time.time()
    report: dict = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "parameters": {
            "max_points": args.max_points,
            "trail_length": args.trail_length,
            "visibility_threshold": args.visibility_threshold,
            "dynamic_threshold": args.dynamic_threshold,
            "background_opacity": args.background_opacity,
            "point_radius": args.point_radius,
            "linewidth": args.linewidth,
            "frames_dir": args.frames_dir,
        },
    }

    try:
        from trackeverything.utils.alltrack3r_utils import CompressedTracks
        from trackeverything.utils.vis_utils import _render_flat_2d_tracks_frames
    except Exception as exc:
        print(f"ERROR: cannot import the upstream renderer: {exc!r}", file=sys.stderr)
        return 3

    try:
        tracks = load_torch_dict(tracks_path)
        geometry = load_torch_dict(geometry_path)

        for key in ("coords", "visibs", "dyn_logits", "src_time", "src_y", "src_x"):
            if key not in tracks:
                raise KeyError(f"tracks.pt has no '{key}'")
        for key in ("intrinsics", "extrinsics"):
            if key not in geometry:
                raise KeyError(f"geometry.pt has no '{key}'")

        coords = tracks["coords"].detach().float().cpu()
        visibs = tracks["visibs"].detach().float().cpu()
        dyn_logits = tracks["dyn_logits"].detach().float().cpu()
        src_time = tracks["src_time"].detach().cpu()
        src_y = tracks["src_y"].detach().cpu()
        src_x = tracks["src_x"].detach().cpu()

        n_tracks, n_frames = int(coords.shape[0]), int(coords.shape[1])

        intrinsics = geometry["intrinsics"].detach().float().cpu()
        extrinsics = geometry["extrinsics"].detach().float().cpu()

        # --- the resolution the cameras refer to --------------------------
        working_hw = None
        out_meta = metadata.get("output", {}) or {}
        working_size = out_meta.get("working_size") or {}
        if working_size.get("width") and working_size.get("height"):
            working_hw = (int(working_size["height"]), int(working_size["width"]))
        elif intrinsics.dim() == 3:
            # fall back to the depth map resolution
            depths = geometry.get("depths")
            if depths is not None:
                working_hw = (int(depths.shape[-2]), int(depths.shape[-1]))
        if working_hw is None:
            raise ValueError(
                "cannot determine the working resolution - metadata.json has no "
                "'output.working_size' and geometry.pt has no depths"
            )

        # --- the frames ----------------------------------------------------
        source_hw = None
        if args.frames_dir:
            frames_dir = Path(args.frames_dir)
            rgb = read_frames_rgb(frames_dir)
            source_hw = (int(rgb.shape[-2]), int(rgb.shape[-1]))
            if rgb.shape[0] != n_frames:
                raise ValueError(
                    f"the source batch has {rgb.shape[0]} frames but the tracks "
                    f"cover {n_frames} frames"
                )
        else:
            rgb = torch.zeros(n_frames, 3, working_hw[0], working_hw[1], dtype=torch.float32)

        # explicit overrides
        if args.width and args.height:
            target_hw = (int(args.height), int(args.width))
        elif source_hw is not None:
            target_hw = source_hw
        else:
            target_hw = working_hw

        if tuple(rgb.shape[-2:]) != tuple(target_hw):
            import cv2

            resized = []
            for frame in rgb:
                frame_np = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                resized.append(
                    cv2.resize(frame_np, (target_hw[1], target_hw[0]),
                               interpolation=cv2.INTER_LINEAR)
                )
            rgb = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).float() / 255.0

        # --- intrinsics for the render resolution --------------------------
        render_intrinsics = scale_intrinsics(intrinsics, working_hw, target_hw)

        # --- mask out points that are behind the camera ---------------------
        behind = behind_camera_mask(coords, render_intrinsics, extrinsics)  # (T, N)
        sentinel_coords = (coords == SENTINEL).any(dim=-1)                  # (N, T)
        sentinel_visibs = visibs <= SENTINEL

        masked_visibs = visibs.clone()
        masked_visibs[behind.permute(1, 0)] = SENTINEL      # (N, T)
        masked_visibs[sentinel_coords] = SENTINEL
        masked_visibs[sentinel_visibs] = SENTINEL

        flat = CompressedTracks(
            coords=coords,
            visibs=masked_visibs,
            dyn_logits=dyn_logits,
            src_time=src_time,
            src_y=src_y,
            src_x=src_x,
        )

        # --- render ----------------------------------------------------------
        frames = _render_flat_2d_tracks_frames(
            rgb,
            render_intrinsics,
            extrinsics,
            flat,
            visib_thresh=float(args.visibility_threshold),
            bkg_opacity=float(args.background_opacity),
            trail_len=int(args.trail_length),
            max_points=int(args.max_points),
            dynamic_thresh=float(args.dynamic_threshold),
            point_radius=int(args.point_radius),
            linewidth=int(args.linewidth),
        )

        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"unexpected renderer output shape {frames.shape}")
        if frames.shape[0] != n_frames:
            raise ValueError(
                f"the renderer produced {frames.shape[0]} frames, expected {n_frames}"
            )

        # --- write the PNG frames -------------------------------------------
        from PIL import Image

        names = []
        for index, frame in enumerate(frames):
            name = f"vis_{index:06d}.png"
            Image.fromarray(frame.astype(np.uint8), mode="RGB").save(
                output_dir / name
            )
            names.append(name)

        report["results"] = {
            "num_frames": int(frames.shape[0]),
            "num_tracks": n_tracks,
            "render_size_hw": [int(target_hw[0]), int(target_hw[1])],
            "working_size_hw": [int(working_hw[0]), int(working_hw[1])],
            "source_size_hw": (
                [int(source_hw[0]), int(source_hw[1])] if source_hw else None
            ),
            "used_source_frames": source_hw is not None,
            "intrinsics_rescaled": tuple(working_hw) != tuple(target_hw),
            "points_behind_camera_masked": int(behind.sum().item()),
            "sentinel_entries_masked": int(sentinel_coords.sum().item()),
            "frame_files": names,
        }
        report["cuda_initialized"] = bool(torch.cuda.is_initialized())
        report["wall_clock_seconds"] = round(time.time() - started, 3)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        (output_dir / "visualization_report.json").write_text(json.dumps(report, indent=2))
        print(
            f"[visualizer] rendered {frames.shape[0]} frames at "
            f"{target_hw[1]}x{target_hw[0]} -> {output_dir}"
        )
        return 0

    except Exception as exc:  # noqa: BLE001 - reported to the node
        import traceback

        traceback.print_exc()
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        try:
            (output_dir / "visualization_report.json").write_text(
                json.dumps(report, indent=2)
            )
        except OSError:
            pass
        print(f"ERROR: visualization failed: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
