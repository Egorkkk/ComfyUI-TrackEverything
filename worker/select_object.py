#!/usr/bin/env python
"""TrackEverything selection worker - pick the tracks of one object.

Reads a saved TrackEverything result (``tracks.pt``) and a sequence of
segmentation masks, and decides which of the stable track IDs belong to the
object.

The masks are the only per-frame input. The tracker itself already merged
physically equal points into stable tokens, and ``cell_to_token`` maps every
image cell of every frame to its token - that is the correspondence used here.

Algorithm
---------
For every frame the mask is mapped onto the tracker's *feature grid* (the
resolution the tracks live on) and compared with ``cell_to_token[t]``:

1. the mask is thresholded and optionally eroded (in the mask's own
   resolution) to drop the border of the object,
2. the mask is brought to the tracker's resolution with the transforms from
   ``metadata.json`` (original -> working -> feature grid) - the resolutions
   are **not** assumed to be equal,
3. every cell of the feature grid is inside the mask when the fraction of
   its pixels that are masked is at least half.

The evidence is then accumulated **per frame**, not per cell:

* a token contributes at most one vote per frame - a token with many cells
  in one frame cannot outvote a token with a single cell in many frames,
* a frame only counts if the token exists in it (``src_time``) and is
  actually observed there (the visibility logit, and the ``-1000`` sentinel
  for "never written"),
* the temporal score of a token is the mean of its per-frame coverage
  (the fraction of the token's cells that are inside the mask) over the
  frames it was observed in.

A token is selected when its score reaches ``min_inside_ratio`` and it was
supported by at least ``min_support_frames`` frames. A single frame without
a point inside the mask therefore lowers the score but does not remove the
track, and a track that appears later is judged on its own frames.

The worker never loads a model and never touches the GPU.
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

if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from console import Console, install_upstream_logging  # noqa: E402
from trackdata import fingerprint, load_torch_dict, tracks_metadata  # noqa: E402

#: A feature cell counts as inside the mask when at least this fraction of
#: its pixels is masked.
CELL_COVERAGE_THRESHOLD = 0.5

#: A visibility logit above this probability means "observed".
VISIBILITY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# mask handling
# ---------------------------------------------------------------------------
def read_mask_frames(masks_dir: Path) -> list:
    """Read the binary mask frames (PNG, 0/255) in file-name order."""
    import numpy as np
    from PIL import Image

    paths = sorted(
        p for p in masks_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise ValueError(f"no mask frames found in {masks_dir}")
    frames = []
    for path in paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"))
        frames.append(arr > 127)
    return frames


def erode_mask(mask, pixels: int):
    """Erode a binary mask by ``pixels`` (in the mask's own resolution)."""
    if pixels <= 0:
        return mask
    import cv2
    import numpy as np

    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def mask_to_grid(mask, working_hw, grid_hw, patch: int, original_hw=None,
                 erode_pixels: int = 0):
    """Map one mask onto the tracker's feature grid.

    The chain is the one the upstream uses for the video:
    ``original -> working -> feature grid``. The mask covers the same field
    of view as the original frames, so it is first brought to the original
    resolution (if it is known and differs), then to the working resolution
    and finally reduced to the feature grid.

    ``erode_pixels`` is measured in the pixels of the **original** frames and
    is applied at the working resolution - that is the scale the tracks live
    on, and the only one where an erosion has a visible effect on a cell.
    """
    import cv2
    import numpy as np

    work_h, work_w = working_hw
    grid_h, grid_w = grid_hw

    if original_hw is not None:
        orig_h, orig_w = original_hw
        if (mask.shape[0], mask.shape[1]) != (orig_h, orig_w):
            mask = cv2.resize(
                mask.astype(np.uint8), (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    if (mask.shape[0], mask.shape[1]) != (work_h, work_w):
        mask = cv2.resize(
            mask.astype(np.uint8), (work_w, work_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # erode at the working resolution
    if erode_pixels > 0:
        if original_hw is not None and original_hw[0] > 0:
            radius = max(1, int(round(erode_pixels * work_h / original_hw[0])))
        else:
            radius = max(1, int(erode_pixels))
        mask = erode_mask(mask, radius)

    # reduce the working resolution to the feature grid: a cell is inside
    # when at least half of its pixels are masked
    occupancy = mask.reshape(grid_h, patch, grid_w, patch).mean(axis=(1, 3))
    return occupancy >= CELL_COVERAGE_THRESHOLD


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def select_tokens(tracks: dict, mask_frames, working_hw, grid_hw, patch: int,
                  original_hw=None, min_inside_ratio: float = 0.6,
                  min_support_frames: int = 2) -> dict:
    """Compute the per-token evidence and the selected ids."""
    import numpy as np
    import torch

    coords = tracks["coords"]
    num_tracks, num_frames = int(coords.shape[0]), int(coords.shape[1])
    cell_to_token = tracks["cell_to_token"]

    # the masks have to be torch bool tensors: mixing a numpy bool array into
    # a torch index would take the deprecated uint8-mask path and pick the
    # wrong cells
    mask_flat = [
        torch.as_tensor(mask.reshape(-1), dtype=torch.bool)
        for mask in mask_frames
    ]

    visibs = tracks["visibs"].detach().float()
    src_time = tracks["src_time"].detach().to(torch.int64)

    # a token is "observed" in a frame when the visibility logit says so;
    # the -1000 sentinel ("never written") gives a probability of 0
    observed = torch.sigmoid(visibs) > VISIBILITY_THRESHOLD

    score_sum = torch.zeros(num_tracks, dtype=torch.float64)
    frames_seen = torch.zeros(num_tracks, dtype=torch.float64)
    positive_frames = torch.zeros(num_tracks, dtype=torch.float64)
    inside_cells_total = torch.zeros(num_tracks, dtype=torch.float64)
    cells_total = torch.zeros(num_tracks, dtype=torch.float64)

    per_frame = []
    for t in range(num_frames):
        ids = cell_to_token[t].detach().to(torch.int64).reshape(-1)
        valid = ids >= 0
        if not bool(valid.any()):
            per_frame.append({"frame": t, "tokens": 0})
            continue

        inside = mask_flat[t]
        # only the cells that carry a valid token id take part
        use = valid & inside

        counts = torch.bincount(ids[valid], minlength=num_tracks + 1)[:num_tracks]
        inside_counts = torch.bincount(ids[use], minlength=num_tracks + 1)[:num_tracks]

        # a token only exists from its birth frame on
        alive = src_time <= t

        # ... and it must be observed in that frame
        seen = observed[:, t] & alive

        total = counts.to(torch.float64)
        inside_f = inside_counts.to(torch.float64)
        coverage = torch.where(total > 0, inside_f / total.clamp(min=1.0),
                               torch.zeros_like(total))

        # per frame: at most one vote, and only for observed tokens
        contributes = seen & (total > 0)
        score_sum += torch.where(contributes, coverage,
                                 torch.zeros_like(coverage))
        frames_seen += contributes.to(torch.float64)
        positive_frames += (contributes & (inside_f > 0)).to(torch.float64)
        inside_cells_total += inside_f
        cells_total += total

        per_frame.append({
            "frame": t,
            "tokens": int((counts > 0).sum()),
            "tokens_inside": int(((inside_counts > 0) & seen).sum()),
        })

    ratio = torch.where(frames_seen > 0, score_sum / frames_seen.clamp(min=1.0),
                        torch.zeros_like(score_sum))

    selected_mask = (
        (ratio >= min_inside_ratio)
        & (positive_frames >= min_support_frames)
        & (frames_seen > 0)
    )
    selected = torch.nonzero(selected_mask, as_tuple=False).reshape(-1).tolist()

    stats = {
        "num_tracks": num_tracks,
        "num_frames": num_frames,
        "num_selected": len(selected),
        "selected_fraction": (len(selected) / num_tracks) if num_tracks else 0.0,
        "median_inside_ratio": float(ratio.median()) if num_tracks else 0.0,
        "mean_inside_ratio": float(ratio.mean()) if num_tracks else 0.0,
        "num_tracks_with_support": int((positive_frames >= min_support_frames).sum()),
    }

    return {
        "selected_token_ids": [int(i) for i in selected],
        "statistics": stats,
        "per_frame": per_frame,
        "_ratio": ratio.tolist(),
        "_positive_frames": positive_frames.tolist(),
        "_frames_seen": frames_seen.tolist(),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TrackEverything object selection worker (CPU only)"
    )
    parser.add_argument("--result-dir", required=True,
                        help="directory with tracks.pt / geometry.pt / metadata.json")
    parser.add_argument("--masks-dir", required=True,
                        help="directory with the binary mask frames (PNG)")
    parser.add_argument("--output-dir", required=True,
                        help="directory that receives selection.json")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--erode-pixels", type=int, default=2)
    parser.add_argument("--min-inside-ratio", type=float, default=0.6)
    parser.add_argument("--min-support-frames", type=int, default=2)
    args = parser.parse_args(argv)

    console = Console()
    install_upstream_logging(console)

    started = time.time()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = result_dir / "tracks.pt"
    if not tracks_path.is_file():
        print(f"ERROR: {tracks_path} is missing", file=sys.stderr)
        return 2

    console.banner(
        "TrackEverything - object selection",
        {
            "result": str(result_dir),
            "masks": args.masks_dir,
            "mask threshold": args.mask_threshold,
            "erode pixels": args.erode_pixels,
            "min inside ratio": args.min_inside_ratio,
            "min support frames": args.min_support_frames,
            "output": str(output_dir),
        },
    )

    try:
        # -- stage 1: load ------------------------------------------------
        console.stage(1, 3, "loading the tracks and the masks")
        tracks = load_torch_dict(tracks_path)
        for key in ("coords", "visibs", "dyn_logits", "src_time", "src_y",
                    "src_x", "cell_to_token"):
            if key not in tracks:
                raise KeyError(f"tracks.pt has no '{key}'")

        metadata: dict = {}
        metadata_path = result_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}

        info = tracks_metadata(tracks)
        num_frames = info["num_frames"]
        grid_h, grid_w = info["feature_grid_hw"]

        out_meta = metadata.get("output", {}) or {}
        working_size = out_meta.get("working_size") or {}
        if not (working_size.get("width") and working_size.get("height")):
            raise ValueError(
                "cannot determine the working resolution - metadata.json has no "
                "'output.working_size'"
            )
        working_hw = (int(working_size["height"]), int(working_size["width"]))

        in_meta = metadata.get("input", {}) or {}
        original_size = in_meta.get("original_size") or {}
        original_hw = None
        if original_size.get("width") and original_size.get("height"):
            original_hw = (int(original_size["height"]), int(original_size["width"]))

        patch = int(out_meta.get("patch_size") or 4)

        mask_frames = read_mask_frames(Path(args.masks_dir))
        console.stage_done(
            time.time() - started,
            f"{info['num_tracks']} tracks over {num_frames} frames, "
            f"{len(mask_frames)} masks, grid {grid_w}x{grid_h}",
        )

        # -- stage 2: the frame count has to match ------------------------
        console.stage(2, 3, "checking the frame count")
        if len(mask_frames) != num_frames:
            raise ValueError(
                f"the mask sequence has {len(mask_frames)} frames but the tracks "
                f"cover {num_frames} frames - they must match exactly"
            )
        mask_shape = mask_frames[0].shape
        for index, frame in enumerate(mask_frames):
            if frame.shape != mask_shape:
                raise ValueError(
                    f"mask frame {index} has the size {frame.shape}, expected "
                    f"{mask_shape} - all mask frames must share one resolution"
                )
        console.stage_done(
            time.time() - started,
            f"{num_frames} masks at {mask_shape[1]}x{mask_shape[0]}",
        )

        # -- stage 3: the selection --------------------------------------
        console.stage(3, 3, "matching the masks against the tracks")
        t0 = time.time()
        grid_masks = [
            mask_to_grid(frame, working_hw, (grid_h, grid_w), patch, original_hw,
                         erode_pixels=int(args.erode_pixels))
            for frame in mask_frames
        ]
        console.detail(
            f"eroded by {args.erode_pixels} px and mapped onto the "
            f"{grid_w}x{grid_h} feature grid"
        )

        result = select_tokens(
            tracks,
            grid_masks,
            working_hw,
            (grid_h, grid_w),
            patch,
            original_hw=original_hw,
            min_inside_ratio=float(args.min_inside_ratio),
            min_support_frames=int(args.min_support_frames),
        )
        selection_seconds = time.time() - t0

        selected = result["selected_token_ids"]
        stats = result["statistics"]
        console.stage_done(
            selection_seconds,
            f"{stats['num_selected']} of {stats['num_tracks']} tracks selected "
            f"({stats['selected_fraction'] * 100:.1f} %)",
        )

        # -- write the selection -----------------------------------------
        source = {
            "result_dir": str(result_dir.resolve()),
            "tracks_path": str(tracks_path.resolve()),
            "fingerprint": fingerprint(tracks),
            **info,
            "working_size": {"width": working_hw[1], "height": working_hw[0]},
            "original_size": (
                {"width": original_hw[1], "height": original_hw[0]}
                if original_hw else None
            ),
        }
        selection = {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "producer": "comfy_nodes.TrackEverythingSelectObject",
            "source": source,
            "parameters": {
                "mask_threshold": float(args.mask_threshold),
                "erode_pixels": int(args.erode_pixels),
                "min_inside_ratio": float(args.min_inside_ratio),
                "min_support_frames": int(args.min_support_frames),
                "cell_coverage_threshold": CELL_COVERAGE_THRESHOLD,
                "visibility_threshold": VISIBILITY_THRESHOLD,
                "mask_size_hw": [int(mask_shape[1]), int(mask_shape[0])],
            },
            "selected_token_ids": selected,
            "num_selected": len(selected),
            "statistics": stats,
        }
        (output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )

        console.summary(
            {
                "selected": len(selected),
                "of tracks": stats["num_tracks"],
                "median ratio": f"{stats['median_inside_ratio']:.3f}",
                "total": f"{time.time() - started:.2f} s",
                "cuda": "not used (CPU only)",
            }
        )
        print(f"[select] {len(selected)} tracks selected -> {output_dir}")
        return 0

    except Exception as exc:  # noqa: BLE001 - reported to the node
        import traceback

        traceback.print_exc()
        print(f"ERROR: the selection failed: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
