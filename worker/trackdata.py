"""Shared helpers for the workers that read a TrackEverything result.

* loading the tensor dictionaries,
* a **fingerprint** of a track set, so that a selection can be checked
  against the result it was made for.

The fingerprint must be identical wherever it is computed - the selection
worker computes it for the selection, the visualizer worker recomputes it for
the tracks it is about to render.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

FINGERPRINT_VERSION = "trackeverything-fingerprint-v1"


def load_torch_dict(path: Path) -> dict:
    """Load a locally produced ``.pt`` file on the CPU."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def fingerprint(tracks: dict) -> str:
    """A cheap but discriminating fingerprint of a track set.

    It combines the shape, the per-track birth information and a sample of
    the coordinates. Two different inference runs of the same clip produce
    different fingerprints, so a selection cannot be applied to the wrong
    result by accident.
    """
    import torch

    coords = tracks["coords"]
    h = hashlib.sha256()
    h.update(FINGERPRINT_VERSION.encode())

    h.update(str(tuple(coords.shape)).encode())
    for key in ("src_time", "src_x", "src_y"):
        value = tracks.get(key)
        if value is not None:
            h.update(str(int(value.to(torch.int64).sum())).encode())

    # a sample of the coordinates - enough to separate two different runs
    flat = coords.detach().float().reshape(coords.shape[0], -1)
    step = max(1, flat.shape[0] // 512)
    h.update(flat[::step].sum().item().hex().encode())

    cell_to_token = tracks.get("cell_to_token")
    if cell_to_token is not None:
        h.update(str(tuple(cell_to_token.shape)).encode())
        step = max(1, cell_to_token.shape[0] // 32)
        h.update(str(int(cell_to_token[::step].sum())).encode())

    return "sha256:" + h.hexdigest()


def tracks_metadata(tracks: dict) -> dict:
    """A few descriptive numbers about a track set."""
    coords = tracks.get("coords")
    cell_to_token = tracks.get("cell_to_token")
    info: dict[str, Any] = {}
    if coords is not None:
        info["num_tracks"] = int(coords.shape[0])
        info["num_frames"] = int(coords.shape[1])
    if cell_to_token is not None:
        info["feature_grid_hw"] = [int(cell_to_token.shape[1]),
                                   int(cell_to_token.shape[2])]
    return info
