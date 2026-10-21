"""TrackEverything nodes for ComfyUI.

Install by copying (or symlinking) this directory into
``<ComfyUI>/custom_nodes/``.

The package is import-safe outside ComfyUI: it never imports ``trackeverything``
or ``torch_scatter`` and never touches the GPU. Everything GPU related is
delegated to the isolated worker process.
"""

from __future__ import annotations

from .bridge import TrackEverythingError, load_result_dir, run_track
from .config import Config, ConfigurationError, assert_isolated
from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    TrackEverythingLoadData,
    TrackEverythingTrack,
)
from .types import TrackData

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TrackEverythingLoadData",
    "TrackEverythingTrack",
    "TrackData",
    "TrackEverythingError",
    "ConfigurationError",
    "Config",
    "assert_isolated",
    "load_result_dir",
    "run_track",
]
