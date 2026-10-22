"""The ``TRACK_DATA`` payload exchanged between the TrackEverything nodes.

This is deliberately a *light* structure: it carries file references and
metadata only - never tensors. The heavy numeric payload stays on disk in
``tracks.pt`` / ``geometry.pt`` so it can be shared between processes without
copying GPU memory around.

Both the ``TrackEverything Track`` node and the ``TrackEverything Load Data``
node return exactly this type, so a workflow can either run the model or
re-load a previous run - the downstream side cannot tell the difference.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

#: Bumped whenever the on-disk metadata layout changes.
SCHEMA_VERSION = 1

TRACKS_FILENAME = "tracks.pt"
GEOMETRY_FILENAME = "geometry.pt"
METADATA_FILENAME = "metadata.json"
WORKER_REPORT_FILENAME = "worker_report.json"
WORKER_LOG_FILENAME = "worker.log"
SELECTION_FILENAME = "selection.json"


@dataclass(frozen=True)
class TrackData:
    """Reference to one TrackEverything result set.

    Attributes:
        result_dir: Directory holding the result files.
        tracks_path: ``tracks.pt`` - the scene tokens.
        geometry_path: ``geometry.pt`` - depths, intrinsics, extrinsics.
        metadata_path: ``metadata.json`` - run metadata.
        metadata: The parsed metadata (same content as ``metadata.json``).
    """

    result_dir: str
    tracks_path: str
    geometry_path: str
    metadata_path: str
    metadata: dict = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    @property
    def num_frames(self) -> Optional[int]:
        return self.metadata.get("output", {}).get("num_frames")

    @property
    def num_tracks(self) -> Optional[int]:
        return self.metadata.get("output", {}).get("num_tracks")

    @property
    def original_size(self) -> Optional[dict]:
        return self.metadata.get("input", {}).get("original_size")

    @property
    def working_size(self) -> Optional[dict]:
        return self.metadata.get("output", {}).get("working_size")

    @property
    def coordinate_transform(self) -> dict:
        return self.metadata.get("coordinate_transform", {})

    @property
    def worker_log_path(self) -> Optional[str]:
        log = self.metadata.get("files", {}).get("worker_log")
        if not log:
            return None
        path = Path(self.result_dir) / log
        return str(path) if path.exists() else None

    def exists(self) -> bool:
        return (
            os.path.isfile(self.tracks_path)
            and os.path.isfile(self.geometry_path)
            and os.path.isfile(self.metadata_path)
        )

    def load_metadata(self) -> dict:
        """Re-read ``metadata.json`` from disk."""
        with open(self.metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- mapping-like access (handy in workflows / custom nodes) --------
    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def keys(self) -> Any:
        return self.metadata.keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self.metadata)

    def __contains__(self, key: object) -> bool:
        return key in self.metadata

    def __repr__(self) -> str:
        return (
            f"TrackData(result_dir={self.result_dir!r}, "
            f"num_frames={self.num_frames}, num_tracks={self.num_tracks})"
        )


@dataclass(frozen=True)
class TrackSelection:
    """A selection of stable track IDs that belong to one object.

    The selection is a **list of identifiers**, not a new inference: the
    original track set stays untouched and is always referenced by its
    fingerprint, so a selection cannot silently be applied to a different
    result.

    Attributes:
        selection_dir: Directory holding ``selection.json``.
        selection_path: The ``selection.json`` file.
        source_result_dir: The result directory the selection was made from.
        source_tracks_path: The ``tracks.pt`` of that result.
        fingerprint: The fingerprint of the source track set.
        token_ids: The selected token IDs (stable across the sequence).
        count: How many tokens are selected.
        parameters: The classification parameters that were used.
        metadata: The parsed ``selection.json``.
    """

    selection_dir: str
    selection_path: str
    source_result_dir: str
    source_tracks_path: str
    fingerprint: str
    token_ids: tuple[int, ...] = ()
    count: int = 0
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    @property
    def num_frames(self) -> Optional[int]:
        return self.metadata.get("source", {}).get("num_frames")

    @property
    def num_tracks(self) -> Optional[int]:
        return self.metadata.get("source", {}).get("num_tracks")

    @property
    def statistics(self) -> dict:
        return self.metadata.get("statistics", {})

    def load_metadata(self) -> dict:
        """Re-read ``selection.json`` from disk."""
        with open(self.selection_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- mapping-like access --------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def keys(self) -> Any:
        return self.metadata.keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self.metadata)

    def __contains__(self, key: object) -> bool:
        return key in self.metadata

    def __repr__(self) -> str:
        return (
            f"TrackSelection(selection_dir={self.selection_dir!r}, "
            f"count={self.count}, fingerprint={self.fingerprint[:19]!r}...)"
        )
