"""TrackEverything worker package.

The worker runs in the **isolated** environment (its own Python 3.10 with
torch 2.8 / CUDA 12.9). It is launched as a separate process by the ComfyUI
nodes and exchanges all data through files.

This package is intentionally *not* imported by the ComfyUI process.
"""

from __future__ import annotations

__all__ = ["runner"]
