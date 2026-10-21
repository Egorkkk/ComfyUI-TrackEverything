"""ComfyUI node definitions for TrackEverything.

Two nodes only (ТЗ №2):

``TrackEverything Track``
    Runs the upstream model on an ``IMAGE`` batch and returns a
    ``TRACK_DATA`` reference.

``TrackEverything Load Data``
    Returns the same ``TRACK_DATA`` reference for an existing result
    directory - no model, no GPU.

Both nodes return the identical :class:`~comfy_nodes.types.TrackData` type,
so a workflow can be rewired freely between them.

The ComfyUI process never imports ``trackeverything`` / ``torch_scatter``;
all GPU work happens in the isolated worker process.
"""

from __future__ import annotations

from typing import Any, Optional

from .bridge import TrackEverythingError, load_result_dir, run_track
from .config import Config, ConfigurationError

#: ComfyUI widgets for a boolean defaulting to True.
BOOLEAN_TRUE = ("BOOLEAN", {"default": True})


class TrackEverythingTrack:
    """Track every point of an image sequence with TrackEverything.

    The frames are written to a temporary folder, the upstream model runs in
    an isolated worker process, and the results are stored in ComfyUI's output
    directory. The node returns a ``TRACK_DATA`` handle - the tensors stay on
    disk.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": ("IMAGE",),
                "num_iters": ("INT", {"default": 3, "min": 1, "max": 16,
                                      "step": 1}),
                "voxel_size": ("FLOAT", {"default": 0.02, "min": 0.001,
                                         "max": 1.0, "step": 0.001}),
                "mem_efficient": BOOLEAN_TRUE,
            },
        }

    RETURN_TYPES = ("TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "track"
    CATEGORY = "TrackEverything"
    DESCRIPTION = (
        "Long-horizon dense 3D point tracking with the upstream TrackEverything "
        "model. Runs in an isolated worker process (own CUDA context) and stores "
        "tracks.pt / geometry.pt / metadata.json in the output directory."
    )
    OUTPUT_NODE = False

    def track(
        self,
        images: Any,
        num_iters: int = 3,
        voxel_size: float = 0.02,
        mem_efficient: bool = True,
    ) -> tuple:
        try:
            config = Config.resolve()
            config.validate()
        except ConfigurationError as exc:
            raise TrackEverythingError(str(exc)) from exc

        return (
            run_track(
                config,
                images,
                num_iters=int(num_iters),
                voxel_size=float(voxel_size),
                mem_efficient=bool(mem_efficient),
            ),
        )


class TrackEverythingLoadData:
    """Load a previously computed TrackEverything result.

    Takes the path of a result directory (the one containing ``tracks.pt``)
    and returns the same ``TRACK_DATA`` handle the Track node produces.
    Does not load the model and does not use the GPU.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "result_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "load"
    CATEGORY = "TrackEverything"
    DESCRIPTION = (
        "Reference an existing TrackEverything result directory. "
        "Returns the same TRACK_DATA as the Track node without running the model."
    )
    OUTPUT_NODE = False

    def load(self, result_dir: str = "") -> tuple:
        if not result_dir or not str(result_dir).strip():
            raise TrackEverythingError(
                "No result directory given. Point the node at the folder that "
                "contains tracks.pt (for example <ComfyUI output>/trackeverything_*)."
            )
        return (load_result_dir(result_dir),)


NODE_CLASS_MAPPINGS = {
    "TrackEverythingTrack": TrackEverythingTrack,
    "TrackEverythingLoadData": TrackEverythingLoadData,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrackEverythingTrack": "TrackEverything Track",
    "TrackEverythingLoadData": "TrackEverything Load Data",
}

__all__ = [
    "TrackEverythingTrack",
    "TrackEverythingLoadData",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
