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

from .bridge import (
    TrackEverythingError,
    load_result_dir,
    run_selection,
    run_track,
    save_result,
    visualize_result,
)
from .config import Config, ConfigurationError
from .types import TrackData, TrackSelection

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


class TrackEverythingSaveData:
    """Persist a TrackEverything result set.

    ``reference`` keeps the files where they are (nothing is copied) and
    reports the location. ``copy`` duplicates ``tracks.pt``, ``geometry.pt``
    and ``metadata.json`` - byte for byte - into a fresh directory under
    ComfyUI's output directory.

    The node only touches files: it loads no model and never uses CUDA.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "track_data": ("TRACK_DATA",),
                "save_mode": (["reference", "copy"], {"default": "reference"}),
                "filename_prefix": ("STRING", {"default": "trackeverything"}),
            },
            "optional": {
                "selection": ("TRACK_SELECTION",),
            },
        }

    RETURN_TYPES = ("TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "save"
    CATEGORY = "TrackEverything"
    DESCRIPTION = (
        "Save a TrackEverything result. 'reference' keeps the original files, "
        "'copy' writes a byte-identical copy into ComfyUI's output directory. "
        "A selection is copied along with the result."
    )
    OUTPUT_NODE = True

    def save(
        self,
        track_data: TrackData,
        save_mode: str = "reference",
        filename_prefix: str = "trackeverything",
        selection: Optional[TrackSelection] = None,
    ) -> dict:
        if track_data is None:
            raise TrackEverythingError("No track_data received.")

        try:
            config = Config.resolve()
        except ConfigurationError as exc:
            raise TrackEverythingError(str(exc)) from exc

        saved, message = save_result(
            track_data,
            save_mode=save_mode,
            filename_prefix=filename_prefix,
            output_root=config.output_root,
            selection=selection,
        )

        # ComfyUI's standard way of surfacing a text line in the UI.
        return {"ui": {"text": [message]}, "result": (saved,)}


class TrackEverythingSelectObject:
    """Select the 3D tracks of one object using a sequence of masks.

    The masks come from a segmentation system (SAM2/SAM3 or any other) and
    have to cover the same frames as the tracks. The result is a stable list
    of track ids - the original track set is **not** modified and no new
    inference is run.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "track_data": ("TRACK_DATA",),
                "masks": ("MASK",),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0,
                                             "max": 1.0, "step": 0.01}),
                "erode_pixels": ("INT", {"default": 2, "min": 0, "max": 30,
                                         "step": 1}),
                "min_inside_ratio": ("FLOAT", {"default": 0.6, "min": 0.0,
                                                "max": 1.0, "step": 0.01}),
                "min_support_frames": ("INT", {"default": 2, "min": 1,
                                                "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("TRACK_SELECTION", "INT", "STRING")
    RETURN_NAMES = ("selection", "selected_count", "summary")
    FUNCTION = "select"
    CATEGORY = "TrackEverything"
    DESCRIPTION = (
        "Select the tracks of one object from a sequence of segmentation "
        "masks. Returns a stable list of track ids - the source track set "
        "stays untouched."
    )
    OUTPUT_NODE = False

    def select(
        self,
        track_data: TrackData,
        masks: Any,
        mask_threshold: float = 0.5,
        erode_pixels: int = 2,
        min_inside_ratio: float = 0.6,
        min_support_frames: int = 2,
    ) -> tuple:
        if track_data is None:
            raise TrackEverythingError("No track_data received.")

        try:
            config = Config.resolve()
            config.validate()
        except ConfigurationError as exc:
            raise TrackEverythingError(str(exc)) from exc

        selection = run_selection(
            config,
            track_data,
            masks,
            mask_threshold=float(mask_threshold),
            erode_pixels=int(erode_pixels),
            min_inside_ratio=float(min_inside_ratio),
            min_support_frames=int(min_support_frames),
        )

        stats = selection.statistics
        summary = (
            f"selected {selection.count} of {stats.get('num_tracks', '?')} tracks "
            f"over {stats.get('num_frames', '?')} frames "
            f"(median inside ratio {stats.get('median_inside_ratio', 0):.3f})"
        )
        return (selection, int(selection.count), summary)


class TrackEverythingVisualize:
    """Render the 2D track overlay of a TrackEverything result.

    Reads the saved ``tracks.pt`` / ``geometry.pt`` and draws the tracks with
    the original upstream renderer - in a separate, CPU-only worker process.
    No model is loaded and no inference is run.

    Connect the original ``IMAGE`` batch to ``images`` to draw the tracks on
    top of the footage; without it the background is black and the render
    resolution is the one recorded in the metadata.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "track_data": ("TRACK_DATA",),
                "max_points": ("INT", {"default": 5000, "min": 1, "max": 200000,
                                       "step": 100}),
                "trail_length": ("INT", {"default": 8, "min": 1, "max": 512,
                                         "step": 1}),
                "visibility_threshold": ("FLOAT", {"default": 0.5, "min": 0.0,
                                                   "max": 1.0, "step": 0.01}),
                "dynamic_threshold": ("FLOAT", {"default": 0.5, "min": 0.0,
                                                "max": 1.0, "step": 0.01}),
                "background_opacity": ("FLOAT", {"default": 0.7, "min": 0.0,
                                                 "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "selection": ("TRACK_SELECTION",),
            },
        }

    # The socket type must be ComfyUI's standard "IMAGE" type - that is what
    # makes the socket blue and lets it connect to the "images" input of the
    # standard nodes (Create Video, Preview Image, ...). A custom type like
    # "images" would render as a grey, unconnectable socket.
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "visualize"
    CATEGORY = "TrackEverything"
    DESCRIPTION = (
        "Draw the tracked points and their trails with the upstream renderer. "
        "Reads a saved result - no model is loaded and no inference is run. "
        "With a selection only the selected tracks are drawn."
    )
    OUTPUT_NODE = False

    def visualize(
        self,
        track_data: TrackData,
        max_points: int = 5000,
        trail_length: int = 8,
        visibility_threshold: float = 0.5,
        dynamic_threshold: float = 0.5,
        background_opacity: float = 0.7,
        images: Any = None,
        selection: Optional[TrackSelection] = None,
    ) -> tuple:
        if track_data is None:
            raise TrackEverythingError("No track_data received.")

        try:
            config = Config.resolve()
            config.validate()
        except ConfigurationError as exc:
            raise TrackEverythingError(str(exc)) from exc

        (rendered,) = visualize_result(
            config,
            track_data,
            images=images,
            selection=selection,
            max_points=int(max_points),
            trail_length=int(trail_length),
            visibility_threshold=float(visibility_threshold),
            dynamic_threshold=float(dynamic_threshold),
            background_opacity=float(background_opacity),
        )
        return (rendered,)


NODE_CLASS_MAPPINGS = {
    "TrackEverythingTrack": TrackEverythingTrack,
    "TrackEverythingLoadData": TrackEverythingLoadData,
    "TrackEverythingSaveData": TrackEverythingSaveData,
    "TrackEverythingSelectObject": TrackEverythingSelectObject,
    "TrackEverythingVisualize": TrackEverythingVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrackEverythingTrack": "TrackEverything Track",
    "TrackEverythingLoadData": "TrackEverything Load Data",
    "TrackEverythingSaveData": "TrackEverything Save Data",
    "TrackEverythingSelectObject": "TrackEverything Select Object",
    "TrackEverythingVisualize": "TrackEverything Visualize",
}

__all__ = [
    "TrackEverythingTrack",
    "TrackEverythingLoadData",
    "TrackEverythingSaveData",
    "TrackEverythingSelectObject",
    "TrackEverythingVisualize",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
