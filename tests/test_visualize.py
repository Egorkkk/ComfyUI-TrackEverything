#!/usr/bin/env python
"""Unit tests for the ``TrackEverything Visualize`` node and its worker.

The geometry tests build a synthetic result set with *known* 3D points, a
known camera matrix and known extrinsics, and check where the points land.
That pins the projection down exactly::

    X_cam = R @ X_world + t
    p_h   = K @ X_cam
    x     = p_h[0] / p_h[2]
    y     = p_h[1] / p_h[2]

The renderer itself is the **original upstream** one; only the adapter from
``SceneTokens`` to ``CompressedTracks`` is ours.

Run with:
    pixi run python tests/test_visualize.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from comfy_nodes import bridge, nodes  # noqa: E402
from comfy_nodes.config import Config  # noqa: E402
from comfy_nodes.types import TrackData  # noqa: E402

# --- the synthetic scene ---------------------------------------------------
W, H = 100, 80                 # render resolution (width, height)
FX, FY, CX, CY = 100.0, 100.0, 50.0, 40.0
NUM_FRAMES = 4

#: World positions chosen so that the expected pixel is known and **distinct**:
#:   x = fx * X / Z + cx
#: (the behind-camera point would land on a perfectly ordinary pixel if the
#:  behind-camera mask were missing - that is exactly what the mask is for)
EXPECTED_PIXELS = {
    "visible": (50, 40),        # X = 0.00, Z = 2 -> 50
    "invisible": (75, 40),      # X = 0.50, Z = 2 -> 75
    "behind_camera": (15, 40),  # X = 0.70, Z = -2 -> 15
    "sentinel": (25, 40),       # X = -0.50, Z = 2 -> 25
    "late_birth": (65, 40),     # X = 0.30, Z = 2 -> 65
}


def build_result(directory: Path) -> Path:
    """Write a result set with one track per scenario.

    The ``visible`` track moves a little so that the trail rendering is
    exercised; its position in frame 0 is the reference one.
    """
    import torch

    directory.mkdir(parents=True, exist_ok=True)

    def track(x, z, src_time=0, visible=True, sentinel=False, velocity=0.0,
              dynamic=False):
        # (1, NUM_FRAMES, 3) - one track, one position per frame
        xs = [x + velocity * t for t in range(NUM_FRAMES)]
        coords = torch.tensor([[[xv, 0.0, z] for xv in xs]], dtype=torch.float32)
        if sentinel:
            coords = torch.full_like(coords, -1000.0)
        visibs = torch.full((1, NUM_FRAMES), 10.0 if visible else -10.0)
        # the upstream renderer draws trails for *dynamic* points only, and
        # static points as 1px dots while dynamic ones become filled circles
        dyn = torch.full((1, NUM_FRAMES), 5.0 if dynamic else 0.0)
        return coords, visibs, dyn, src_time

    specs = [
        track(0.0, 2.0, src_time=0, visible=True, velocity=0.05, dynamic=True),
        track(0.5, 2.0, src_time=0, visible=False),
        track(0.7, -2.0, src_time=0, visible=True),
        track(-0.5, 2.0, src_time=0, visible=True, sentinel=True),
        track(0.3, 2.0, src_time=2, visible=True),
    ]

    tracks = {
        "coords": torch.cat([s[0] for s in specs], dim=0),
        "visibs": torch.cat([s[1] for s in specs], dim=0),
        "dyn_logits": torch.cat([s[2] for s in specs], dim=0),
        "src_time": torch.tensor([s[3] for s in specs], dtype=torch.int64),
        "src_y": torch.zeros(len(specs), dtype=torch.int64),
        "src_x": torch.zeros(len(specs), dtype=torch.int64),
        "cell_to_token": torch.zeros(NUM_FRAMES, H // 4, W // 4, dtype=torch.int64),
    }
    torch.save(tracks, directory / "tracks.pt")

    intrinsics = torch.tensor(
        [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    extrinsics = torch.eye(4, dtype=torch.float32)
    geometry = {
        "norm_factor": 1.0,
        "depths": torch.ones(NUM_FRAMES, H, W, dtype=torch.float32),
        "intrinsics": intrinsics.expand(NUM_FRAMES, 3, 3).contiguous(),
        "extrinsics": extrinsics.expand(NUM_FRAMES, 4, 4).contiguous(),
    }
    torch.save(geometry, directory / "geometry.pt")

    metadata = {
        "schema_version": 1,
        "input": {"num_frames": NUM_FRAMES,
                  "original_size": {"width": W, "height": H}},
        "output": {
            "num_frames": NUM_FRAMES,
            "num_tracks": len(specs),
            "working_size": {"width": W, "height": H},
            "feature_grid": {"height": H // 4, "width": W // 4},
        },
        "files": {"tracks": "tracks.pt", "geometry": "geometry.pt",
                  "metadata": "metadata.json"},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return directory


def render(result_dir: Path, out_dir: Path, **kwargs) -> np.ndarray:
    """Run the visualizer worker in-process and return the rendered frames."""
    sys.path.insert(0, str(PROJECT_ROOT / "worker"))
    import visualize as viz  # noqa: PLC0415

    argv = [
        "--result-dir", str(result_dir),
        "--output-dir", str(out_dir),
        "--max-points", str(kwargs.get("max_points", 5000)),
        "--trail-length", str(kwargs.get("trail_length", 8)),
        "--visibility-threshold", str(kwargs.get("visibility_threshold", 0.5)),
        "--dynamic-threshold", str(kwargs.get("dynamic_threshold", 0.5)),
        "--background-opacity", str(kwargs.get("background_opacity", 0.7)),
    ]
    if kwargs.get("frames_dir"):
        argv += ["--frames-dir", str(kwargs["frames_dir"])]
    code = viz.main(argv)
    assert code == 0, f"the visualizer failed with exit code {code}"

    from PIL import Image

    paths = sorted(out_dir.glob("vis_*.png"))
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def is_lit(frame: np.ndarray, x: int, y: int) -> bool:
    """True when the pixel carries a drawn mark (the background is black)."""
    return bool(frame[y, x].max() > 0)


class VisualizeCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="te_vis_test_")
        self.tmp = Path(self._tmp)
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self.result = build_result(self.tmp / "result")

    def _track_data(self) -> TrackData:
        metadata = json.loads((self.result / "metadata.json").read_text())
        return bridge.build_track_data(self.result, metadata)


class TestRegistration(VisualizeCase):
    def test_node_is_registered(self):
        self.assertIn("TrackEverythingVisualize", nodes.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            nodes.NODE_DISPLAY_NAME_MAPPINGS["TrackEverythingVisualize"],
            "TrackEverything Visualize",
        )

    def test_input_interface(self):
        spec = nodes.TrackEverythingVisualize.INPUT_TYPES()
        required = spec["required"]
        optional = spec["optional"]
        self.assertIn("track_data", required)
        self.assertIn("images", optional)
        self.assertEqual(required["max_points"][1]["default"], 5000)
        self.assertEqual(required["trail_length"][1]["default"], 8)
        self.assertEqual(required["visibility_threshold"][1]["default"], 0.5)
        self.assertEqual(required["dynamic_threshold"][1]["default"], 0.5)
        self.assertEqual(required["background_opacity"][1]["default"], 0.7)

    def test_returned_value_follows_the_image_type_contract(self):
        """The value must satisfy ComfyUI's IMAGE type contract."""
        import torch

        config = Config.resolve()
        config.worker_python = Path(sys.executable)
        (rendered,) = bridge.visualize_result(config, self._track_data())

        self.assertIsInstance(rendered, torch.Tensor)
        self.assertEqual(rendered.dtype, torch.float32)
        self.assertEqual(rendered.ndim, 4)          # (B, H, W, C)
        self.assertEqual(rendered.shape[-1], 3)     # RGB
        self.assertGreaterEqual(float(rendered.min()), 0.0)
        self.assertLessEqual(float(rendered.max()), 1.0)

    def test_output_socket_uses_the_standard_comfyui_image_type(self):
        """The socket type must be ComfyUI's standard "IMAGE" type.

        A custom type name (e.g. "images") makes the socket grey and
        unconnectable: it cannot be linked to the "images" input of the
        standard nodes (Create Video, Preview Image, ...), because ComfyUI
        only links sockets whose type names match exactly.
        """
        self.assertEqual(
            nodes.TrackEverythingVisualize.RETURN_TYPES,
            ("IMAGE",),
            "the output socket must use the standard ComfyUI type 'IMAGE'",
        )
        # the *display* name stays lowercase
        self.assertEqual(nodes.TrackEverythingVisualize.RETURN_NAMES, ("images",))

    def test_output_type_matches_a_standard_image_input(self):
        """The output type must match the input type of the standard nodes."""
        standard_image_inputs = ["IMAGE"]
        for out_type in nodes.TrackEverythingVisualize.RETURN_TYPES:
            self.assertIn(
                out_type, standard_image_inputs,
                "the output type must be connectable to a standard IMAGE input",
            )

    def test_input_socket_uses_the_standard_image_type(self):
        optional = nodes.TrackEverythingVisualize.INPUT_TYPES()["optional"]
        self.assertEqual(optional["images"][0], "IMAGE")


class TestGeometryHelpers(unittest.TestCase):
    def test_scale_intrinsics_is_the_inverse_of_the_upstream_rule(self):
        sys.path.insert(0, str(PROJECT_ROOT / "worker"))
        import torch
        import visualize as viz

        k = torch.tensor([[[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]])
        # upstream: working K = original K scaled to the model resolution
        working = viz.scale_intrinsics(k, (H, W), (384, 512))
        # our inverse brings it back exactly
        back = viz.scale_intrinsics(working, (384, 512), (H, W))
        self.assertTrue(torch.allclose(back, k, atol=1e-5))

    def test_behind_camera_mask(self):
        sys.path.insert(0, str(PROJECT_ROOT / "worker"))
        import torch
        import visualize as viz

        coords = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]])  # (1, 2, 3)
        extrinsics = torch.eye(4).unsqueeze(0)
        intrinsics = torch.eye(3).unsqueeze(0)
        mask = viz.behind_camera_mask(coords, intrinsics, extrinsics)  # (T=2, N=1)
        self.assertEqual(tuple(mask.shape), (2, 1))
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[1, 0]))


class TestRendering(VisualizeCase):
    def test_black_background_render(self):
        frames = render(self.result, self.tmp / "out_black")
        self.assertEqual(frames.shape, (NUM_FRAMES, H, W, 3))
        self.assertEqual(frames.dtype, np.uint8)

    def test_known_point_lands_on_the_expected_pixel(self):
        frames = render(self.result, self.tmp / "out_known")
        x, y = EXPECTED_PIXELS["visible"]
        self.assertTrue(is_lit(frames[0], x, y),
                        f"the visible point is missing at ({x}, {y})")

    def test_invisible_point_is_not_drawn(self):
        frames = render(self.result, self.tmp / "out_invis")
        x, y = EXPECTED_PIXELS["invisible"]
        self.assertFalse(is_lit(frames[0], x, y),
                         f"an invisible point was drawn at ({x}, {y})")

    def test_point_behind_the_camera_is_not_drawn(self):
        # without the behind-camera mask this point would land on exactly the
        # same pixel as the visible one - so the mask is what makes it vanish
        frames = render(self.result, self.tmp / "out_behind")
        x, y = EXPECTED_PIXELS["behind_camera"]
        self.assertFalse(is_lit(frames[0], x, y),
                         f"a point behind the camera was drawn at ({x}, {y})")

    def test_sentinel_entry_is_not_drawn(self):
        frames = render(self.result, self.tmp / "out_sentinel")
        x, y = EXPECTED_PIXELS["sentinel"]
        self.assertFalse(is_lit(frames[0], x, y),
                         f"a sentinel entry was drawn at ({x}, {y})")

    def test_src_time_controls_the_first_appearance(self):
        frames = render(self.result, self.tmp / "out_srctime")
        x, y = EXPECTED_PIXELS["late_birth"]
        # born at frame 2 -> invisible in frames 0 and 1
        self.assertFalse(is_lit(frames[0], x, y))
        self.assertFalse(is_lit(frames[1], x, y))
        # ... and visible from frame 2 on
        self.assertTrue(is_lit(frames[2], x, y))
        self.assertTrue(is_lit(frames[3], x, y))

    def test_frame_count_and_resolution(self):
        frames = render(self.result, self.tmp / "out_count")
        self.assertEqual(len(frames), NUM_FRAMES)
        self.assertEqual(frames.shape[1:], (H, W, 3))

    def test_render_over_source_frames(self):
        from PIL import Image

        frames_dir = self.tmp / "frames"
        frames_dir.mkdir()
        for index in range(NUM_FRAMES):
            arr = np.full((H, W, 3), 40, dtype=np.uint8)
            Image.fromarray(arr).save(frames_dir / f"frame_{index:06d}.png")

        frames = render(self.result, self.tmp / "out_src", frames_dir=frames_dir)
        self.assertEqual(frames.shape, (NUM_FRAMES, H, W, 3))
        # the background is the (darkened) source frame, not black
        self.assertGreater(int(frames[0].mean()), 0)

    def test_render_resolution_follows_the_source_frames(self):
        from PIL import Image

        sw, sh = 60, 50
        frames_dir = self.tmp / "frames2"
        frames_dir.mkdir()
        for index in range(NUM_FRAMES):
            arr = np.full((sh, sw, 3), 30, dtype=np.uint8)
            Image.fromarray(arr).save(frames_dir / f"frame_{index:06d}.png")

        frames = render(self.result, self.tmp / "out_src2", frames_dir=frames_dir)
        self.assertEqual(frames.shape, (NUM_FRAMES, sh, sw, 3))

    def test_max_points_limits_the_number_of_marks(self):
        many = self.tmp / "many"
        build_many(many, num_tracks=200)
        few = render(many, self.tmp / "out_many", max_points=200)
        limited = render(many, self.tmp / "out_few", max_points=10)
        self.assertEqual(few.shape, limited.shape)
        # fewer points -> strictly fewer lit pixels
        self.assertLess(
            int((limited.max(axis=-1) > 0).sum()), int((few.max(axis=-1) > 0).sum())
        )

    def test_trail_length_changes_the_overlay(self):
        short = render(self.result, self.tmp / "out_short", trail_length=1)
        long = render(self.result, self.tmp / "out_long", trail_length=8)
        self.assertEqual(short.shape, long.shape)
        self.assertGreater(
            int((long.max(axis=-1) > 0).sum()), int((short.max(axis=-1) > 0).sum()),
            "a longer trail must cover more pixels",
        )

    def test_dynamic_points_are_drawn_larger_than_static_ones(self):
        """Static points are 1px dots, dynamic ones become filled circles."""
        frames = render(self.result, self.tmp / "out_dyn")
        # the dynamic track sits at (50, 40) in frame 0 and is a filled circle
        dynamic_footprint = int((frames[0][36:45, 46:55].max(axis=-1) > 0).sum())
        # the static track ("late_birth") is not born yet in frame 0
        self.assertGreater(dynamic_footprint, 1)

        # in frame 2 the static track is born -> a single 1px dot
        static_footprint = int((frames[2][36:45, 61:70].max(axis=-1) > 0).sum())
        self.assertEqual(static_footprint, 1)

    def test_renderer_does_not_touch_the_input_files(self):
        import hashlib

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        before = {p.name: digest(p) for p in self.result.iterdir()}
        render(self.result, self.tmp / "out_ro")
        after = {p.name: digest(p) for p in self.result.iterdir()}
        self.assertEqual(before, after)

    def test_no_cuda_and_no_model_loading(self):
        import torch

        out = self.tmp / "out_nocuda"
        render(self.result, out)
        report = json.loads((out / "visualization_report.json").read_text())
        self.assertFalse(report["cuda_initialized"])
        self.assertNotIn("model", json.dumps(report["parameters"]))

    def test_the_worker_prints_a_stage_report(self):
        """The worker must report its stages to the console."""
        import contextlib
        import io

        sys.path.insert(0, str(PROJECT_ROOT / "worker"))
        import visualize as viz

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = viz.main([
                "--result-dir", str(self.result),
                "--output-dir", str(self.tmp / "out_report"),
                "--max-points", "5000",
            ])
        self.assertEqual(code, 0)
        text = buffer.getvalue()

        self.assertIn("TrackEverything - visualization", text)
        self.assertIn("[1/4] loading tracks.pt and geometry.pt", text)
        self.assertIn("[2/4] preparing the scene", text)
        self.assertIn("[3/4] rendering the tracks", text)
        self.assertIn("[4/4] writing the PNG frames", text)
        # the parameters are echoed in the banner
        self.assertIn("max points", text)
        # and the summary carries the result
        self.assertIn("cuda", text)
        self.assertIn("not used", text)


def build_many(directory: Path, num_tracks: int = 200) -> Path:
    """A result set with many tracks spread over the image."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    xs = rng.uniform(-0.4, 0.4, num_tracks)
    ys = rng.uniform(-0.3, 0.3, num_tracks)
    coords = np.zeros((num_tracks, NUM_FRAMES, 3), dtype=np.float32)
    coords[..., 0] = xs[:, None]
    coords[..., 1] = ys[:, None]
    coords[..., 2] = 2.0

    tracks = {
        "coords": torch.from_numpy(coords),
        "visibs": torch.full((num_tracks, NUM_FRAMES), 10.0),
        "dyn_logits": torch.zeros(num_tracks, NUM_FRAMES),
        "src_time": torch.zeros(num_tracks, dtype=torch.int64),
        "src_y": torch.zeros(num_tracks, dtype=torch.int64),
        "src_x": torch.zeros(num_tracks, dtype=torch.int64),
        "cell_to_token": torch.zeros(NUM_FRAMES, H // 4, W // 4, dtype=torch.int64),
    }
    torch.save(tracks, directory / "tracks.pt")

    intrinsics = torch.tensor(
        [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    geometry = {
        "norm_factor": 1.0,
        "depths": torch.ones(NUM_FRAMES, H, W),
        "intrinsics": intrinsics.expand(NUM_FRAMES, 3, 3).contiguous(),
        "extrinsics": torch.eye(4).expand(NUM_FRAMES, 4, 4).contiguous(),
    }
    torch.save(geometry, directory / "geometry.pt")
    (directory / "metadata.json").write_text(json.dumps({
        "output": {"num_frames": NUM_FRAMES, "num_tracks": num_tracks,
                   "working_size": {"width": W, "height": H}},
        "input": {"num_frames": NUM_FRAMES,
                  "original_size": {"width": W, "height": H}},
    }))
    return directory


class TestNodeLevel(VisualizeCase):
    def test_visualize_result_returns_an_image_tensor(self):
        import torch

        config = Config.resolve()
        config.worker_python = Path(sys.executable)
        (rendered,) = bridge.visualize_result(config, self._track_data())
        self.assertIsInstance(rendered, torch.Tensor)
        self.assertEqual(rendered.dtype, torch.float32)
        self.assertEqual(rendered.ndim, 4)
        self.assertEqual(tuple(rendered.shape), (NUM_FRAMES, H, W, 3))
        self.assertGreaterEqual(float(rendered.min()), 0.0)
        self.assertLessEqual(float(rendered.max()), 1.0)

    def test_frame_count_mismatch_is_reported(self):
        config = Config.resolve()
        config.worker_python = Path(sys.executable)
        images = np.zeros((NUM_FRAMES + 2, H, W, 3), dtype=np.float32)
        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.visualize_result(config, self._track_data(), images=images)
        self.assertIn("frames", str(ctx.exception))

    def test_missing_result_files_are_reported(self):
        config = Config.resolve()
        config.worker_python = Path(sys.executable)
        (self.result / "tracks.pt").unlink()
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.visualize_result(config, self._track_data())

    def test_node_returns_the_image_batch(self):
        config = Config.resolve()
        config.worker_python = Path(sys.executable)
        node = nodes.TrackEverythingVisualize()
        # the node resolves the config itself, so point the env var at a
        # worker that exists and let the node use the default config
        result = node.visualize(self._track_data())
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result[0].shape), (NUM_FRAMES, H, W, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
