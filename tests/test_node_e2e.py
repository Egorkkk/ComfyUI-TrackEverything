#!/usr/bin/env python
"""End-to-end test of the Track node with the **real** worker.

This one loads the actual model on the GPU, so it is slow (~40 s). It
verifies that the whole chain works:

    IMAGE batch -> PNG frames -> worker process -> tracks.pt / geometry.pt
    -> metadata.json -> TRACK_DATA

Run with:
    pixi run python tests/test_node_e2e.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from comfy_nodes import bridge  # noqa: E402
from comfy_nodes.config import Config  # noqa: E402
from comfy_nodes.types import TrackData  # noqa: E402


def load_synthetic_clip(path: Path, max_frames: int = 8) -> np.ndarray:
    """Read the synthetic test clip as a float32 IMAGE batch (B, H, W, 3)."""
    import av

    container = av.open(str(path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    arr = np.stack(frames[:max_frames]).astype(np.float32) / 255.0
    return arr


class TestNodeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.video = PROJECT_ROOT / "test_data" / "synthetic_16f.mp4"
        if not cls.video.exists():
            raise unittest.SkipTest(
                f"{cls.video} is missing - run scripts/make_test_video.py first"
            )
        cls._tmp = tempfile.mkdtemp(prefix="te_node_e2e_")
        cls.tmp = Path(cls._tmp)
        cls.config = Config.resolve()
        cls.config.validate()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_real_inference_through_the_node(self):
        images = load_synthetic_clip(self.video, max_frames=8)
        out_dir = self.tmp / "result"

        started = time.time()
        track_data = bridge.run_track(
            self.config,
            images,
            num_iters=3,
            voxel_size=0.02,
            mem_efficient=True,
            output_dir=out_dir,
        )
        elapsed = time.time() - started

        self.assertIsInstance(track_data, TrackData)
        self.assertTrue(track_data.exists())

        meta = track_data.metadata
        print(f"\n  inference took {elapsed:.1f}s")
        print(f"  tracks       : {track_data.num_tracks}")
        print(f"  frames       : {track_data.num_frames}")
        print(f"  original size: {track_data.original_size}")
        print(f"  working size : {track_data.working_size}")
        print(f"  norm_factor  : {track_data.coordinate_transform['norm_factor']}")

        # --- the numbers must match the standalone pipeline -----------------
        self.assertEqual(meta["input"]["num_frames"], 8)
        self.assertEqual(meta["input"]["original_size"], {"width": 384, "height": 256})
        self.assertEqual(meta["output"]["num_frames"], 8)
        # the standalone run of the same clip produced 2415 tracks at 8 frames
        self.assertEqual(meta["output"]["num_tracks"], 2415)
        self.assertEqual(meta["output"]["working_size"], {"width": 512, "height": 384})
        self.assertAlmostEqual(
            meta["coordinate_transform"]["norm_factor"], 1.2362807989120483, places=6
        )

        # --- the payload must be the upstream format ------------------------
        import torch

        tracks = torch.load(track_data.tracks_path, map_location="cpu", weights_only=True)
        self.assertEqual(
            sorted(tracks.keys()),
            ["cell_to_token", "coords", "dyn_logits", "src_time", "src_x", "src_y", "visibs"],
        )
        self.assertEqual(tuple(tracks["coords"].shape), (2415, 8, 3))
        self.assertEqual(tuple(tracks["cell_to_token"].shape), (8, 96, 128))

        geometry = torch.load(track_data.geometry_path, map_location="cpu", weights_only=True)
        self.assertEqual(
            sorted(geometry.keys()), ["depths", "extrinsics", "intrinsics", "norm_factor"]
        )
        self.assertEqual(tuple(geometry["depths"].shape), (8, 384, 512))

        # --- the load node must return the same payload ---------------------
        reloaded = bridge.load_result_dir(str(out_dir))
        self.assertIs(type(reloaded), type(track_data))
        self.assertEqual(reloaded.num_tracks, track_data.num_tracks)
        self.assertEqual(reloaded.num_frames, track_data.num_frames)

        # --- the worker log must be preserved --------------------------------
        log = Path(track_data.worker_log_path)
        self.assertTrue(log.is_file())
        log_text = log.read_text()
        self.assertIn("worker", log_text)
        self.assertIn("--frames-dir", log_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
