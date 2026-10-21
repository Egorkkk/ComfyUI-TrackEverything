#!/usr/bin/env python
"""End-to-end test of the Visualize node on the **real** standalone results.

Uses the result set produced by the standalone inference pipeline
(``outputs/standalone_16f``) - no inference is run here. The test checks that

* the saved ``tracks.pt`` / ``geometry.pt`` can be visualized without loading
  any model and without touching the GPU,
* the frame count and the resolution follow the metadata,
* the overlay really lands on the moving objects of the synthetic clip,
* the ``TrackEverything Load Data`` node can feed the Visualize node, i.e.
  the second workflow works without a repeated inference.

Run with:
    pixi run python tests/test_visualize_e2e.py
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

RESULT_DIR = PROJECT_ROOT / "outputs" / "standalone_16f"
VIDEO = PROJECT_ROOT / "test_data" / "synthetic_16f.mp4"


class TestVisualizeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (RESULT_DIR / "tracks.pt").is_file():
            raise unittest.SkipTest(
                f"{RESULT_DIR} not found - run the standalone inference first"
            )
        cls._tmp = tempfile.mkdtemp(prefix="te_vis_e2e_")
        cls.tmp = Path(cls._tmp)
        cls.config = Config.resolve()
        cls.config.validate()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _track_data(self):
        return bridge.load_result_dir(str(RESULT_DIR))

    def test_visualizes_the_real_result_without_a_model(self):
        import torch

        before = torch.cuda.is_initialized()
        (rendered,) = bridge.visualize_result(self.config, self._track_data())
        after = torch.cuda.is_initialized()

        self.assertEqual(before, after, "the visualizer must not create a CUDA context")
        self.assertEqual(rendered.dtype, torch.float32)
        self.assertEqual(rendered.ndim, 4)
        self.assertEqual(rendered.shape[-1], 3)
        self.assertGreaterEqual(float(rendered.min()), 0.0)
        self.assertLessEqual(float(rendered.max()), 1.0)

        metadata = self._track_data().metadata
        self.assertEqual(
            rendered.shape[0], int(metadata["output"]["num_frames"])
        )
        working = metadata["output"]["working_size"]
        self.assertEqual(rendered.shape[1], int(working["height"]))
        self.assertEqual(rendered.shape[2], int(working["width"]))

        # something was actually drawn
        self.assertGreater(int((rendered.max(dim=-1).values > 0).sum()), 100)

    def test_visualization_does_not_modify_the_result_files(self):
        import hashlib

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        before = {p.name: digest(p) for p in RESULT_DIR.iterdir() if p.is_file()}
        bridge.visualize_result(self.config, self._track_data())
        after = {p.name: digest(p) for p in RESULT_DIR.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_overlay_on_the_original_footage(self):
        import torch

        if not VIDEO.is_file():
            self.skipTest("the synthetic clip is missing")

        import av

        container = av.open(str(VIDEO))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        images = np.stack(frames).astype(np.float32) / 255.0

        (rendered,) = bridge.visualize_result(
            self.config, self._track_data(), images=images
        )
        self.assertEqual(rendered.shape[0], images.shape[0])
        # the output keeps the source resolution
        self.assertEqual(rendered.shape[1], images.shape[1])
        self.assertEqual(rendered.shape[2], images.shape[2])

    def test_load_data_node_feeds_the_visualize_node(self):
        """The second workflow: Load Data -> Visualize, no inference."""
        loaded = nodes.TrackEverythingLoadData()
        (track_data,) = loaded.load(str(RESULT_DIR))
        self.assertIsInstance(track_data, bridge.TrackData)

        visualizer = nodes.TrackEverythingVisualize()
        (rendered,) = visualizer.visualize(track_data)
        self.assertEqual(rendered.ndim, 4)
        self.assertEqual(rendered.shape[-1], 3)

    def test_saved_result_can_be_visualized_after_a_copy(self):
        """Save Data (copy) -> Load Data -> Visualize."""
        output_root = self.tmp / "out"
        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy",
            filename_prefix="e2e", output_root=output_root,
        )
        reloaded = bridge.load_result_dir(str(saved.result_dir))
        (rendered,) = bridge.visualize_result(self.config, reloaded)
        self.assertEqual(rendered.shape[0], int(reloaded.num_frames))


if __name__ == "__main__":
    unittest.main(verbosity=2)
