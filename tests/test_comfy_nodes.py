#!/usr/bin/env python
"""Unit tests for the TrackEverything ComfyUI nodes.

The bridge is exercised against a **fake worker** (``tests/fake_worker.py``)
so no model is loaded and no GPU is needed - the tests run in well under a
second. A separate slow test (``tests/test_e2e.sh``) covers the real path.

Run with:
    pixi run python -m unittest discover -s tests -v
    # or
    pixi run python tests/test_comfy_nodes.py
"""

from __future__ import annotations

import json
import os
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
from comfy_nodes.config import Config, assert_isolated  # noqa: E402
from comfy_nodes.types import TrackData  # noqa: E402

FAKE_WORKER = TESTS_DIR / "fake_worker.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_config(tmp: Path, worker_script: Path = FAKE_WORKER) -> Config:
    """A Config that points at the fake worker and a temporary output root."""
    return Config(
        project_root=PROJECT_ROOT,
        worker_python=Path(sys.executable),
        output_root=tmp / "out",
        upstream_dir=PROJECT_ROOT / "external" / "trackeverything",
        worker_script=worker_script,
        model_repo="katefgroup/TrackEverything",
        pi3_checkpoint="yyfz233/Pi3",
        device="cuda",
        geometry_backend="pi3",
    )


def make_images(frames: int = 4, height: int = 8, width: int = 12) -> np.ndarray:
    """A float32 IMAGE batch in [0, 1], shape (B, H, W, 3)."""
    rng = np.random.default_rng(7)
    return rng.random((frames, height, width, 3), dtype=np.float32)


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="te_nodes_test_")
        self.tmp = Path(self._tmp)
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# image batch handling
# ---------------------------------------------------------------------------
class TestImageBatch(TempDirCase):
    def test_conversion_matches_upstream_formula(self):
        images = make_images(frames=3)
        batch = bridge.validate_image_batch(images)

        expected = (images.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        self.assertEqual(batch.frames.dtype, np.uint8)
        self.assertTrue(np.array_equal(batch.frames, expected))
        self.assertEqual(batch.original_size, {"width": 12, "height": 8})
        self.assertEqual(batch.num_frames, 3)
        self.assertEqual(batch.source_dtype, "float32")
        self.assertEqual(batch.source_range, "[0, 1]")

    def test_uint8_batch_is_passed_through(self):
        images = (make_images(frames=2) * 255).astype(np.uint8)
        batch = bridge.validate_image_batch(images)
        self.assertTrue(np.array_equal(batch.frames, images))
        self.assertEqual(batch.source_dtype, "uint8")

    def test_values_are_clipped(self):
        images = np.zeros((1, 2, 2, 3), dtype=np.float32)
        images[0, 0, 0] = (2.0, -1.0, 0.5)
        batch = bridge.validate_image_batch(images)
        self.assertEqual(list(batch.frames[0, 0, 0]), [255, 0, 127])

    def test_truncation_not_rounding(self):
        # upstream: (x * 255).clip(0, 255).astype(uint8) -> truncation
        images = np.zeros((1, 2, 2, 3), dtype=np.float32)
        images[0, 0, 0] = (0.5 / 255.0, 0.0, 0.0)
        batch = bridge.validate_image_batch(images)
        self.assertEqual(int(batch.frames[0, 0, 0, 0]), 0)

    def test_rejects_none(self):
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.validate_image_batch(None)

    def test_rejects_empty_batch(self):
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.validate_image_batch(np.zeros((0, 8, 8, 3), dtype=np.float32))

    def test_rejects_wrong_channel_count(self):
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.validate_image_batch(np.zeros((2, 8, 8, 4), dtype=np.float32))

    def test_rejects_wrong_ndim(self):
        # (H, W) without a channel axis is not a valid frame
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.validate_image_batch(np.zeros((8, 8), dtype=np.float32))

    def test_single_frame_is_accepted_as_a_batch_of_one(self):
        batch = bridge.validate_image_batch(np.zeros((8, 12, 3), dtype=np.float32))
        self.assertEqual(batch.num_frames, 1)
        self.assertEqual(batch.original_size, {"width": 12, "height": 8})

    def test_rejects_non_float_non_uint8(self):
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.validate_image_batch(np.zeros((2, 8, 8, 3), dtype=np.int32))

    def test_rejects_tiny_frames(self):
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.validate_image_batch(np.zeros((2, 1, 1, 3), dtype=np.float32))


class TestFrameWriting(TempDirCase):
    def test_frame_order_and_names(self):
        images = make_images(frames=12)
        batch = bridge.validate_image_batch(images)
        frames_dir = self.tmp / "frames"
        names = bridge.write_frames(batch, frames_dir)

        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], "frame_000000.png")
        self.assertEqual(names[-1], "frame_000011.png")

        # upstream sorts by file name -> the sorted order must equal the
        # original order of the batch
        on_disk = sorted(p.name for p in frames_dir.glob("*.png"))
        self.assertEqual(on_disk, sorted(names))

    def test_png_content_matches_the_batch(self):
        from PIL import Image

        images = make_images(frames=3)
        batch = bridge.validate_image_batch(images)
        frames_dir = self.tmp / "frames"
        bridge.write_frames(batch, frames_dir)

        for index in range(3):
            with Image.open(frames_dir / f"frame_{index:06d}.png") as img:
                self.assertEqual(img.size, (12, 8))     # (W, H)
                self.assertEqual(img.mode, "RGB")
                arr = np.asarray(img)
            self.assertTrue(np.array_equal(arr, batch.frames[index]))


# ---------------------------------------------------------------------------
# worker invocation
# ---------------------------------------------------------------------------
class TestWorkerArgv(TempDirCase):
    def test_argv_is_explicit_and_complete(self):
        cfg = make_config(self.tmp)
        argv = bridge.build_worker_argv(
            cfg,
            frames_dir=Path("/tmp/frames"),
            output_dir=Path("/tmp/out"),
            num_iters=5,
            voxel_size=0.05,
            mem_efficient=True,
            max_frames=8,
        )

        self.assertEqual(argv[0], str(cfg.worker_python))
        self.assertEqual(argv[1], str(cfg.worker_script))
        self.assertIn("--frames-dir", argv)
        self.assertIn("--output-dir", argv)
        self.assertIn("--num-iters", argv)
        self.assertEqual(argv[argv.index("--num-iters") + 1], "5")
        self.assertEqual(argv[argv.index("--voxel-size") + 1], "0.05")
        self.assertIn("--mem-efficient", argv)
        self.assertNotIn("--no-mem-efficient", argv)
        self.assertEqual(argv[argv.index("--max-frames") + 1], "8")
        self.assertEqual(argv[argv.index("--geometry-backend") + 1], "pi3")

        # every element is a plain string -> safe for a shell-less exec
        self.assertTrue(all(isinstance(a, str) for a in argv))

    def test_no_mem_efficient_flag(self):
        cfg = make_config(self.tmp)
        argv = bridge.build_worker_argv(
            cfg, Path("/f"), Path("/o"), mem_efficient=False
        )
        self.assertIn("--no-mem-efficient", argv)
        self.assertNotIn("--mem-efficient", argv)


class TestWorkerExecution(TempDirCase):
    def test_successful_run(self):
        cfg = make_config(self.tmp)
        out = self.tmp / "out"
        argv = bridge.build_worker_argv(cfg, self.tmp / "frames", out)
        # the fake worker needs the frames to exist
        (self.tmp / "frames").mkdir(parents=True, exist_ok=True)

        result = bridge.run_worker(cfg, argv, out)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.log_path.is_file())
        self.assertIn("fake-worker", result.stdout)
        self.assertTrue((out / "tracks.pt").is_file())
        self.assertTrue((out / "worker_report.json").is_file())

    def test_worker_failure_is_reported(self):
        cfg = make_config(self.tmp)
        out = self.tmp / "out"
        argv = bridge.build_worker_argv(cfg, self.tmp / "frames", out)
        argv.append("--fail")
        (self.tmp / "frames").mkdir(parents=True, exist_ok=True)

        result = bridge.run_worker(cfg, argv, out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA out of memory", result.stderr)
        # the log must be preserved for diagnostics
        self.assertTrue(result.log_path.is_file())
        self.assertIn("simulated worker failure", result.log_path.read_text())

    def test_timeout_is_reported(self):
        cfg = make_config(self.tmp)
        out = self.tmp / "out"
        argv = bridge.build_worker_argv(cfg, self.tmp / "frames", out)
        argv.append("--hang")
        (self.tmp / "frames").mkdir(parents=True, exist_ok=True)

        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.run_worker(cfg, argv, out, timeout=3)
        self.assertIn("timed out", str(ctx.exception))

    def test_missing_worker_binary_is_reported(self):
        cfg = make_config(self.tmp)
        cfg.worker_python = Path("/nonexistent/python")
        out = self.tmp / "out"
        argv = bridge.build_worker_argv(cfg, self.tmp / "frames", out)
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.run_worker(cfg, argv, out)


# ---------------------------------------------------------------------------
# full bridge pipeline (fake worker)
# ---------------------------------------------------------------------------
class TestRunTrack(TempDirCase):
    def test_end_to_end_with_fake_worker(self):
        cfg = make_config(self.tmp)
        images = make_images(frames=5)

        track_data = bridge.run_track(cfg, images, output_dir=self.tmp / "result")

        self.assertIsInstance(track_data, TrackData)
        self.assertTrue(track_data.exists())
        self.assertTrue(Path(track_data.tracks_path).is_file())
        self.assertTrue(Path(track_data.geometry_path).is_file())
        self.assertTrue(Path(track_data.metadata_path).is_file())

        meta = track_data.metadata
        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["input"]["num_frames"], 5)
        self.assertEqual(meta["input"]["original_size"], {"width": 12, "height": 8})
        self.assertEqual(meta["output"]["num_frames"], 5)
        self.assertEqual(meta["output"]["num_tracks"], 7)
        self.assertEqual(meta["output"]["working_size"], {"width": 512, "height": 384})
        self.assertEqual(meta["output"]["feature_grid"], {"height": 96, "width": 128})
        self.assertEqual(meta["inference"]["num_iters"], 3)
        self.assertEqual(meta["inference"]["voxel_size"], 0.02)
        self.assertTrue(meta["inference"]["mem_efficient"])

    def test_metadata_contains_the_coordinate_transform(self):
        cfg = make_config(self.tmp)
        track_data = bridge.run_track(
            cfg, make_images(frames=4), output_dir=self.tmp / "result"
        )
        transform = track_data.coordinate_transform
        self.assertEqual(transform["norm_factor"], 1.5)
        self.assertIn("unnormalise", transform)
        self.assertEqual(transform["sentinel"]["value"], -1000)
        # 384 -> 512 wide, 256 -> 384 high is the model's fixed resolution
        self.assertAlmostEqual(transform["pixel_mapping"]["scale_x"], 512 / 12)
        self.assertAlmostEqual(transform["pixel_mapping"]["scale_y"], 384 / 8)

    def test_metadata_records_the_worker_command_and_versions(self):
        cfg = make_config(self.tmp)
        track_data = bridge.run_track(
            cfg, make_images(frames=2), output_dir=self.tmp / "result"
        )
        meta = track_data.metadata
        self.assertEqual(meta["worker"]["returncode"], 0)
        self.assertTrue(meta["worker"]["command"][0].endswith("python"))
        self.assertIn("upstream_commit", meta["versions"])
        self.assertEqual(meta["files"]["tracks"], "tracks.pt")
        self.assertEqual(meta["files"]["geometry"], "geometry.pt")

    def test_worker_failure_produces_a_clear_error(self):
        cfg = make_config(self.tmp)
        # the fake worker fails when it sees the --fail flag; inject it by
        # wrapping the script
        failing = self.tmp / "failing_worker.py"
        failing.write_text(
            f"import runpy, sys\n"
            f"sys.argv.append('--fail')\n"
            f"runpy.run_path({str(FAKE_WORKER)!r}, run_name='__main__')\n"
        )
        cfg.worker_script = failing

        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.run_track(cfg, make_images(frames=2), output_dir=self.tmp / "result")
        message = str(ctx.exception)
        self.assertIn("worker failed", message)
        self.assertIn("Log:", message)

    def test_missing_results_produce_a_clear_error(self):
        cfg = make_config(self.tmp)
        missing = self.tmp / "missing_worker.py"
        missing.write_text(
            f"import runpy, sys\n"
            f"sys.argv.append('--missing-outputs')\n"
            f"runpy.run_path({str(FAKE_WORKER)!r}, run_name='__main__')\n"
        )
        cfg.worker_script = missing

        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.run_track(cfg, make_images(frames=2), output_dir=self.tmp / "result")
        self.assertIn("did not produce", str(ctx.exception))

    def test_frames_are_cleaned_up(self):
        cfg = make_config(self.tmp)
        before = set(Path(tempfile.gettempdir()).glob("trackeverything_frames_*"))
        bridge.run_track(cfg, make_images(frames=2), output_dir=self.tmp / "result")
        after = set(Path(tempfile.gettempdir()).glob("trackeverything_frames_*"))
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# Load Data node
# ---------------------------------------------------------------------------
class TestLoadData(TempDirCase):
    def _make_result(self) -> Path:
        cfg = make_config(self.tmp)
        result = self.tmp / "result"
        bridge.run_track(cfg, make_images(frames=3), output_dir=result)
        return result

    def test_loads_an_existing_result(self):
        result = self._make_result()
        track_data = bridge.load_result_dir(str(result))
        self.assertIsInstance(track_data, TrackData)
        self.assertTrue(track_data.exists())
        self.assertEqual(track_data.num_frames, 3)
        self.assertEqual(track_data.num_tracks, 7)

    def test_missing_directory_is_reported(self):
        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.load_result_dir(str(self.tmp / "nope"))
        self.assertIn("does not exist", str(ctx.exception))

    def test_incomplete_directory_is_reported(self):
        result = self.tmp / "incomplete"
        result.mkdir()
        (result / "tracks.pt").write_bytes(b"x")
        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.load_result_dir(str(result))
        self.assertIn("missing", str(ctx.exception))

    def test_corrupt_metadata_is_reported(self):
        result = self._make_result()
        (result / "metadata.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.load_result_dir(str(result))
        self.assertIn("Cannot read", str(ctx.exception))

    def test_does_not_touch_the_gpu(self):
        import torch

        result = self._make_result()
        before = torch.cuda.is_initialized()
        bridge.load_result_dir(str(result))
        after = torch.cuda.is_initialized()
        self.assertEqual(before, after, "loading results must not create a CUDA context")


# ---------------------------------------------------------------------------
# node level
# ---------------------------------------------------------------------------
class TestNodes(TempDirCase):
    def test_node_interface(self):
        self.assertEqual(nodes.TrackEverythingTrack.RETURN_TYPES, ("TRACK_DATA",))
        self.assertEqual(nodes.TrackEverythingLoadData.RETURN_TYPES, ("TRACK_DATA",))
        inputs = nodes.TrackEverythingTrack.INPUT_TYPES()["required"]
        self.assertIn("images", inputs)
        self.assertEqual(inputs["num_iters"][1]["default"], 3)
        self.assertEqual(inputs["voxel_size"][1]["default"], 0.02)
        self.assertEqual(inputs["mem_efficient"][1]["default"], True)
        self.assertIn("result_dir", nodes.TrackEverythingLoadData.INPUT_TYPES()["required"])

    def test_both_nodes_return_the_same_payload_type(self):
        cfg = make_config(self.tmp)
        result = self.tmp / "result"
        bridge.run_track(cfg, make_images(frames=2), output_dir=result)

        loaded = bridge.load_result_dir(str(result))
        produced = bridge.build_track_data(result, json.loads(
            (result / "metadata.json").read_text()))

        self.assertIs(type(loaded), type(produced))
        self.assertEqual(sorted(loaded.keys()), sorted(produced.keys()))
        self.assertEqual(loaded.num_frames, produced.num_frames)
        self.assertEqual(loaded.num_tracks, produced.num_tracks)
        self.assertEqual(
            sorted(loaded.coordinate_transform.keys()),
            sorted(produced.coordinate_transform.keys()),
        )

    def test_load_data_node_requires_a_path(self):
        node = nodes.TrackEverythingLoadData()
        with self.assertRaises(bridge.TrackEverythingError):
            node.load("")

    def test_load_data_node_returns_track_data(self):
        cfg = make_config(self.tmp)
        result = self.tmp / "result"
        bridge.run_track(cfg, make_images(frames=2), output_dir=result)
        node = nodes.TrackEverythingLoadData()
        (payload,) = node.load(str(result))
        self.assertIsInstance(payload, TrackData)


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------
class TestIsolation(unittest.TestCase):
    def test_upstream_packages_are_not_imported(self):
        self.assertNotIn("trackeverything", sys.modules)
        self.assertNotIn("torch_scatter", sys.modules)

    def test_isolated_site_packages_are_not_on_sys_path(self):
        cfg = Config.resolve()
        try:
            same_interpreter = (
                Path(sys.executable).resolve() == cfg.worker_python.resolve()
            )
        except OSError:
            same_interpreter = False

        if same_interpreter:
            # the harness itself runs inside the isolated environment
            self.skipTest("test harness runs inside the worker environment")

        isolated = str(cfg.worker_python.parent.parent / "lib")
        for entry in sys.path:
            self.assertFalse(
                entry and isolated in str(entry),
                f"worker site-packages leaked into sys.path: {entry}",
            )

    def test_assert_isolated_passes_in_this_process(self):
        assert_isolated(Config.resolve())

    def test_config_is_not_hardcoded_to_one_user(self):
        source = (PROJECT_ROOT / "comfy_nodes" / "config.py").read_text()
        self.assertNotIn("/home/ocdev", source)
        self.assertNotIn("/home/egorkkk", source)

    def test_package_imports_in_a_foreign_interpreter_without_torch(self):
        """The ComfyUI side must not depend on the worker's packages.

        Runs the import in a *different* interpreter (the system python 3.12
        here) and checks that neither torch nor numpy nor the upstream
        packages are pulled in.
        """
        import subprocess

        candidates = [
            "/usr/bin/python3",
            shutil.which("python3") or "",
        ]
        interpreter = next((c for c in candidates if c and Path(c).exists()), None)
        if interpreter is None:
            self.skipTest("no second interpreter available")

        code = (
            "import sys; sys.path.insert(0, %r);"
            "import comfy_nodes;"
            "from comfy_nodes.config import Config, assert_isolated;"
            "assert_isolated(Config.resolve());"
            "assert 'torch' not in sys.modules, 'torch leaked';"
            "assert 'numpy' not in sys.modules, 'numpy leaked';"
            "assert 'trackeverything' not in sys.modules;"
            "assert 'torch_scatter' not in sys.modules;"
            "print('OK')" % str(PROJECT_ROOT)
        )
        proc = subprocess.run(
            [interpreter, "-c", code], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(
            proc.returncode, 0,
            f"import failed in {interpreter}:\n{proc.stdout}\n{proc.stderr}",
        )
        self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
