#!/usr/bin/env python
"""Unit tests for the ``TrackEverything Save Data`` node.

Covers the reference / copy modes, byte-identical copies, the metadata
rewriting, missing and corrupted files, and the "never overwrite" rule.

Run with:
    pixi run python tests/test_save_data.py
"""

from __future__ import annotations

import hashlib
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

from comfy_nodes import bridge, nodes  # noqa: E402
from comfy_nodes.types import TrackData  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_result(directory: Path, *, num_tracks: int = 5, num_frames: int = 4) -> Path:
    """Write a small but valid result set."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)

    tracks = {
        "coords": torch.rand(num_tracks, num_frames, 3) * 0.5,
        "visibs": torch.rand(num_tracks, num_frames) * 4.0,
        "dyn_logits": torch.rand(num_tracks, num_frames) * 4.0 - 2.0,
        "src_time": torch.zeros(num_tracks, dtype=torch.int64),
        "src_y": torch.zeros(num_tracks, dtype=torch.int64),
        "src_x": torch.zeros(num_tracks, dtype=torch.int64),
        "cell_to_token": torch.zeros(num_frames, 4, 6, dtype=torch.int64),
    }
    torch.save(tracks, directory / "tracks.pt")

    geometry = {
        "norm_factor": 1.25,
        "depths": torch.ones(num_frames, 8, 10),
        "intrinsics": torch.eye(3).expand(num_frames, 3, 3).contiguous(),
        "extrinsics": torch.eye(4).expand(num_frames, 4, 4).contiguous(),
    }
    torch.save(geometry, directory / "geometry.pt")

    metadata = {
        "schema_version": 1,
        "input": {"num_frames": num_frames, "original_size": {"width": 10, "height": 8}},
        "output": {
            "num_frames": num_frames,
            "num_tracks": num_tracks,
            "working_size": {"width": 10, "height": 8},
        },
        "files": {"tracks": "tracks.pt", "geometry": "geometry.pt",
                  "metadata": "metadata.json"},
        "worker": {
            "command": ["/abs/python", "/abs/worker.py", "--output-dir", str(directory)],
            "returncode": 0,
        },
        "environment": {"project_root": "/abs/project", "output_root": str(directory)},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "worker.log").write_text("log line\n")
    return directory


class SaveDataCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="te_save_test_")
        self.tmp = Path(self._tmp)
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self.source = write_result(self.tmp / "source")
        self.output_root = self.tmp / "out"

    def _track_data(self, directory: Path | None = None) -> TrackData:
        directory = directory or self.source
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}
        return bridge.build_track_data(directory, metadata)


class TestRegistration(SaveDataCase):
    def test_output_node_is_registered(self):
        self.assertIn("TrackEverythingSaveData", nodes.NODE_CLASS_MAPPINGS)
        self.assertIs(
            nodes.NODE_CLASS_MAPPINGS["TrackEverythingSaveData"],
            nodes.TrackEverythingSaveData,
        )

    def test_output_node_flag(self):
        self.assertTrue(nodes.TrackEverythingSaveData.OUTPUT_NODE)

    def test_display_name(self):
        self.assertEqual(
            nodes.NODE_DISPLAY_NAME_MAPPINGS["TrackEverythingSaveData"],
            "TrackEverything Save Data",
        )

    def test_input_interface(self):
        required = nodes.TrackEverythingSaveData.INPUT_TYPES()["required"]
        self.assertIn("track_data", required)
        self.assertIn("save_mode", required)
        self.assertIn("filename_prefix", required)
        self.assertEqual(required["save_mode"][0], ["reference", "copy"])
        self.assertEqual(required["save_mode"][1]["default"], "reference")

    def test_returns_ui_text_and_result(self):
        node = nodes.TrackEverythingSaveData()
        out = node.save(self._track_data(), save_mode="reference")
        self.assertIn("ui", out)
        self.assertIn("text", out["ui"])
        self.assertTrue(out["ui"]["text"][0])
        self.assertIsInstance(out["result"][0], TrackData)


class TestReferenceMode(SaveDataCase):
    def test_does_not_copy_anything(self):
        before = sorted(p.name for p in self.output_root.glob("*")) if self.output_root.exists() else []
        saved, message = bridge.save_result(
            self._track_data(), save_mode="reference", output_root=self.output_root
        )
        after = sorted(p.name for p in self.output_root.glob("*")) if self.output_root.exists() else []
        self.assertEqual(before, after)
        self.assertEqual(str(saved.result_dir), str(self.source))
        self.assertIn("referenced", message)

    def test_source_files_are_untouched(self):
        hashes = {
            name: sha256(self.source / name)
            for name in ("tracks.pt", "geometry.pt", "metadata.json")
        }
        mtimes = {
            name: (self.source / name).stat().st_mtime_ns
            for name in ("tracks.pt", "geometry.pt", "metadata.json")
        }
        bridge.save_result(self._track_data(), save_mode="reference")
        for name in ("tracks.pt", "geometry.pt", "metadata.json"):
            self.assertEqual(sha256(self.source / name), hashes[name])
            self.assertEqual((self.source / name).stat().st_mtime_ns, mtimes[name])

    def test_requires_all_three_files(self):
        (self.source / "geometry.pt").unlink()
        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.save_result(self._track_data(), save_mode="reference")
        self.assertIn("geometry.pt", str(ctx.exception))


class TestCopyMode(SaveDataCase):
    def test_copies_the_files(self):
        saved, message = bridge.save_result(
            self._track_data(), save_mode="copy",
            filename_prefix="myrun", output_root=self.output_root,
        )
        destination = Path(saved.result_dir)
        self.assertTrue(destination.is_dir())
        self.assertTrue((destination / "tracks.pt").is_file())
        self.assertTrue((destination / "geometry.pt").is_file())
        self.assertTrue((destination / "metadata.json").is_file())
        self.assertIn("copied", message)
        self.assertIn("myrun", destination.name)

    def test_pt_files_are_byte_identical(self):
        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy", output_root=self.output_root
        )
        destination = Path(saved.result_dir)
        for name in ("tracks.pt", "geometry.pt"):
            self.assertEqual(
                sha256(self.source / name), sha256(destination / name),
                f"{name} was modified during the copy",
            )

    def test_tensor_values_are_unchanged(self):
        import torch

        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy", output_root=self.output_root
        )
        destination = Path(saved.result_dir)
        a = torch.load(self.source / "tracks.pt", map_location="cpu", weights_only=True)
        b = torch.load(destination / "tracks.pt", map_location="cpu", weights_only=True)
        self.assertEqual(sorted(a.keys()), sorted(b.keys()))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertEqual(a[key].dtype, b[key].dtype)

    def test_worker_log_is_copied(self):
        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy", output_root=self.output_root
        )
        destination = Path(saved.result_dir)
        self.assertTrue((destination / "worker.log").is_file())
        self.assertEqual(
            (destination / "worker.log").read_text(),
            (self.source / "worker.log").read_text(),
        )

    def test_metadata_keeps_the_provenance(self):
        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy", output_root=self.output_root
        )
        metadata = json.loads((Path(saved.result_dir) / "metadata.json").read_text())
        self.assertIn("copy", metadata)
        self.assertEqual(
            metadata["copy"]["source_result_dir"], str(self.source.resolve())
        )
        self.assertEqual(metadata["copy"]["save_mode"], "copy")

    def test_metadata_paths_follow_the_copy(self):
        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy", output_root=self.output_root
        )
        destination = Path(saved.result_dir).resolve()
        metadata = json.loads((Path(saved.result_dir) / "metadata.json").read_text())

        # the *live* references must point at the copy, not at the source
        self.assertNotIn(
            str(self.source.resolve()),
            json.dumps(metadata["worker"]),
            "the worker command still references the source directory",
        )
        self.assertNotIn(
            str(self.source.resolve()),
            metadata["environment"]["output_root"],
            "environment.output_root still references the source directory",
        )
        self.assertIn(str(destination), json.dumps(metadata["worker"]))
        self.assertEqual(metadata["environment"]["output_root"], str(destination))

        # the relative file names stay relative and valid
        self.assertEqual(metadata["files"]["tracks"], "tracks.pt")
        self.assertTrue((destination / metadata["files"]["tracks"]).is_file())

        # the environment description is provenance and stays untouched
        self.assertEqual(metadata["environment"]["project_root"], "/abs/project")

        # the provenance of the copy is recorded explicitly
        self.assertEqual(
            metadata["copy"]["source_result_dir"], str(self.source.resolve())
        )

    def test_result_dir_is_recorded(self):
        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy", output_root=self.output_root
        )
        metadata = json.loads((Path(saved.result_dir) / "metadata.json").read_text())
        self.assertEqual(metadata["result_dir"], str(Path(saved.result_dir).resolve()))

    def test_copied_result_can_be_loaded_again(self):
        saved, _ = bridge.save_result(
            self._track_data(), save_mode="copy", output_root=self.output_root
        )
        reloaded = bridge.load_result_dir(str(saved.result_dir))
        self.assertIsInstance(reloaded, TrackData)
        self.assertEqual(reloaded.num_tracks, self._track_data().num_tracks)
        self.assertEqual(reloaded.num_frames, self._track_data().num_frames)

    def test_repeated_copies_do_not_overwrite(self):
        first, _ = bridge.save_result(
            self._track_data(), save_mode="copy",
            filename_prefix="run", output_root=self.output_root,
        )
        second, _ = bridge.save_result(
            self._track_data(), save_mode="copy",
            filename_prefix="run", output_root=self.output_root,
        )
        self.assertNotEqual(str(first.result_dir), str(second.result_dir))
        self.assertTrue(Path(first.result_dir).is_dir())
        self.assertTrue(Path(second.result_dir).is_dir())
        for directory in (first.result_dir, second.result_dir):
            self.assertTrue((Path(directory) / "tracks.pt").is_file())

    def test_no_incomplete_directory_is_left_behind(self):
        # corrupt the metadata so the copy fails half way through
        (self.source / "metadata.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.save_result(
                self._track_data(), save_mode="copy", output_root=self.output_root
            )
        leftovers = self._result_like_leftovers()
        self.assertEqual(
            leftovers, [], f"incomplete result left behind: {leftovers}"
        )

    def _result_like_leftovers(self) -> list:
        """Directories that look like a saved result (or a staging folder)."""
        if not self.output_root.exists():
            return []
        found = []
        for path in self.output_root.rglob("*"):
            if not path.is_dir():
                continue
            if (path / "tracks.pt").exists() or ".tmp-" in path.name:
                found.append(path)
        return found

    def test_missing_file_is_reported(self):
        (self.source / "tracks.pt").unlink()
        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            bridge.save_result(
                self._track_data(), save_mode="copy", output_root=self.output_root
            )
        self.assertIn("tracks.pt", str(ctx.exception))
        self.assertEqual(self._result_like_leftovers(), [])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(bridge.TrackEverythingError):
            bridge.save_result(self._track_data(), save_mode="bogus")

    def test_node_does_not_use_cuda(self):
        import torch

        # redirect the node's output root so the test does not pollute the
        # real output directory (this also exercises the env var config)
        previous = os.environ.get("TRACKEVERYTHING_OUTPUT_ROOT")
        os.environ["TRACKEVERYTHING_OUTPUT_ROOT"] = str(self.output_root)
        try:
            before = torch.cuda.is_initialized()
            node = nodes.TrackEverythingSaveData()
            node.save(self._track_data(), save_mode="copy")
            self.assertEqual(before, torch.cuda.is_initialized())
        finally:
            if previous is None:
                os.environ.pop("TRACKEVERYTHING_OUTPUT_ROOT", None)
            else:
                os.environ["TRACKEVERYTHING_OUTPUT_ROOT"] = previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
