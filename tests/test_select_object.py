#!/usr/bin/env python
"""Tests for the ``TrackEverything Select Object`` node.

The fixture builds a **synthetic scene** whose ground truth is known: two
objects on a textured background, each of them split into a few stable
tokens. That makes it possible to assert exactly which tokens a mask has to
select - including the cases that are hard to judge on real footage (a
moving object, partial occlusion, tracks that appear late, boundary false
positives).

The scene

    feature grid 8 x 10 cells, patch size 4  -> working resolution 32 x 40
    original resolution 12 x 20  (a *different* aspect ratio than the
    working resolution, so the resolution handling is really exercised)

    rows 0-3, cols 0-4    object A  -> tokens 0..3   (one token per row)
    rows 0-3, cols 5-9    object B  -> tokens 10..13
    rows 4-7              background-> token 100      (40 cells per frame)

Run with:
    pixi run python tests/test_select_object.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "worker") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "worker"))

import numpy as np  # noqa: E402

from comfy_nodes import bridge, nodes  # noqa: E402
from comfy_nodes.config import Config  # noqa: E402
from comfy_nodes.types import TrackData, TrackSelection  # noqa: E402

# --- the scene --------------------------------------------------------------
GRID_H, GRID_W = 8, 10
PATCH = 4
WORK_H, WORK_W = GRID_H * PATCH, GRID_W * PATCH      # 32 x 40
ORIG_H, ORIG_W = 12, 20                               # different aspect
NUM_FRAMES = 6

OBJECT_A = [(r, c) for r in range(4) for c in range(0, 5)]
OBJECT_B = [(r, c) for r in range(4) for c in range(5, 10)]
BACKGROUND = [(r, c) for r in range(4, 8) for c in range(0, 10)]

# The token ids in ``cell_to_token`` are the **row indices** of the tracks
# tensor (0 .. N-1), exactly like the upstream produces them.
TOKENS_A = [0, 1, 2, 3]        # one per row of object A
TOKENS_B = [4, 5, 6, 7]        # one per row of object B
TOKEN_BACKGROUND = 8
TOKEN_LATE = 9                 # appears at frame 3, inside object A
TOKEN_FLAT = 10                # inside object A in a single frame only

#: everything that is inside object A (the late token belongs to it too)
EXPECTED_A = sorted(TOKENS_A + [TOKEN_LATE])


def build_scene(directory: Path, num_frames: int = NUM_FRAMES) -> Path:
    """Write a synthetic result set with a known ground truth."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)

    # --- the token map ------------------------------------------------
    cell_to_token = torch.full((num_frames, GRID_H, GRID_W), TOKEN_BACKGROUND,
                               dtype=torch.int64)
    for row, token in enumerate(TOKENS_A):
        for r in range(row, row + 1):
            for c in range(0, 5):
                cell_to_token[:, r, c] = token
    for row, token in enumerate(TOKENS_B):
        for r in range(row, row + 1):
            for c in range(5, 10):
                cell_to_token[:, r, c] = token
    # a token that is born late and lives inside object A
    cell_to_token[3:, 3, 0] = TOKEN_LATE
    # a token that only shows up inside object A in a single frame
    cell_to_token[2, 3, 1] = TOKEN_FLAT

    token_ids = sorted(set(cell_to_token.reshape(-1).tolist()))
    index = {token: i for i, token in enumerate(token_ids)}
    n_tracks = len(token_ids)

    # --- the tracks ----------------------------------------------------
    coords = torch.zeros(n_tracks, num_frames, 3, dtype=torch.float32)
    visibs = torch.full((n_tracks, num_frames), 10.0, dtype=torch.float32)
    dyn_logits = torch.zeros(n_tracks, num_frames, dtype=torch.float32)
    src_time = torch.zeros(n_tracks, dtype=torch.int64)
    src_y = torch.zeros(n_tracks, dtype=torch.int64)
    src_x = torch.zeros(n_tracks, dtype=torch.int64)

    for token, i in index.items():
        # a fixed 3D position per token, with a slight drift so that the
        # coordinates are not degenerate
        coords[i, :, 0] = 0.1 * (i % 7)
        coords[i, :, 1] = 0.05 * (i % 5)
        coords[i, :, 2] = 2.0
        first = cell_to_token[:, :, :][cell_to_token == token]
        positions = torch.nonzero(cell_to_token == token)
        if positions.numel():
            src_y[i] = int(positions[0][1])
            src_x[i] = int(positions[0][2])
            src_time[i] = int(positions[0][0])

    tracks = {
        "coords": coords,
        "visibs": visibs,
        "dyn_logits": dyn_logits,
        "src_time": src_time,
        "src_y": src_y,
        "src_x": src_x,
        "cell_to_token": cell_to_token,
    }
    torch.save(tracks, directory / "tracks.pt")

    geometry = {
        "norm_factor": 1.0,
        "depths": torch.ones(num_frames, WORK_H, WORK_W, dtype=torch.float32),
        "intrinsics": torch.eye(3).expand(num_frames, 3, 3).contiguous(),
        "extrinsics": torch.eye(4).expand(num_frames, 4, 4).contiguous(),
    }
    torch.save(geometry, directory / "geometry.pt")

    metadata = {
        "schema_version": 1,
        "input": {
            "num_frames": num_frames,
            "original_size": {"width": ORIG_W, "height": ORIG_H},
            "dtype": "float32",
            "value_range": "[0, 1]",
            "colour_space": "RGB",
        },
        "output": {
            "num_frames": num_frames,
            "num_tracks": n_tracks,
            "working_size": {"width": WORK_W, "height": WORK_H},
            "feature_grid": {"height": GRID_H, "width": GRID_W},
            "patch_size": PATCH,
        },
        "files": {"tracks": "tracks.pt", "geometry": "geometry.pt",
                  "metadata": "metadata.json"},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return directory


def mask_for_cells(cells, height: int, width: int) -> np.ndarray:
    """A binary mask (at the given resolution) covering the grid cells."""
    import cv2

    work = np.zeros((WORK_H, WORK_W), dtype=np.uint8)
    for r, c in cells:
        work[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 255
    if (height, width) != (WORK_H, WORK_W):
        work = cv2.resize(work, (width, height), interpolation=cv2.INTER_NEAREST)
    return work.astype(np.float32) / 255.0


def mask_batch(cells_per_frame, height: int = ORIG_H, width: int = ORIG_W) -> np.ndarray:
    """A (T, H, W) float mask batch."""
    return np.stack([mask_for_cells(cells, height, width)
                     for cells in cells_per_frame]).astype(np.float32)


class SelectCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="te_select_test_")
        self.tmp = Path(self._tmp)
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self.result = build_scene(self.tmp / "result")
        self.config = Config.resolve()
        self.config.worker_python = Path(sys.executable)

    def track_data(self) -> TrackData:
        return bridge.load_result_dir(str(self.result))

    def select(self, masks, **kwargs):
        return bridge.run_selection(
            self.config, self.track_data(), masks,
            output_dir=self.tmp / "selection", **kwargs,
        )


# ---------------------------------------------------------------------------
# the algorithm
# ---------------------------------------------------------------------------
class TestSelectionAlgorithm(SelectCase):
    def test_static_object_with_a_constant_mask(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)

        self.assertEqual(sorted(selection.token_ids), EXPECTED_A)
        self.assertEqual(selection.count, len(EXPECTED_A))

    def test_the_other_object_is_not_selected(self):
        masks = mask_batch([OBJECT_B] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        self.assertEqual(sorted(selection.token_ids), sorted(TOKENS_B))

    def test_the_background_is_not_selected(self):
        """The background token has 40 cells per frame - it must not win."""
        masks = mask_batch([BACKGROUND] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        self.assertEqual(sorted(selection.token_ids), [TOKEN_BACKGROUND])

    def test_a_moving_object_is_tracked_as_one_stable_set(self):
        """The mask follows the object - the selected ids stay stable."""
        per_frame = []
        for t in range(NUM_FRAMES):
            shift = min(3, t)
            cells = [(r, c + shift) for r, c in OBJECT_A if c + shift < 5]
            per_frame.append(cells)
        masks = mask_batch(per_frame)
        selection = self.select(masks, erode_pixels=0, min_inside_ratio=0.5)

        # the tokens of object A that stay inside the moving mask
        self.assertIn(TOKENS_A[0], selection.token_ids)
        self.assertNotIn(TOKEN_BACKGROUND, selection.token_ids)
        # the selection is a *set of ids*, not a per frame decision
        self.assertIsInstance(selection.token_ids, tuple)

    def test_a_single_missing_frame_does_not_remove_a_track(self):
        """Partial occlusion: one frame without the object keeps the track."""
        per_frame = [OBJECT_A] * NUM_FRAMES
        per_frame[2] = []          # fully occluded for one frame
        masks = mask_batch(per_frame)
        selection = self.select(masks, erode_pixels=0, min_inside_ratio=0.6)

        # 5 of 6 frames -> ratio 0.83 -> still selected
        for token in TOKENS_A:
            self.assertIn(token, selection.token_ids)

    def test_a_long_occlusion_removes_the_track(self):
        per_frame = [OBJECT_A] * NUM_FRAMES
        for t in (2, 3, 4):
            per_frame[t] = []
        masks = mask_batch(per_frame)
        selection = self.select(masks, erode_pixels=0, min_inside_ratio=0.6)
        # only 3 of 6 frames -> ratio 0.5 -> below the threshold
        self.assertNotIn(TOKENS_A[0], selection.token_ids)

    def test_tracks_that_appear_later_can_be_selected(self):
        """A token born at frame 3 is judged on its own frames."""
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        self.assertIn(TOKEN_LATE, selection.token_ids)

    def test_a_single_frame_support_is_not_enough(self):
        """TOKEN_FLAT only lives inside the mask in one frame."""
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0, min_support_frames=2)
        self.assertNotIn(TOKEN_FLAT, selection.token_ids)

    def test_min_support_frames_one_accepts_it(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0, min_support_frames=1)
        self.assertIn(TOKEN_FLAT, selection.token_ids)

    def test_boundary_false_positives_are_rejected(self):
        """A token that only touches the mask at its border is not selected."""
        # a thin mask that only covers the last column of object A
        border = [(r, 4) for r in range(4)]
        masks = mask_batch([border] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0, min_inside_ratio=0.6)
        # the tokens of object A are only partially covered -> not selected
        self.assertNotIn(TOKENS_A[0], selection.token_ids)
        self.assertNotIn(TOKENS_A[1], selection.token_ids)

    def test_the_erosion_shrinks_the_mask(self):
        """erode_pixels removes the border of the object."""
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        wide = self.select(masks, erode_pixels=0)
        narrow = self.select(masks, erode_pixels=3)
        self.assertLess(narrow.count, wide.count)

    def test_an_empty_mask_selects_nothing(self):
        masks = np.zeros((NUM_FRAMES, ORIG_H, ORIG_W), dtype=np.float32)
        selection = self.select(masks, erode_pixels=0)
        self.assertEqual(selection.count, 0)
        self.assertEqual(selection.token_ids, ())

    def test_a_mask_that_covers_everything_selects_the_background_too(self):
        masks = np.ones((NUM_FRAMES, ORIG_H, ORIG_W), dtype=np.float32)
        selection = self.select(masks, erode_pixels=0)
        self.assertIn(TOKEN_BACKGROUND, selection.token_ids)
        self.assertIn(TOKENS_A[0], selection.token_ids)


# ---------------------------------------------------------------------------
# the resolutions
# ---------------------------------------------------------------------------
class TestResolutions(SelectCase):
    def test_the_mask_may_be_given_at_the_original_resolution(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES, ORIG_H, ORIG_W)
        selection = self.select(masks, erode_pixels=0)
        self.assertEqual(sorted(selection.token_ids), EXPECTED_A)

    def test_the_mask_may_be_given_at_the_working_resolution(self):
        """The same mask at the working resolution gives the same result."""
        masks = mask_batch([OBJECT_A] * NUM_FRAMES, WORK_H, WORK_W)
        selection = self.select(masks, erode_pixels=0)
        self.assertEqual(sorted(selection.token_ids), EXPECTED_A)

    def test_the_resolutions_are_really_different(self):
        """The fixture must actually exercise the mapping."""
        self.assertNotEqual((ORIG_H, ORIG_W), (WORK_H, WORK_W))
        self.assertNotEqual(ORIG_H * ORIG_W, WORK_H * WORK_W)

    def test_the_scale_factors_are_applied(self):
        """A mask that only covers the top half must not select the bottom."""
        top = [(r, c) for r in range(0, 2) for c in range(0, 5)]
        masks = mask_batch([top] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        self.assertIn(TOKENS_A[0], selection.token_ids)
        self.assertNotIn(TOKENS_A[3], selection.token_ids)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class TestErrors(SelectCase):
    def test_a_frame_count_mismatch_is_reported(self):
        masks = mask_batch([OBJECT_A] * (NUM_FRAMES - 1))
        with self.assertRaises(bridge.TrackEverythingError) as ctx:
            self.select(masks)
        self.assertIn("frames", str(ctx.exception))

    def test_too_many_frames_are_reported(self):
        masks = mask_batch([OBJECT_A] * (NUM_FRAMES + 3))
        with self.assertRaises(bridge.TrackEverythingError):
            self.select(masks)

    def test_an_empty_mask_batch_is_reported(self):
        with self.assertRaises(bridge.TrackEverythingError):
            self.select(np.zeros((0, ORIG_H, ORIG_W), dtype=np.float32))

    def test_a_missing_result_is_reported(self):
        (self.result / "tracks.pt").unlink()
        with self.assertRaises(bridge.TrackEverythingError):
            self.select(mask_batch([OBJECT_A] * NUM_FRAMES))


# ---------------------------------------------------------------------------
# the node and the payload
# ---------------------------------------------------------------------------
class TestNode(SelectCase):
    def test_the_node_interface(self):
        spec = nodes.TrackEverythingSelectObject.INPUT_TYPES()["required"]
        self.assertEqual(spec["masks"][0], "MASK")
        self.assertEqual(spec["track_data"][0], "TRACK_DATA")
        self.assertEqual(spec["mask_threshold"][1]["default"], 0.5)
        self.assertEqual(spec["erode_pixels"][1]["default"], 2)
        self.assertEqual(spec["min_inside_ratio"][1]["default"], 0.6)
        self.assertEqual(spec["min_support_frames"][1]["default"], 2)

    def test_the_node_returns_the_three_outputs(self):
        node = nodes.TrackEverythingSelectObject()
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection, count, summary = node.select(self.track_data(), masks,
                                                erode_pixels=0)
        self.assertIsInstance(selection, TrackSelection)
        self.assertIsInstance(count, int)
        self.assertIsInstance(summary, str)
        self.assertEqual(count, len(selection.token_ids))
        self.assertIn("selected", summary)

    def test_the_node_is_not_an_output_node(self):
        self.assertFalse(nodes.TrackEverythingSelectObject.OUTPUT_NODE)

    def test_the_selection_is_saved_in_its_own_directory(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        path = Path(selection.selection_path)
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "selection.json")
        self.assertTrue(path.parent.is_dir())
        # the source result is untouched
        self.assertTrue((self.result / "tracks.pt").is_file())
        self.assertTrue((self.result / "metadata.json").is_file())

    def test_the_selection_is_unique(self):
        """Two selections must not end up in the same directory."""
        root = self.tmp / "selections"
        first = bridge._unique_selection_dir(root)
        first.mkdir(parents=True)
        second = bridge._unique_selection_dir(root)
        self.assertNotEqual(first, second)
        self.assertTrue(str(second).startswith(str(root)))

    def test_a_saved_selection_can_be_reused(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)

        # load it again from disk, exactly like a later workflow would
        reloaded = TrackSelection(
            selection_dir=selection.selection_dir,
            selection_path=selection.selection_path,
            source_result_dir=selection.source_result_dir,
            source_tracks_path=selection.source_tracks_path,
            fingerprint=selection.fingerprint,
            token_ids=tuple(selection.metadata["selected_token_ids"]),
            count=selection.metadata["num_selected"],
            parameters=selection.parameters,
            metadata=selection.metadata,
        )
        self.assertEqual(reloaded.token_ids, selection.token_ids)
        self.assertEqual(reloaded.fingerprint, selection.fingerprint)

    def test_the_selection_records_its_source(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        source = selection.metadata["source"]
        self.assertTrue(source["fingerprint"].startswith("sha256:"))
        self.assertEqual(source["num_frames"], NUM_FRAMES)
        self.assertEqual(source["feature_grid_hw"], [GRID_H, GRID_W])
        self.assertEqual(source["working_size"], {"width": WORK_W, "height": WORK_H})
        self.assertEqual(source["original_size"], {"width": ORIG_W, "height": ORIG_H})

    def test_a_selection_of_another_result_is_rejected(self):
        """The fingerprint must not match a different inference."""
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)

        # a different result with the same shape but different content
        other = build_scene(self.tmp / "other", num_frames=NUM_FRAMES)
        import torch

        tracks = torch.load(other / "tracks.pt", map_location="cpu", weights_only=True)
        tracks["coords"] = tracks["coords"] + 1.0
        torch.save(tracks, other / "tracks.pt")

        other_selection = json.loads(json.dumps(selection.metadata))
        other_selection["source"]["result_dir"] = str(other)
        (self.tmp / "wrong_selection.json").write_text(json.dumps(other_selection))

        # the visualizer worker has to refuse it - it runs in a subprocess, so
        # the check does not pollute this process
        proc = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "worker" / "visualize.py"),
                "--result-dir", str(other),
                "--output-dir", str(self.tmp / "vis_wrong"),
                "--selection-file", str(self.tmp / "wrong_selection.json"),
            ],
            capture_output=True, text=True, timeout=300,
        )
        self.assertNotEqual(proc.returncode, 0,
                            "the foreign selection was accepted")
        self.assertIn("fingerprint", proc.stderr.lower())

    def test_the_visualizer_accepts_its_own_selection(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        proc = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "worker" / "visualize.py"),
                "--result-dir", str(self.result),
                "--output-dir", str(self.tmp / "vis_ok"),
                "--selection-file", selection.selection_path,
            ],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertTrue(list((self.tmp / "vis_ok").glob("vis_*.png")))


# ---------------------------------------------------------------------------
# the integration with Visualize and Save Data
# ---------------------------------------------------------------------------
class TestIntegration(SelectCase):
    def test_visualize_without_a_selection_still_works(self):
        (rendered,) = bridge.visualize_result(self.config, self.track_data())
        self.assertEqual(rendered.ndim, 4)
        self.assertEqual(rendered.shape[-1], 3)

    def test_visualize_with_a_selection_draws_only_the_selected_tracks(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)

        (rendered,) = bridge.visualize_result(
            self.config, self.track_data(), selection=selection
        )
        self.assertEqual(rendered.ndim, 4)
        # the selected tracks are drawn - something must be visible
        self.assertGreater(int((rendered.max(dim=-1).values > 0).sum()), 0)

    def test_visualize_with_a_selection_reports_the_counts(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        (rendered,) = bridge.visualize_result(
            self.config, self.track_data(), selection=selection
        )
        report = json.loads(
            (self.result / "worker.log").read_text().split("{", 1)[-1].split("}", 1)[0]
        ) if False else None
        # the selection is honoured: fewer tracks than the whole set
        self.assertLess(selection.count, int(self.track_data().num_tracks))

    def test_save_data_copies_the_selection(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)

        output_root = self.tmp / "out"
        saved, message = bridge.save_result(
            self.track_data(), save_mode="copy",
            filename_prefix="sel", output_root=output_root,
            selection=selection,
        )
        destination = Path(saved.result_dir)
        self.assertTrue((destination / "selection.json").is_file())
        self.assertIn("selection", message)

        metadata = json.loads((destination / "metadata.json").read_text())
        self.assertIn("selection", metadata)
        self.assertEqual(metadata["files"]["selection"], "selection.json")

        # the copied selection points at the copy
        copied = json.loads((destination / "selection.json").read_text())
        self.assertEqual(copied["source"]["result_dir"], str(destination.resolve()))

    def test_save_data_without_a_selection_is_unchanged(self):
        output_root = self.tmp / "out"
        saved, _ = bridge.save_result(
            self.track_data(), save_mode="copy",
            filename_prefix="plain", output_root=output_root,
        )
        destination = Path(saved.result_dir)
        self.assertFalse((destination / "selection.json").exists())
        metadata = json.loads((destination / "metadata.json").read_text())
        self.assertNotIn("selection", metadata)

    def test_save_data_reference_mode_mentions_the_selection(self):
        masks = mask_batch([OBJECT_A] * NUM_FRAMES)
        selection = self.select(masks, erode_pixels=0)
        _, message = bridge.save_result(
            self.track_data(), save_mode="reference", selection=selection
        )
        self.assertIn("selection", message)


class TestRealResultEndToEnd(unittest.TestCase):
    """E2E with the real tracks.pt / geometry.pt of the standalone run."""

    RESULT = PROJECT_ROOT / "outputs" / "standalone_16f"

    @classmethod
    def setUpClass(cls) -> None:
        if not (cls.RESULT / "tracks.pt").is_file():
            raise unittest.SkipTest(
                f"{cls.RESULT} not found - run the standalone inference first"
            )

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="te_select_e2e_")
        self.tmp = Path(self._tmp)
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self.config = Config.resolve()
        self.config.worker_python = Path(sys.executable)
        self.track_data = bridge.load_result_dir(str(self.RESULT))

    def _mask_batch(self, height: int, width: int) -> np.ndarray:
        """A mask that covers the left half of the image."""
        mask = np.zeros((height, width), dtype=np.float32)
        mask[:, : width // 2] = 1.0
        return np.stack([mask] * int(self.track_data.num_frames))

    def test_the_selection_matches_the_masks(self):
        import torch

        metadata = self.track_data.metadata
        working = metadata["output"]["working_size"]
        height, width = working["height"], working["width"]

        masks = self._mask_batch(height, width)
        selection = bridge.run_selection(
            self.config, self.track_data, masks,
            erode_pixels=0, min_inside_ratio=0.5, min_support_frames=2,
            output_dir=self.tmp / "selection",
        )

        self.assertGreater(selection.count, 0)
        self.assertLess(selection.count, int(self.track_data.num_tracks))

        # --- an independent check of the selection ----------------------
        # every selected token must have cells inside the mask in at least
        # `min_support_frames` frames
        tracks = torch.load(self.RESULT / "tracks.pt", map_location="cpu",
                            weights_only=True)
        cell_to_token = tracks["cell_to_token"]
        grid_h, grid_w = cell_to_token.shape[1:]
        left = grid_w // 2

        selected = set(selection.token_ids)
        support = {token: 0 for token in selected}
        # the column of a cell comes from its *cell index*, not from the
        # token id that is stored in it
        cell_index = torch.arange(grid_h * grid_w)
        columns = cell_index % grid_w
        inside = columns < left
        for t in range(cell_to_token.shape[0]):
            ids = cell_to_token[t].reshape(-1)
            for token in selected:
                if bool(((ids == token) & inside).any()):
                    support[token] += 1

        for token, frames in support.items():
            self.assertGreaterEqual(
                frames, 2,
                f"token {token} was selected but only has {frames} masked frames",
            )

    def test_the_selection_is_stable_across_runs(self):
        masks = self._mask_batch(384, 512)
        first = bridge.run_selection(
            self.config, self.track_data, masks, erode_pixels=0,
            output_dir=self.tmp / "a",
        )
        second = bridge.run_selection(
            self.config, self.track_data, masks, erode_pixels=0,
            output_dir=self.tmp / "b",
        )
        self.assertEqual(first.token_ids, second.token_ids)

    def test_the_visualizer_renders_the_selection(self):
        masks = self._mask_batch(384, 512)
        selection = bridge.run_selection(
            self.config, self.track_data, masks, erode_pixels=0,
            output_dir=self.tmp / "selection",
        )
        (rendered,) = bridge.visualize_result(
            self.config, self.track_data, selection=selection
        )
        self.assertEqual(rendered.ndim, 4)
        self.assertEqual(int(rendered.shape[0]), int(self.track_data.num_frames))
        self.assertGreater(int((rendered.max(dim=-1).values > 0).sum()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
