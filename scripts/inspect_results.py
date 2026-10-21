#!/usr/bin/env python
"""Stage E - inspect and validate the files produced by `trackeverything.demo_hf`.

Reads `tracks.pt` and `geometry.pt` from an inference output directory and
reports dimensions, dtypes and basic statistics, then runs a set of
consistency checks.

Important semantics (from upstream `utils/alltrack3r_utils.SceneTokens`):
  * `visibs` holds *raw visibility logits*, and -1000 marks cells that were
    never written (missing / not-yet-existing trajectories).
  * The same -1000 sentinel can appear in `coords` for trajectories that do
    not exist at a given target frame.
  * -1000 is therefore NOT a valid 3D coordinate and is excluded from the
    numeric statistics below.

The script never converts the results to JSON or any other lossy format: the
original tensors are read and reported in their native dtype.

Run with:  pixi run python scripts/inspect_results.py outputs/standalone
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

SENTINEL = -1000.0


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_pt(path: Path) -> dict:
    """Load a trusted, locally produced .pt file on CPU."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        print(f"  loaded with weights_only=True")
    except Exception as exc:
        print(f"  weights_only=True failed ({type(exc).__name__}); "
              f"retrying with weights_only=False (file is locally produced)")
        obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def describe(name: str, t: torch.Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        print(f"  {name:16s} : type={type(t).__name__} value={t!r}")
        return
    finite = torch.isfinite(t)
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    print(f"  {name:16s} : shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={_fmt(t.min())} max={_fmt(t.max())} "
          f"nan={n_nan} inf={n_inf}")


def _fmt(v: torch.Tensor) -> str:
    return f"{v.item():.6g}"


def stats_block(title: str, t: torch.Tensor, sentinel_mask: bool = True) -> None:
    """Print statistics for a tensor, excluding the -1000 sentinel if asked."""
    print(f"  [{title}]")
    if t.numel() == 0:
        print("    (empty)")
        return

    f = t.float()
    mask = torch.isfinite(f)
    if sentinel_mask:
        mask = mask & (f != SENTINEL)

    n_valid = int(mask.sum().item())
    n_total = f.numel()
    n_sentinel = int((f == SENTINEL).sum().item())

    print(f"    elements        : {n_total}  valid={n_valid}  sentinel(-1000)={n_sentinel}")
    if n_valid == 0:
        print("    no valid elements")
        return

    v = f[mask]
    print(f"    min             : {v.min().item():.6g}")
    print(f"    max             : {v.max().item():.6g}")
    print(f"    mean            : {v.mean().item():.6g}")
    print(f"    std             : {v.std().item():.6g}")
    print(f"    median          : {v.median().item():.6g}")


# ---------------------------------------------------------------------------
# main inspection
# ---------------------------------------------------------------------------
def inspect(tracks_path: Path, geometry_path: Path) -> list[str]:
    problems: list[str] = []
    print("=" * 78)
    print(f"tracks.pt   : {tracks_path}")
    print("=" * 78)
    tracks = load_pt(tracks_path)
    print(f"  type={type(tracks).__name__} keys={sorted(tracks.keys())}")

    for key in ("coords", "visibs", "dyn_logits", "src_time", "src_y", "src_x", "cell_to_token"):
        if key not in tracks:
            problems.append(f"tracks.pt is missing key '{key}'")
            continue
        describe(key, tracks[key])

    coords = tracks.get("coords")
    visibs = tracks.get("visibs")
    dyn = tracks.get("dyn_logits")
    src_time = tracks.get("src_time")
    src_y = tracks.get("src_y")
    src_x = tracks.get("src_x")
    c2t = tracks.get("cell_to_token")

    print()
    print("--- statistics -------------------------------------------------")
    if coords is not None:
        stats_block("coords (N,T,3)", coords)
    if visibs is not None:
        stats_block("visibs (N,T)", visibs)
    if dyn is not None:
        stats_block("dyn_logits (N,T)", dyn)
    if src_time is not None:
        stats_block("src_time (N,)", src_time, sentinel_mask=False)
    if src_y is not None:
        stats_block("src_y (N,)", src_y, sentinel_mask=False)
    if src_x is not None:
        stats_block("src_x (N,)", src_x, sentinel_mask=False)
    if c2t is not None:
        stats_block("cell_to_token (T,H',W')", c2t, sentinel_mask=False)

    print("=" * 78)
    print(f"geometry.pt : {geometry_path}")
    print("=" * 78)
    geometry = load_pt(geometry_path)
    print(f"  type={type(geometry).__name__} keys={sorted(geometry.keys())}")

    for key in ("depths", "intrinsics", "extrinsics", "norm_factor"):
        if key not in geometry:
            problems.append(f"geometry.pt is missing key '{key}'")
            continue
        describe(key, geometry[key])

    depths = geometry.get("depths")
    intr = geometry.get("intrinsics")
    extr = geometry.get("extrinsics")
    norm = geometry.get("norm_factor")

    print()
    print("--- statistics -------------------------------------------------")
    if depths is not None:
        stats_block("depths (T,H,W)", depths)
    if intr is not None:
        stats_block("intrinsics (T,3,3)", intr)
    if extr is not None:
        stats_block("extrinsics (T,4,4)", extr)
    if norm is not None:
        stats_block("norm_factor", norm.reshape(-1) if norm.dim() else norm, sentinel_mask=False)

    # -------------------------------------------------------------------
    # consistency checks
    # -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("CONSISTENCY CHECKS")
    print("=" * 78)

    def check(label: str, passed: bool, detail: str = "") -> None:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f"  ->  {detail}" if detail else ""))
        if not passed:
            problems.append(f"{label}: {detail}")

    # 1. frame count agreement
    n_tracks = None
    n_frames = None
    if coords is not None and coords.dim() == 3:
        n_tracks, n_frames = coords.shape[0], coords.shape[1]
        print(f"    reference: T={n_frames} frames, N={n_tracks} tracks")

    for name, t, expect_T in (
        ("visibs", visibs, n_frames),
        ("dyn_logits", dyn, n_frames),
        ("cell_to_token", c2t, n_frames),
        ("depths", depths, n_frames),
        ("intrinsics", intr, n_frames),
        ("extrinsics", extr, n_frames),
    ):
        if t is None or expect_T is None:
            continue
        check(
            f"frame count agrees: {name}",
            t.shape[0] == expect_T,
            f"{name}.shape[0]={t.shape[0]} vs T={expect_T}",
        )

    if coords is not None and visibs is not None:
        check("tracks count agrees: visibs", visibs.shape[0] == n_tracks,
              f"{visibs.shape[0]} vs {n_tracks}")
    if coords is not None and dyn is not None:
        check("tracks count agrees: dyn_logits", dyn.shape[0] == n_tracks,
              f"{dyn.shape[0]} vs {n_tracks}")
    if coords is not None and src_time is not None:
        check("tracks count agrees: src_time", src_time.shape[0] == n_tracks,
              f"{src_time.shape[0]} vs {n_tracks}")

    # 2. cell_to_token bounds
    if c2t is not None and n_tracks is not None:
        cmin = int(c2t.min().item())
        cmax = int(c2t.max().item())
        print(f"    cell_to_token range: [{cmin}, {cmax}] (valid tokens 0..{n_tracks - 1})")
        check("cell_to_token upper bound", cmax < n_tracks, f"max={cmax}, N={n_tracks}")
        check("cell_to_token lower bound", cmin >= -1, f"min={cmin} (only -1 allowed as 'no token')")
        if cmin >= 0:
            n_cells = c2t.numel()
            n_used = int(torch.unique(c2t).numel())
            print(f"    distinct tokens used: {n_used} / {n_tracks} "
                  f"({100.0 * n_used / max(1, n_tracks):.1f}% of tracks)")
            print(f"    grid cells covered  : {n_cells}")
            check("cell_to_token fully populated (no holes)",
                  int((c2t < 0).sum().item()) == 0,
                  f"{(c2t < 0).sum().item()} negative entries")

    # 3. NaN / Inf
    for name, t in (
        ("coords", coords), ("visibs", visibs), ("dyn_logits", dyn),
        ("src_time", src_time), ("src_y", src_y), ("src_x", src_x),
        ("cell_to_token", c2t),
        ("depths", depths), ("intrinsics", intr), ("extrinsics", extr),
    ):
        if t is None or not t.is_floating_point():
            continue
        n_nan = int(torch.isnan(t).sum().item())
        n_inf = int(torch.isinf(t).sum().item())
        check(f"no NaN/Inf in {name}", n_nan == 0 and n_inf == 0,
              f"nan={n_nan} inf={n_inf}")

    # 4. meaningful numeric data
    if coords is not None:
        valid = torch.isfinite(coords.float()) & (coords.float() != SENTINEL)
        frac = float(valid.float().mean().item())
        print(f"    coords: {frac * 100:.2f}% of entries are real 3D coordinates")
        check("coords contain real 3D values", frac > 0.0, f"valid fraction={frac:.4f}")
        if valid.any():
            std = coords.float()[valid].std().item()
            print(f"    coords std over valid entries: {std:.6g}")
            check("coords are not constant", std > 1e-6, f"std={std:.6g}")

    if depths is not None:
        d = depths.float()
        d = d[torch.isfinite(d)]
        if d.numel():
            print(f"    depths: min={d.min().item():.6g} max={d.max().item():.6g} "
                  f"mean={d.mean().item():.6g}")
            check("depths are positive", bool((d > 0).all().item()),
                  f"min={d.min().item():.6g}")

    if intr is not None:
        f = intr.float()
        diag = torch.stack([f[:, 0, 0], f[:, 1, 1], f[:, 2, 2]], dim=-1)
        print(f"    intrinsics focal lengths (fx,fy,cx): "
              f"fx mean={diag[:, 0].mean().item():.4f} "
              f"fy mean={diag[:, 1].mean().item():.4f} "
              f"cx mean={diag[:, 2].mean().item():.4f}")
        check("intrinsics have positive focal length", bool((diag[:, :2] > 0).all().item()))

    if extr is not None:
        e = extr.float()
        bottom = e[:, 3, :]
        check("extrinsics last row is [0,0,0,1]",
              bool(torch.allclose(bottom, torch.tensor([0.0, 0.0, 0.0, 1.0]).expand_as(bottom), atol=1e-4)),
              f"max dev={float((bottom - torch.tensor([0.0,0.0,0.0,1.0])).abs().max()):.3e}")

    return problems


def check_reloadable(output_dir: Path) -> list[str]:
    """Stage E5 - the files must be loadable from a *separate* process."""
    print()
    print("=" * 78)
    print("RELOAD IN A SEPARATE PROCESS")
    print("=" * 78)
    problems: list[str] = []
    code = (
        "import torch,sys;"
        f"d=torch.load({str(output_dir / 'tracks.pt')!r}, map_location='cpu', weights_only=True);"
        "print('  tracks.pt reloaded in pid', __import__('os').getpid(),"
        " '-> keys', sorted(d.keys()));"
        f"g=torch.load({str(output_dir / 'geometry.pt')!r}, map_location='cpu', weights_only=True);"
        "print('  geometry.pt reloaded -> keys', sorted(g.keys()))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr.strip())
        problems.append("files are not reloadable in a separate process")
        print(f"[FAIL] reload in separate process -> exit code {res.returncode}")
    else:
        print("[PASS] reload in separate process")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", nargs="?", default="outputs/standalone")
    args = ap.parse_args()

    out = Path(args.output_dir)
    tracks_path = out / "tracks.pt"
    geometry_path = out / "geometry.pt"

    if not tracks_path.exists() or not geometry_path.exists():
        print(f"ERROR: expected {tracks_path} and {geometry_path} to exist.")
        print("       Run the inference first, e.g. `pixi run infer`.")
        sys.exit(2)

    problems = inspect(tracks_path, geometry_path)
    problems += check_reloadable(out)

    print()
    print("=" * 78)
    if problems:
        print(f"RESULT: {len(problems)} PROBLEM(S) FOUND")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
