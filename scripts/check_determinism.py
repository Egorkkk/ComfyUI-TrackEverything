#!/usr/bin/env python
"""Reproducibility check: compare the tensors of two inference runs.

The upstream project is *structurally* deterministic for a fixed input clip
and a fixed backend: the same number of tracks, the same token assignment and
bit-identical geometry. The tracker itself is **not** bit-reproducible on GPU,
because several upstream kernels reduce with atomics / non-deterministic
reduction orders, so repeated runs differ by a few float32 ULPs.

This script therefore reports, for every tensor:

  * whether the values are bit-identical,
  * the maximum absolute difference,
  * the maximum relative difference (against the tensor's own scale),

and classifies the run as reproducible when the relative difference stays
below `--rtol` (default 1e-4, which is the float32 noise floor for these
quantities).

Usage:
    pixi run python scripts/check_determinism.py outputs/run_a outputs/run_b
    pixi run python scripts/check_determinism.py --exact a b   # require bit-identity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

FILES = ("tracks.pt", "geometry.pt")


def load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def compare(a: dict, b: dict, label: str, rtol: float, exact: bool) -> list[str]:
    problems: list[str] = []
    if set(a.keys()) != set(b.keys()):
        problems.append(f"{label}: key sets differ: {sorted(a.keys())} vs {sorted(b.keys())}")
        return problems
    for k in sorted(a.keys()):
        ta, tb = a[k], b[k]
        if isinstance(ta, torch.Tensor) != isinstance(tb, torch.Tensor):
            problems.append(f"{label}.{k}: type mismatch")
            continue
        if not isinstance(ta, torch.Tensor):
            if ta != tb:
                problems.append(f"{label}.{k}: value mismatch ({ta!r} vs {tb!r})")
            continue
        if ta.shape != tb.shape:
            problems.append(f"{label}.{k}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        if ta.dtype != tb.dtype:
            problems.append(f"{label}.{k}: dtype {ta.dtype} vs {tb.dtype}")
            continue

        if not ta.is_floating_point():
            same = torch.equal(ta, tb)
            max_abs = 0.0 if same else float((ta.double() - tb.double()).abs().max())
            max_rel = 0.0 if same else 1.0
        else:
            da = ta.double()
            db = tb.double()
            diff = (da - db).abs()
            diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = float(diff.max()) if diff.numel() else 0.0
            scale = float(da.abs().max()) if da.numel() else 0.0
            max_rel = (max_abs / scale) if scale > 0 else 0.0
            # NaN-aware equality
            same = bool(((ta == tb) | (torch.isnan(ta) & torch.isnan(tb))).all())

        if same:
            verdict = "PASS"
            detail = "bit-identical"
        elif max_rel <= rtol:
            verdict = "PASS"
            detail = f"differs within float32 noise (rel={max_rel:.3e} <= {rtol:.0e})"
        else:
            verdict = "FAIL"
            detail = f"differs by rel={max_rel:.3e} > {rtol:.0e}"
            problems.append(f"{label}.{k}: {detail} (max|diff|={max_abs:.3e})")

        print(f"[{verdict}] {label}.{k:16s} shape={tuple(ta.shape)} "
              f"dtype={ta.dtype} max|diff|={max_abs:.3e} rel={max_rel:.3e}  {detail}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--rtol", type=float, default=1e-4,
                    help="max allowed relative difference for float tensors (default 1e-4)")
    ap.add_argument("--exact", action="store_true",
                    help="require bit-identical tensors (no tolerance)")
    args = ap.parse_args()

    dir_a, dir_b = Path(args.run_a), Path(args.run_b)
    rtol = 0.0 if args.exact else args.rtol

    problems: list[str] = []
    for fname in FILES:
        pa, pb = dir_a / fname, dir_b / fname
        if not pa.exists() or not pb.exists():
            problems.append(f"{fname} missing in one of the runs")
            print(f"[FAIL] {fname} missing")
            continue
        print(f"--- {fname} ---")
        problems += compare(load(pa), load(pb), fname, rtol, args.exact)

    print()
    if problems:
        print(f"RESULT: {len(problems)} DIFFERENCE(S) FOUND")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("RESULT: RUNS MATCH WITHIN TOLERANCE"
          if not args.exact else "RESULT: RUNS ARE BIT-IDENTICAL")


if __name__ == "__main__":
    main()
