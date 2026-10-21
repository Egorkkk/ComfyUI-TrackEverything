#!/usr/bin/env python
"""Stage D1 - generate a short synthetic test clip.

Creates a small textured RGB video so the standalone inference can be smoke
tested without any user media. This is a *functional* test of the pipeline
(not a quality benchmark of the tracker).

Content:
  * a static, finely textured background (checkerboard + coloured noise) so
    that the geometry backend has trackable features everywhere,
  * three textured objects translating smoothly across the frame.

Written with PyAV (H.264) because the upstream `trackeverything.demo_hf`
decodes videos with PyAV as well - guaranteed round-trip compatibility.

Run with:  pixi run python scripts/make_test_video.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_background(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Static high-frequency textured background (H, W, 3) uint8."""
    rng = np.random.default_rng(seed)

    # Fine checkerboard, 8 px cells -> strong, unambiguous corners.
    yy, xx = np.mgrid[0:height, 0:width]
    checker = (((yy // 8) + (xx // 8)) % 2).astype(np.float32)
    bg = np.stack([checker, checker, checker], axis=-1) * 70.0 + 60.0

    # Low-frequency colour gradient so the background is not flat grey.
    grad = np.linspace(0.0, 40.0, width, dtype=np.float32)[None, :, None]
    bg = bg + grad

    # Fine per-pixel noise: gives the tracker dense, unique texture.
    noise = rng.normal(0.0, 12.0, size=(height, width, 3)).astype(np.float32)
    bg = bg + noise

    return np.clip(bg, 0, 255).astype(np.uint8)


def draw_disc(img: np.ndarray, cy: float, cx: float, radius: float, rgb, seed: int) -> None:
    """Draw a textured disc (in place) at pixel centre (cy, cx)."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    if not mask.any():
        return
    tex = rng.normal(0.0, 25.0, size=(y1 - y0, x1 - x0, 3)).astype(np.float32)
    patch = np.array(rgb, dtype=np.float32)[None, None, :] + tex
    img[y0:y1, x0:x1][mask] = np.clip(patch[mask], 0, 255).astype(np.uint8)


def build_frames(n_frames: int, height: int, width: int) -> np.ndarray:
    """Return (T, H, W, 3) uint8 RGB frames."""
    bg = make_background(height, width)
    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)

    for t in range(n_frames):
        f = bg.copy()
        p = t / max(1, n_frames - 1)  # 0 -> 1

        # Three objects moving linearly along different directions.
        draw_disc(f, cy=60 + 18 * p, cx=70 + 150 * p, radius=26,
                  rgb=(220, 70, 60), seed=100 + t)
        draw_disc(f, cy=190 - 22 * p, cx=300 - 170 * p, radius=22,
                  rgb=(60, 200, 90), seed=200 + t)
        draw_disc(f, cy=140 + 26 * p, cx=190 - 40 * p, radius=18,
                  rgb=(70, 110, 235), seed=300 + t)

        frames[t] = f

    return frames


def write_mp4(frames: np.ndarray, path: Path, fps: int = 8) -> None:
    """Encode (T, H, W, 3) uint8 RGB to H.264 mp4 via PyAV."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="test_data/synthetic_16f.mp4")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    frames = build_frames(args.frames, args.height, args.width)
    write_mp4(frames, out, fps=args.fps)

    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({args.frames} frames, {args.width}x{args.height}, "
          f"{args.fps} fps, {size_kb:.1f} KiB)")

    # Round-trip verification with the same decoder the upstream code uses.
    import av

    container = av.open(str(out))
    decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    arr = np.stack(decoded)
    print(f"Decoded back: {arr.shape} {arr.dtype}")
    assert arr.shape == frames.shape, (arr.shape, frames.shape)
    mae = np.abs(arr.astype(np.float32) - frames.astype(np.float32)).mean()
    print(f"Round-trip mean absolute error: {mae:.3f} (lossy H.264 expected)")


if __name__ == "__main__":
    main()
