#!/usr/bin/env python
"""Patch croco's ``kernels.cu`` for the TrackEverything / Pi3 memory layout.

Why a patch is needed
---------------------
croco's CUDA kernel starts with::

    TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D,
                "tokens are not contiguous");

The kernel receives the tokens as ``(B, N, H, D)`` and requires the **head**
stride to equal ``D`` - i.e. it expects the memory layout of croco's own
attention, where a head's ``D`` values are adjacent.

Pi3 keeps the attention tensor as a contiguous ``(B, H, N, D)``, so after the
``transpose(1, 2)`` the head stride is ``N * D`` and the assertion fires
("tokens are not contiguous").

The kernel body itself is layout agnostic: it reads and writes through
``torch::PackedTensorAccessor32``, which resolves the element offsets from the
strides. Only the assertion is too strict for Pi3's layout, so the patch
relaxes exactly that one check and keeps the algorithm untouched.

The patch is applied with an exact-match replacement; if the expected source
text is not found the build fails loudly instead of silently producing a
broken extension.

Run with:  python scripts/patch_curope.py <path to croco.cpp>
"""

from __future__ import annotations

import sys
from pathlib import Path

ORIGINAL = """    TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D, "tokens are not contiguous");
"""

PATCHED = """    // PATCHED for the TrackEverything / Pi3 integration.
    //
    // The original check was:
    //     TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D,
    //                 "tokens are not contiguous");
    // It requires the *head* stride to equal D, which only holds for croco's
    // own attention layout. Pi3 keeps the attention tensor contiguous in
    // (B, H, N, D), so after transpose(1, 2) the head stride is N * D and the
    // assertion fired ("tokens are not contiguous").
    //
    // The kernel body is layout agnostic - it goes through
    // torch::PackedTensorAccessor32, which resolves the offsets from the
    // strides - so only the last dimension really has to be contiguous.
    TORCH_CHECK(tokens.stride(3) == 1, "the last dimension must be contiguous");
"""


def patch(path: Path) -> bool:
    text = path.read_text()
    if PATCHED in text:
        print(f"    {path} is already patched")
        return False
    if ORIGINAL not in text:
        print(
            f"ERROR: the expected line was not found in {path}.\n"
            "       The croco source changed - refusing to patch blindly.",
            file=sys.stderr,
        )
        sys.exit(1)
    path.write_text(text.replace(ORIGINAL, PATCHED, 1))
    print(f"    patched {path}")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    patch(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
