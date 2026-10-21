"""Build script for the ``curope`` CUDA extension (croco's cuRoPE2D kernels).

We do not use croco's own ``setup.py`` because it derives the architecture
flags from ``torch.cuda.get_gencode_flags()`` - the list of architectures
*torch itself* was compiled for. That list does not contain ``sm_89``
(Ada / RTX 4060 Ti), so the extension would not run on that GPU.

This setup file compiles the **unmodified** croco sources with an explicit
architecture list:

    sm_86  - NVIDIA A40 (Ampere)
    sm_89  - RTX 4060 Ti (Ada Lovelace)
    +PTX   - compute_89, so newer GPUs can JIT the kernel

The sources are taken from the croco checkout that ``scripts/build_curope.sh``
clones; nothing inside the croco repository is edited by this file.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#: where the croco checkout lives (set by the build script)
CROCO_ROOT = Path(os.environ.get("CROCO_ROOT", "/tmp/opencode/curope_build/croco"))
Curope_DIR = CROCO_ROOT / "models" / "curope"

#: The GPUs this extension has to run on.
#:   8.6 - NVIDIA A40
#:   8.9 - RTX 4060 Ti
ARCH_FLAGS = [
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_89,code=compute_89",
]

setup(
    name="curope",
    version="0.0.0",
    description="cuRoPE2D CUDA kernels (croco), built for sm_86 / sm_89",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                str(Curope_DIR / "curope.cpp"),
                str(Curope_DIR / "kernels.cu"),
            ],
            extra_compile_args=dict(
                # NOTE: --use_fast_math is deliberately NOT used. It maps
                # sinf/cosf onto the fast intrinsic approximations, which
                # shifts the rope values by ~1e-5 and - amplified through the
                # Pi3 transformer and the tracker's voxelisation - changes the
                # resulting geometry and track set. The kernel is memory
                # bound anyway, so the precise math costs nothing.
                nvcc=["-O3"] + ARCH_FLAGS,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
