#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build and install the CUDA implementation of Pi3's RoPE2D.
#
# The upstream TrackEverything imports it like this
# (external/trackeverything/third_party/pi3/models/layers/pos_embed.py):
#
#     try:
#         from models.curope import cuRoPE2D
#         RoPE2D = cuRoPE2D
#     except ImportError:
#         print('Warning, cannot find cuda-compiled version of RoPE2D, ...')
#         class RoPE2D(torch.nn.Module):   # slow pure-pytorch fallback
#
# so the upstream needs a top-level package ``models`` that contains the
# ``curope`` sub-package of the original croco repository
# (https://github.com/naver/croco, models/curope).
#
# We install exactly that layout into the isolated environment:
#
#   <site-packages>/models/curope/__init__.py     (copied from croco)
#   <site-packages>/models/curope/curope2d.py    (copied from croco)
#   <site-packages>/curope...so                   (the compiled extension)
#
# ``models`` has no ``__init__.py`` in croco either - it is a namespace
# package - so nothing else in the environment is shadowed. The upstream
# TrackEverything source is NOT modified.
#
# Two deviations from the stock croco sources, both documented:
#
#   1. ``scripts/curope_build/setup.py`` replaces croco's ``setup.py`` so the
#      architecture list is explicit (sm_86 for the A40, sm_89 for the
#      RTX 4060 Ti). croco's own setup.py uses
#      ``torch.cuda.get_gencode_flags()``, which lists the architectures
#      *torch* was built for and does not contain sm_89.
#   2. ``scripts/patch_curope.py`` relaxes ONE over-strict assertion in
#      kernels.cu that rejects Pi3's attention memory layout. The kernel
#      body is untouched - see the script for the details.
#
# Run with:  pixi run bash scripts/build_curope.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/cuda_env.sh
source "${SCRIPT_DIR}/cuda_env.sh"

CROCO_REPO="${CROCO_REPO:-https://github.com/naver/croco.git}"
CROCO_COMMIT="${CROCO_COMMIT:-5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb}"
WORKDIR="${WORKDIR:-/tmp/opencode/curope_build}"
PREFIX="${CONDA_PREFIX:-$REPO_ROOT/.pixi/envs/default}"

export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_HOME="${CUDA_HOME:-$PREFIX}"
export CROCO_ROOT="${WORKDIR}/croco"

echo "==> croco commit : $CROCO_COMMIT"
echo "==> cuda archs   : $CUDA_ARCH_LIST"
echo "==> CUDA_HOME    : $CUDA_HOME"

merge_cuda_layout "$PREFIX"

echo "==> nvcc"
command -v nvcc >/dev/null 2>&1 || { echo "    nvcc not found!" >&2; exit 1; }
nvcc --version | tail -2

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# ---------------------------------------------------------------------------
# 1. get the sources
# ---------------------------------------------------------------------------
if [ ! -d croco/.git ]; then
  echo "==> cloning croco"
  rm -rf croco
  git clone --filter=blob:none "${CROCO_REPO}" croco
fi

cd croco
git fetch --depth 1 origin "${CROCO_COMMIT}" 2>/dev/null || git fetch origin
git checkout --force "${CROCO_COMMIT}"
echo "==> croco at $(git rev-parse HEAD)"

Curope_DIR="models/curope"
[ -d "${Curope_DIR}" ] || { echo "    ${Curope_DIR} not found in croco!" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. relax the one over-strict assertion in the kernel
# ---------------------------------------------------------------------------
echo "==> patching the kernel for Pi3's attention layout"
python "${SCRIPT_DIR}/patch_curope.py" "${Curope_DIR}/kernels.cu"

# ---------------------------------------------------------------------------
# 3. build the CUDA extension with our own (arch-explicit) setup file
# ---------------------------------------------------------------------------
echo "==> building the 'curope' CUDA extension"
cd "${REPO_ROOT}/scripts/curope_build"
# --no-build-isolation: the environment already provides torch / setuptools
python -m pip install --no-build-isolation --no-deps --force-reinstall . 2>&1 | tail -20

# setuptools leaves its build artefacts next to setup.py - remove them so the
# checkout stays clean (and nothing binary can ever be committed)
rm -rf "${REPO_ROOT}/scripts/curope_build/build" \
       "${REPO_ROOT}/scripts/curope_build"/*.egg-info

# ---------------------------------------------------------------------------
# 4. install the python wrapper as the 'models.curope' package
# ---------------------------------------------------------------------------
SITE_PACKAGES="$(site_packages_dir)"
echo "==> installing the python wrapper into ${SITE_PACKAGES}/models/curope"
mkdir -p "${SITE_PACKAGES}/models/curope"
cp -f "${WORKDIR}/croco/${Curope_DIR}/__init__.py" \
      "${WORKDIR}/croco/${Curope_DIR}/curope2d.py" \
      "${SITE_PACKAGES}/models/curope/"
# 'models' stays a namespace package (no __init__.py), exactly as in croco
rm -f "${SITE_PACKAGES}/models/__init__.py"

# ---------------------------------------------------------------------------
# 5. verify
# ---------------------------------------------------------------------------
echo "==> verifying"
python "${REPO_ROOT}/scripts/check_curope.py"
