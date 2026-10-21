#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build + install `torch_scatter` 2.1.2 from upstream source.
#
# WHY NOT THE PREBUILT WHEEL?
#   The upstream README does:
#       pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
#   From this machine data.pyg.org serves exactly ~16 KiB per connection and
#   then stalls (measured ~800 B/s), while other hosts reach 40 MB/s. There
#   is no mirror of the PyG wheel index that carries torch_scatter, and PyPI
#   only publishes an sdist for torch-scatter 2.1.2.
#
#   So we compile the *same* upstream version (2.1.2, sdist from PyPI) against
#   the *same* Torch build (2.8.0+cu129), using the nvcc 12.9 provided by the
#   pixi environment. The resulting extension is numerically equivalent to the
#   official wheel - it is the same C++/CUDA sources.
#
# Run with:  pixi run bash scripts/build_torch_scatter.sh
# ---------------------------------------------------------------------------
set -euo pipefail

VERSION="2.1.2"
WORKDIR="${WORKDIR:-/tmp/opencode/torch_scatter_build}"
PREFIX="${CONDA_PREFIX:-$PWD/.pixi/envs/default}"

# ---------------------------------------------------------------------------
# conda-forge splits the CUDA toolkit across several packages and keeps the
# real headers in <prefix>/targets/x86_64-linux/include, while nvcc (a real
# binary in <prefix>/bin) looks for them in <prefix>/include. Torch's
# cpp_extension additionally expects <prefix>/lib64.
#
# Merge the two views with symlinks so that a plain CUDA_HOME=<prefix> works.
# This only touches the (git-ignored) pixi environment.
# ---------------------------------------------------------------------------
CUDA_TARGETS="$PREFIX/targets/x86_64-linux"
if [ -d "$CUDA_TARGETS" ]; then
  echo "==> merging conda-forge CUDA layout into a single toolkit root"
  for f in "$CUDA_TARGETS"/include/*; do
    b="$(basename "$f")"
    [ -e "$PREFIX/include/$b" ] || ln -sfn "$f" "$PREFIX/include/$b"
  done
  for f in "$CUDA_TARGETS"/lib/*; do
    b="$(basename "$f")"
    [ -e "$PREFIX/lib/$b" ] || ln -sfn "$f" "$PREFIX/lib/$b"
  done
  [ -e "$PREFIX/lib64" ] || ln -sfn "$PREFIX/lib" "$PREFIX/lib64"
  [ -e "$PREFIX/nvvm" ] || ln -sfn "$CUDA_TARGETS/nvvm" "$PREFIX/nvvm"
fi

# The GPU is an RTX 4060 Ti (Ada Lovelace, sm_89).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_HOME="${CUDA_HOME:-$PREFIX}"

echo "==> prefix        : $PREFIX"
echo "==> CUDA_HOME     : $CUDA_HOME"
echo "==> nvcc          : $(command -v nvcc || echo '<not found>')"
nvcc --version 2>/dev/null | tail -2 || true

mkdir -p "$WORKDIR"
cd "$WORKDIR"

SDIST="torch_scatter-${VERSION}.tar.gz"
URL="https://files.pythonhosted.org/packages/source/t/torch-scatter/${SDIST}"
if [ ! -f "$SDIST" ]; then
  echo "==> downloading sdist"
  curl -fsSL --retry 5 --retry-all-errors -o "$SDIST" "$URL"
fi
rm -rf "torch_scatter-${VERSION}"
tar xzf "$SDIST"

cd "torch_scatter-${VERSION}"

# ninja makes the extension build far faster (torch's cpp_extension uses it
# when available; otherwise it falls back to a serial build).
python -m pip install --quiet --upgrade ninja setuptools wheel 2>/dev/null || true

echo "==> building torch_scatter ${VERSION} (CUDA, arch ${TORCH_CUDA_ARCH_LIST})"
# --no-build-isolation: the env already provides torch / setuptools.
python -m pip install --no-build-isolation --no-deps -v . 2>&1 | tail -40

echo "==> verifying"
# Verify from a neutral cwd: inside the source tree the local `torch_scatter/`
# directory would shadow the freshly installed package.
cd "$WORKDIR"
python - <<'PY'
import torch, torch_scatter
print("torch      :", torch.__version__)
print("scatter    :", torch_scatter.__version__)
src = torch.randn(64, 16, device="cuda")
idx = torch.randint(0, 8, (64,), device="cuda")
out = torch_scatter.scatter_mean(src, idx, dim=0)
torch.cuda.synchronize()
print("cuda scatter_mean ->", tuple(out.shape), out.dtype)
assert out.shape == (8, 16)
print("OK")
PY
