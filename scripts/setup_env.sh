#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage B - dependency installation for the isolated TrackEverything env.
#
# Follows the upstream README "Installation" section step by step:
#   1. torch + torchvision from the PyTorch cu129 index
#   2. torch_scatter from the PyG wheel index (prebuilt for torch 2.8+cu129)
#   3. everything else from the pinned requirements.txt
#   4. the cuRoPE2D CUDA kernel (croco) used by Pi3
#
# Deliberately does NOT build libs/ops (deformable attention): it is only
# needed for training, not for inference.
#
# Run with:  pixi run bash scripts/setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REQS="external/trackeverything/requirements.txt"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"
PYG_LINKS="https://data.pyg.org/whl/torch-2.8.0+cu129.html"

echo "==> [1/3] torch + torchvision (cu129)"
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "${PYTORCH_INDEX}"

echo "==> [2/3] torch_scatter"
# Preferred path: the prebuilt wheel for torch 2.8.0+cu129.
# NOTE: data.pyg.org is throttled to ~16 KiB per connection from this host,
# so this step is only attempted when the extension is not already present.
# The fallback (build the identical 2.1.2 sources locally) is
# scripts/build_torch_scatter.sh.
if python -c "import torch_scatter, torch; assert torch.__version__.startswith('2.8.0')" 2>/dev/null; then
  echo "    torch_scatter already installed - skipping"
else
  python -m pip install --resume-retries 10 --timeout 120 torch_scatter -f "${PYG_LINKS}" \
    || echo "    !! prebuilt wheel unavailable, run scripts/build_torch_scatter.sh instead"
fi

echo "==> [3/3] remaining pinned requirements"
# requirements.txt already carries --extra-index-url (pytorch cu129) and
# --find-links (pyg cu129), so torch/torchvision resolve to the exact pinned
# local versions and are left untouched if already installed.
#
# torch_scatter is skipped here on purpose: the pin in requirements.txt is
# "==2.1.2+pt28cu129" (a prebuilt-wheel local version tag) which cannot be
# satisfied by the locally compiled extension (version "2.1.2"). It is
# installed by scripts/build_torch_scatter.sh instead. See README §4.
REQ_TMP="$(mktemp)"
grep -vE '^torch_scatter' "${REQS}" > "${REQ_TMP}"
python -m pip install --resume-retries 10 --timeout 120 -r "${REQ_TMP}"
rm -f "${REQ_TMP}"

echo "==> [4/4] the CUDA RoPE2D kernel (croco's cuRoPE2D)"
# Pi3 uses the compiled cuRoPE2D kernel when it is importable and falls back
# to a slow pure-pytorch implementation otherwise. The build needs the CUDA
# toolkit (nvcc 12.9), which the pixi environment provides.
# See scripts/build_curope.sh and README §4.
bash scripts/build_curope.sh || {
  echo "    !! the cuRoPE2D kernel could not be built."
  echo "    !! Pi3 will use its pure-pytorch RoPE fallback (slower)."
}

echo "==> done"
python -m pip list 2>/dev/null | grep -iE "^(torch|torchvision|torch-scatter|torch_scatter|numpy|transformers|huggingface-hub|opencv|imageio|av) " || true
