#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage B - dependency installation for the isolated TrackEverything env.
#
# Follows the upstream README "Installation" section step by step:
#   1. torch + torchvision from the PyTorch cu129 index
#   2. torch_scatter from the PyG wheel index (prebuilt for torch 2.8+cu129)
#   3. everything else from the pinned requirements.txt
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

echo "==> [2/3] torch_scatter (prebuilt for torch 2.8.0+cu129)"
python -m pip install torch_scatter -f "${PYG_LINKS}"

echo "==> [3/3] remaining pinned requirements"
# requirements.txt already carries --extra-index-url (pytorch cu129) and
# --find-links (pyg cu129), so torch/torch_scatter resolve to the exact
# pinned local versions and are left untouched if already installed.
python -m pip install -r "${REQS}"

echo "==> done"
python -m pip list 2>/dev/null | grep -iE "^(torch|torchvision|torch-scatter|torch_scatter|numpy|transformers|huggingface-hub|opencv|imageio|av) " || true
