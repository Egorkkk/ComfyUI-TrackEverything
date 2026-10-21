#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-fetch the DINOv2 ViT-S/14 backbone that TrackEverything loads.
#
# Upstream (`models/encoders/dinov2.py`) pulls it from
#   https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
#
# That host is throttled to ~16 KiB per connection from this machine (same
# mechanism as data.pyg.org), so the file is fetched from the Hugging Face
# mirror `hdtech/dinov2_vits14_pretrain` instead and dropped into the standard
# torch.hub cache location. torch.hub then finds it and skips the download.
#
# Integrity is verified by size AND by comparing the first bytes against the
# upstream host (which is reachable, just slow).
#
# Run with:  pixi run bash scripts/fetch_dinov2_weights.sh
# ---------------------------------------------------------------------------
set -euo pipefail

FILENAME="dinov2_vits14_pretrain.pth"
UPSTREAM_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/${FILENAME}"
MIRROR_URL="https://huggingface.co/hdtech/dinov2_vits14_pretrain/resolve/main/${FILENAME}"
CACHE_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
TARGET="${CACHE_DIR}/${FILENAME}"

mkdir -p "${CACHE_DIR}"

if [ -s "${TARGET}" ]; then
  echo "==> already present: ${TARGET} ($(stat -c%s "${TARGET}") bytes)"
  exit 0
fi

echo "==> expected size from upstream:"
EXPECTED=$(curl -sSIL --max-time 30 "${UPSTREAM_URL}" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{n=$2}END{gsub(/\r/,"",n);print n}')
echo "    ${EXPECTED} bytes"

echo "==> downloading from the Hugging Face mirror"
curl -fSL --retry 5 --retry-all-errors --max-time 600 -o "${TARGET}.part" "${MIRROR_URL}"

ACTUAL=$(stat -c%s "${TARGET}.part")
if [ "${ACTUAL}" != "${EXPECTED}" ]; then
  echo "!! size mismatch: got ${ACTUAL}, expected ${EXPECTED}" >&2
  rm -f "${TARGET}.part"
  exit 1
fi

echo "==> verifying the first bytes against the upstream host"
PROBE_BYTES=15916
curl -sS --max-time 60 -r "0-$((PROBE_BYTES - 1))" -o "${TARGET}.probe" "${UPSTREAM_URL}" || true
if [ -s "${TARGET}.probe" ]; then
  A=$(head -c "${PROBE_BYTES}" "${TARGET}.part" | md5sum | cut -d' ' -f1)
  B=$(md5sum "${TARGET}.probe" | cut -d' ' -f1)
  if [ "${A}" != "${B}" ]; then
    echo "!! prefix mismatch (hf=${A} upstream=${B})" >&2
    rm -f "${TARGET}.part" "${TARGET}.probe"
    exit 1
  fi
  echo "    prefix md5 ${A} - identical to upstream"
fi
rm -f "${TARGET}.probe"

mv "${TARGET}.part" "${TARGET}"
echo "==> OK: ${TARGET} (${ACTUAL} bytes)"
