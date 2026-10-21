#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end smoke test of the standalone TrackEverything pipeline.
#
#   env check -> synthetic clip -> inference -> result validation
#
# Uses a short (4 frame) clip so the test stays fast; it verifies the code
# path, not tracking quality.
#
# Run with:  pixi run bash tests/test_e2e.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="outputs/e2e_test"
FRAMES="${FRAMES:-4}"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "=============================================================="
echo " 1/4  environment"
echo "=============================================================="
pixi run python scripts/check_env.py > /tmp/e2e_check_env.log 2>&1 \
  || { cat /tmp/e2e_check_env.log; fail "environment check"; }
grep -q "ALL CHECKS PASSED" /tmp/e2e_check_env.log || fail "environment check did not pass"
echo "OK"

echo "=============================================================="
echo " 2/4  synthetic test clip"
echo "=============================================================="
pixi run python scripts/make_test_video.py --out test_data/synthetic_16f.mp4 \
  || fail "could not create the test clip"
test -s test_data/synthetic_16f.mp4 || fail "test clip is empty"
echo "OK"

echo "=============================================================="
echo " 3/4  inference (${FRAMES} frames, Pi3 geometry backend)"
echo "=============================================================="
rm -rf "${OUT}"
pixi run python scripts/run_inference.py \
  --video test_data/synthetic_16f.mp4 \
  --output_dir "${OUT}" \
  --geometry_backend pi3 \
  --mode dense \
  --num_iters 3 \
  --voxel_size 0.02 \
  --max_frames "${FRAMES}" \
  || fail "inference returned a non-zero exit code"

test -s "${OUT}/tracks.pt"   || fail "${OUT}/tracks.pt missing or empty"
test -s "${OUT}/geometry.pt" || fail "${OUT}/geometry.pt missing or empty"
echo "OK"

echo "=============================================================="
echo " 4/4  result validation"
echo "=============================================================="
pixi run python scripts/inspect_results.py "${OUT}" || fail "result validation failed"
echo "OK"

echo "=============================================================="
echo " 5/5  ComfyUI node tests (fake worker + real worker)"
echo "=============================================================="
pixi run python tests/test_comfy_nodes.py || fail "node unit tests failed"
pixi run python tests/test_node_e2e.py || fail "node end-to-end test failed"
echo "OK"

echo
echo "=============================================================="
echo " E2E TEST PASSED"
echo "=============================================================="
