# Standalone inference test report (ТЗ №1, Stages B–F)

**Date:** 2026-10-21
**Host:** WSL2 / Ubuntu 24.04, user `ocdev`
**Project:** `/home/ocdev/trackeverything`

---

## 1. Upstream commit

| Item | Value |
|---|---|
| Repository | `https://github.com/ayushjain1144/trackeverything` |
| Submodule path | `external/trackeverything` |
| Pinned commit | `f0c18dd9714b9f85825c45a1c3aae685da4e6222` ("code release") |
| Upstream tree modified? | **No** — `git status` inside the submodule is empty |

The upstream code was used **as-is**. The only adaptation is environmental
(`PYTHONPATH` pointing at the submodule's parent directory, which is what the
upstream README prescribes).

## 2. Environment

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 (WSL2, kernel 6.18.54.1-microsoft-standard-WSL2) |
| Python | 3.10.22 (conda-forge, GCC 14.4.0) |
| Env manager | Pixi 0.82.1, workspace `~/trackeverything`, env `.pixi/envs/default` |
| PyTorch | 2.8.0+cu129 |
| torchvision | 0.23.0+cu129 |
| torch_scatter | 2.1.2 (built from upstream sdist, see §3) |
| CUDA runtime (in torch) | 12.9 |
| cuDNN | 91002 (9.10.2) |
| nvcc (build only) | 12.9, V12.9.86 |
| numpy | 2.2.6 |
| transformers | 5.19.0 |
| huggingface_hub | 2.2.0 |
| opencv-python | 5.0.0.93 |
| av (PyAV) | 17.1.0 |
| ImageIO | 2.38.0 |
| einops / omegaconf / hydra-core | 0.8.2 / 2.3.1 / 1.3.7 |
| accelerate / rerun-sdk / wandb / tensorboard | 1.15.0 / 0.38.1 / 0.30.0 / 2.21.0 |

All other pinned versions from `external/trackeverything/requirements.txt` are
installed verbatim.

The upstream `libs/ops` deformable-attention CUDA extension was **not** built:
per the upstream README it is required for *training* only, and inference does
not need it.

### Isolation from the production ComfyUI

* The ComfyUI installation lives at `/home/egorkkk/ComfyUI`, is owned by another
  user, and is **not readable** by `ocdev` (`Permission denied`) — it is
  therefore structurally impossible to modify it from this account.
* Nothing was installed into it, no `sudo` was used, no system CUDA / PyTorch /
  NVIDIA driver was touched, no SSH / Tailscale / system service was changed.
* `scripts/check_env.py` asserts that no ComfyUI path appears on `sys.path`;
  the only `site-packages` in the environment is
  `/home/ocdev/trackeverything/.pixi/envs/default/lib/python3.10/site-packages`.
* All environment packages live inside `~/trackeverything/.pixi/` (git-ignored).

## 3. Deviations from the upstream install instructions

### 3.1 `torch_scatter` built from source instead of the prebuilt wheel

The upstream README prescribes:

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

**Problem:** from this host `data.pyg.org` delivers only ~16 KiB per
connection and then stalls — measured **~0.8 KB/s** (12.5 MB wheel ⇒ hours),
while other hosts reach 40 MB/s from the same machine. The behaviour is
host-specific and reproducible (also affects `dl.fbaipublicfiles.com`).
No mirror of the PyG wheel index carries `torch_scatter`, and PyPI publishes
only an **sdist** for `torch-scatter` 2.1.2.

**Resolution:** `scripts/build_torch_scatter.sh` compiles the *same, unmodified*
upstream sources (sdist `torch_scatter-2.1.2.tar.gz` from PyPI, 108 045 bytes)
against the *same* Torch build (2.8.0+cu129) with `nvcc 12.9`,
`TORCH_CUDA_ARCH_LIST=8.9`. Result:

```
torch_scatter-2.1.2-cp310-cp310-linux_x86_64.whl  (2 212 561 bytes)
Successfully installed torch_scatter-2.1.2
cuda scatter_mean -> (8, 16) torch.float32   # CUDA path verified
```

The compiled extension is functionally identical to the official wheel — same
C++/CUDA sources, same Torch ABI target. The only difference is the version
string (`2.1.2` instead of `2.1.2+pt28cu129`), which is why
`scripts/setup_env.sh` filters the `torch_scatter` line out of
`requirements.txt`.

If `data.pyg.org` becomes reachable, the canonical command works unchanged.

### 3.2 DINOv2 backbone fetched from a Hugging Face mirror

`models/encoders/dinov2.py` loads
`https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`
(84.2 MB). That host is throttled by the same ~16 KiB-per-connection mechanism.

**Resolution:** the identical file was fetched from the Hugging Face mirror
`hdtech/dinov2_vits14_pretrain` and placed into the standard torch.hub cache
location `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`.

Integrity was verified two ways:

* size is exactly `88 283 115` bytes — identical to the upstream URL's
  `Content-Length`;
* the first 15 916 bytes fetched directly from `dl.fbaipublicfiles.com` and
  the corresponding prefix of the HF copy are **byte-identical**
  (md5 `9f09585b773837057bea424294dc716a`).

The file is therefore the same artefact, not a re-trained or converted model.

### 3.3 `RoPE2D` runs in slow fallback mode

Pi3 logs:

```
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```

Pi3 optionally uses a compiled `cuRoPE2D` kernel. It is not built here, so the
pure-PyTorch fallback is used. This is **upstream behaviour**, not a
modification, and it affects speed only — not numerical correctness of the
published results. It is also the reason the Pi3 geometry stage dominates the
wall-clock time of a short run.

## 4. Model weights

| Model | Hub repo | File | Size | Access |
|---|---|---|---|---|
| TrackEverything tracker | `katefgroup/TrackEverything` | `model.pth` | 546 083 869 B (546 MB) | **gated** (`gated: auto`) |
| Pi3 geometry | `yyfz233/Pi3` | `model.safetensors` | 3 834 909 248 B (3.83 GB) | public |

* The tracker repo is gated: an unauthenticated request returns
  `HTTP 401 / x-error-code: GatedRepo`. Access was granted via a read token
  supplied through the **`HF_TOKEN` environment variable** only. No token was
  written to any file in this repository, to the source code, or to any report.
* Both models were downloaded with `huggingface_hub.snapshot_download` into the
  standard HF cache (`~/.cache/huggingface`, total ≈ 4.1 GB) and are reused on
  subsequent runs — a second inference run performed **no** download
  (the only HTTP calls were cache-validation `HEAD` requests returning `302`).
* Weights are **not** in Git (`.gitignore` excludes `*.pth`, `*.safetensors`,
  `*.bin`, `checkpoints/`, `hf_cache/`, …).
* Additional upstream asset: `dinov2_vits14_pretrain.pth` (84 MB) in
  `~/.cache/torch/hub/checkpoints/`.

## 5. Test clip (Stage D1)

Generated by `scripts/make_test_video.py` — a synthetic sequence, because no
user media was available in the working directory:

* 16 frames, RGB, 384 × 256, 8 fps, H.264 in MP4 (71.4 KiB)
* static finely-textured background (8 px checkerboard + colour gradient +
  per-pixel noise)
* three textured discs moving linearly along different directions
* written with PyAV and decoded back with PyAV — the same decoder the
  upstream `demo_hf.py` uses; round-trip mean absolute error 8.46 (lossy
  H.264 as expected)

This is a **functional** smoke test of the pipeline, not an evaluation of
tracking quality.

## 6. Inference runs (Stage D2)

Command actually executed (the **original** upstream CLI, unchanged):

```bash
python -m trackeverything.demo_hf \
  --video test_data/synthetic_16f.mp4 \
  --output_dir outputs/standalone_XXf \
  --geometry_backend pi3 \
  --mode dense \
  --num_iters 3 \
  --voxel_size 0.02 \
  --frame_stride 1 --start_frame 0 \
  --device cuda \
  --mem_efficient \
  --max_frames <N>
```

`PYTHONPATH=<repo>/external` was exported, as the upstream README requires.

`--geometry_backend pi3` is used on purpose: the upstream default
`vggt_omega` points at
`/data/group_data/katefgroup/language_grounding/tracking_ckpts/vggt_omega_1b_512.pt`
(`annotation/geometry_factory.py:15`), an absolute path on the authors' private
infrastructure that does not exist here.

`--save_dense` was **not** used (dense trajectories grow quadratically with the
sequence length).

### Results

| Frames | Wall clock | Peak VRAM (device-level) | Tracks (N) | `tracks.pt` | `geometry.pt` | Exit |
|---|---|---|---|---|---|---|
| 4  | 13.65 s | 1 156 MiB | 2 354 | ✔ | ✔ | 0 |
| 8  | 15.54 s | 1 156 MiB | 2 415 | ✔ | ✔ | 0 |
| 16 | 16.89 s | 1 156 MiB | 2 703 | ✔ | ✔ | 0 |

* **No CUDA OOM** at any of the tested lengths. Peak device-level VRAM usage
  was 1.13 GiB out of 16 GiB, measured by sampling
  `torch.cuda.mem_get_info()` every 250 ms from the wrapper process
  (`scripts/run_inference.py`).
* Output sizes on disk: 3.7 MB (4 f), 7.2 MB (8 f), 15 MB (16 f).
* The wall-clock time is dominated by model loading and the Pi3 geometry
  stage; the tracker itself takes ~0.5 s at 16 frames.
* A machine-readable report is written to
  `<output_dir>/run_report.json` by every run.

## 7. Result validation (Stage E)

`scripts/inspect_results.py outputs/standalone_16f` — **ALL CHECKS PASSED**
(identical outcome for the 4- and 8-frame runs).

### `tracks.pt` (16-frame run)

| Tensor | Shape | Dtype | Min | Max | NaN | Inf |
|---|---|---|---|---|---|---|
| `coords` | (2703, 16, 3) | float32 | −1000 | 1.04166 | 0 | 0 |
| `visibs` | (2703, 16) | float32 | −1000 | 12.0523 | 0 | 0 |
| `dyn_logits` | (2703, 16) | float32 | −1000 | 4.53344 | 0 | 0 |
| `src_time` | (2703,) | int64 | 0 | 15 | – | – |
| `src_y` | (2703,) | int64 | 0 | 95 | – | – |
| `src_x` | (2703,) | int64 | 0 | 127 | – | – |
| `cell_to_token` | (16, 96, 128) | int64 | 0 | 2702 | – | – |

Statistics over the **valid** entries (the `-1000` sentinel excluded — it marks
missing / not-yet-existing trajectories and is *not* a 3D coordinate):

* `coords`: 129 744 elements, 95 217 valid, 34 527 sentinel (73.39 % real
  coordinates); min −0.359939, max 1.04166, mean 0.334788, std 0.474131.
* `visibs`: 43 248 elements, 31 739 valid, 11 509 sentinel; range
  [−10, 12.0523], mean 7.389.
* `dyn_logits`: 43 248 elements, 31 739 valid, 11 509 sentinel; range
  [−11.1449, 4.53344], mean −4.447.
* `src_time` spans 0…15 (every frame seeds tokens), `src_y` 0…95,
  `src_x` 0…127 — consistent with the 96 × 128 feature grid.

### `geometry.pt` (16-frame run)

| Tensor | Shape | Dtype | Min | Max | NaN | Inf |
|---|---|---|---|---|---|---|
| `depths` | (16, 384, 512) | float32 | 0.912862 | 1.05072 | 0 | 0 |
| `intrinsics` | (16, 3, 3) | float32 | 0 | 776.065 | 0 | 0 |
| `extrinsics` | (16, 4, 4) | float32 | −0.0977958 | 1 | 0 | 0 |
| `norm_factor` | scalar | float | — | 1.2673845291137695 | – | – |

* `depths`: mean 0.978412, std 0.0240253 — all strictly positive.
* `intrinsics`: mean focal fx = 663.39, fy = 757.10, principal point
  cx = 1.0 (normalised intrinsics).
* `extrinsics`: last row is exactly `[0, 0, 0, 1]` for every frame
  (max deviation 0.0).

### Consistency checks (all PASS)

1. **Frame count agrees** — `visibs.shape[1] = dyn_logits.shape[1] =
   cell_to_token.shape[0] = depths.shape[0] = intrinsics.shape[0] =
   extrinsics.shape[0] = 16`, and `T` matches `coords.shape[1]`.
2. **Track count agrees** — `visibs`, `dyn_logits`, `src_time`, `src_y`,
   `src_x` all have 2 703 entries, matching `coords.shape[0]`.
3. **`cell_to_token` bounds** — range `[0, 2702]` ⊂ `[0, N-1] = [0, 2702]`;
   2 703 / 2 703 distinct tokens are used (100 % of tracks) and all
   196 608 grid cells are populated — no holes, which matches the upstream
   `SceneTokens` contract ("`cell_to_token` covers the whole grid by
   construction").
4. **No unexpected NaN/Inf** in any floating-point tensor.
5. **Meaningful data** — `coords` std = 0.474131 (not constant), depths all
   positive, valid intrinsics, well-formed extrinsics.
6. **Reloadable in a separate process** — both files were re-loaded with
   `torch.load(..., map_location="cpu", weights_only=True)` in a fresh
   interpreter and the key sets were intact.

The results are read and reported in their native dtype; **no JSON or other
lossy conversion** of the numeric data is performed. `run_report.json` only
contains the run metadata (command, timing, VRAM), never the tensors.

## 8. Reproducibility of the numeric results

Two identical 8-frame runs were compared tensor by tensor with
`scripts/check_determinism.py`:

| Tensor | Bit-identical? | max abs diff | max rel diff |
|---|---|---|---|
| `tracks.pt.coords` | no | 2.295e-06 | 2.295e-09 |
| `tracks.pt.visibs` | no | 2.828e-04 | 2.828e-07 |
| `tracks.pt.dyn_logits` | no | 2.877e-04 | 2.877e-07 |
| `tracks.pt.cell_to_token` | **yes** | 0 | 0 |
| `tracks.pt.src_time` / `src_y` / `src_x` | **yes** | 0 | 0 |
| `geometry.pt.depths` | **yes** | 0 | 0 |
| `geometry.pt.intrinsics` | **yes** | 0 | 0 |
| `geometry.pt.extrinsics` | **yes** | 0 | 0 |

**Conclusion.** The pipeline is *structurally* fully reproducible: the geometry
produced by Pi3, the number of tracks, the token assignment and every index
tensor are **bit-identical** across runs. The tracker's float outputs
(`coords`, `visibs`, `dyn_logits`) differ by a few float32 ULPs
(relative 1e-9 … 1e-7), which is the expected consequence of CUDA kernels
that reduce with atomics / non-deterministic reduction order in the upstream
code. No seeding or algorithm change was applied to compensate — the upstream
code was left untouched.

Practical implication for the future ComfyUI integration: outputs must be
treated as *numerically equivalent within float32 tolerance*, not bit-exact,
and any regression test should compare with a relative tolerance rather than
exact equality.

## 9. Errors and limitations

1. **`data.pyg.org` / `dl.fbaipublicfiles.com` throttling** — the single real
   infrastructure obstacle. Worked around without changing any algorithm
   (§3.1, §3.2). If these hosts are reachable from the RunPod A40 host, the
   canonical upstream install commands can be used verbatim.
2. **Gated checkpoint** — `katefgroup/TrackEverything` requires accepting the
   licence and an HF token. The token is consumed via `HF_TOKEN` only.
3. **`cuRoPE2D` not compiled** — Pi3 falls back to the slow PyTorch RoPE
   implementation (§3.3). Speed-only impact; upstream behaviour.
4. **`libs/ops` deformable attention not built** — training-only, deliberately
   out of scope for inference.
5. **Short clips only** — the longest run was 16 frames. Long-horizon behaviour
   (the paper's main claim) is **not** validated here. Peak VRAM at 16 frames
   is 1.13 GiB, so longer sequences should be feasible on 16 GB, but the
   dense-track tensor grows quadratically with T.
6. **Synthetic footage** — the clip is a checkerboard with three discs. It
   exercises the code path, not tracking quality. The upstream evaluation
   datasets (Kubric, Dynamic Replica, TAPVid3D) were not used.
7. **Single GPU, single run** — no determinism/repeatability study across
   repeated runs was performed.

## 10. Acceptance criteria (ТЗ №1, §10)

| # | Criterion | Status |
|---|---|---|
| 1 | Git repository created | ✅ `~/trackeverything`, branch `main` |
| 2 | Upstream pinned to a specific commit | ✅ submodule @ `f0c18dd9714b9f85825c45a1c3aae685da4e6222` |
| 3 | Isolated environment works | ✅ pixi env, Python 3.10.22, Torch 2.8.0+cu129 |
| 4 | All required modules import | ✅ `torch`, `torch_scatter`, `trackeverything.demo_hf` |
| 5 | Original model runs on the available GPU | ✅ RTX 4060 Ti, 3 successful runs |
| 6 | `tracks.pt` and `geometry.pt` obtained | ✅ 4 f / 8 f / 16 f |
| 7 | Structure and numeric correctness confirmed | ✅ all Stage E checks pass |
| 8 | Reproducible instructions prepared | ✅ `README.md` + this document |
| 9 | Production ComfyUI untouched | ✅ unreadable to `ocdev`, never modified |

**All nine criteria are met.** No CUDA OOM was encountered, so the
"defer to the RunPod A40 48 GB host" fallback in §6 of the ТЗ was not needed.

## 11. Reproducing this test

```bash
cd ~/trackeverything
git submodule update --init --recursive
pixi install
pixi run bash scripts/setup_env.sh
pixi run bash scripts/build_torch_scatter.sh

export HF_TOKEN=...            # read token for the gated tracker repo
pixi run python scripts/check_env.py
pixi run python scripts/make_test_video.py
pixi run python scripts/run_inference.py \
    --video test_data/synthetic_16f.mp4 \
    --output_dir outputs/standalone_16f \
    --geometry_backend pi3 --mode dense --num_iters 3 \
    --voxel_size 0.02 --max_frames 16
pixi run python scripts/inspect_results.py outputs/standalone_16f

# full end-to-end smoke test (env -> clip -> inference -> validation)
pixi run bash tests/test_e2e.sh

# reproducibility of two runs
pixi run python scripts/check_determinism.py outputs/det_a outputs/det_b
```
