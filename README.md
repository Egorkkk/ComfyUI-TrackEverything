# TrackEverything — standalone inference (development sandbox)

This repository is an **isolated development environment** for integrating the
upstream [TrackEverything](https://github.com/ayushjain1144/trackeverything)
project (CMU / Meta, *"Long Horizon Dense Tracking via De-Duplicating 3D Scene
Representations"*) into ComfyUI for VFX and 3D point tracking.

**Current stage (ТЗ №1):** Git project layout, an isolated Python environment,
and a working **standalone** run of the *original* upstream inference code.
ComfyUI nodes are intentionally **not** part of this stage.

Two hard rules govern everything in this repo:

1. **The production ComfyUI at `/home/egorkkk/ComfyUI` is never touched.**
   No file changes, no packages installed into its Python environment, no
   changes to system CUDA / PyTorch / NVIDIA drivers, no `sudo`.
2. **The upstream TrackEverything repository is not modified.**
   It is consumed as a Git submodule, pinned to a specific commit.

---

## 1. Layout

```text
~/trackeverything/
├── comfy_nodes/            # ComfyUI nodes (installed into custom_nodes/)
│   ├── __init__.py             # NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
│   ├── nodes.py                # Track + Load Data + Save Data + Visualize
│   ├── bridge.py               # IMAGE batch -> PNG -> worker -> TRACK_DATA
│   ├── config.py               # runtime path resolution + isolation guard
│   └── types.py                # the TRACK_DATA payload
├── worker/                 # runs inside the isolated environment
│   ├── __init__.py
│   ├── runner.py               # calls the original upstream demo_hf
│   └── visualize.py            # calls the original upstream renderer (CPU)
├── external/
│   └── trackeverything/    # upstream, Git submodule (read-only, pinned)
├── scripts/                # setup / check / test / inspection helpers
│   ├── setup_env.sh            # install torch, torchvision, deps
│   ├── build_torch_scatter.sh  # build torch_scatter from source (see §4)
│   ├── fetch_dinov2_weights.sh # DINOv2 backbone from an HF mirror (see §4)
│   ├── check_env.py            # Stage B3 environment verification
│   ├── make_test_video.py      # Stage D1 synthetic test clip
│   ├── run_inference.py        # Stage D run + timing / peak VRAM
│   ├── inspect_results.py      # Stage E tracks.pt / geometry.pt validation
│   ├── check_determinism.py    # reproducibility of two runs
│   └── diagnose_vram.py        # per-stage CUDA memory diagnostics
├── tests/
│   ├── test_e2e.sh          # end-to-end smoke test
│   ├── test_comfy_nodes.py  # node unit tests (fake worker, no GPU)
│   ├── test_node_e2e.py     # node end-to-end test (real model)
│   ├── test_save_data.py    # Save Data node tests
│   ├── test_visualize.py    # Visualize node tests (synthetic scene)
│   ├── test_visualize_e2e.py# Visualize end-to-end test (standalone results)
│   └── fake_worker.py       # stand-in worker used by the unit tests
├── test_data/              # test clips (git-ignored)
├── outputs/                # inference results (git-ignored)
├── docs/                   # reports
│   ├── standalone-test.md
│   ├── vram-diagnostics.md
│   └── comfyui-install.md       # installation guide for a ComfyUI user
├── config.json.example     # optional runtime configuration
├── pixi.toml               # isolated environment definition
├── pixi.lock               # pinned environment lock
├── README.md
├── .gitignore
└── .gitmodules
```

## 2. Requirements

* WSL2 Ubuntu 24.04, user `ocdev`
* NVIDIA GPU with CUDA (tested: RTX 4060 Ti, 16 GB, driver 610.57.01)
* `git`, `curl`, `gcc`/`g++`
* ~25 GB free disk (torch + CUDA toolkit + model weights)

## 3. Install

```bash
cd ~/trackeverything

# 3.1 fetch the upstream submodule (already pinned in .gitmodules / the index)
git submodule update --init --recursive
git -C external/trackeverything log --oneline -1
# -> f0c18dd code release

# 3.2 create the isolated environment (Python 3.10, conda-forge via pixi)
pixi install

# 3.3 install torch 2.8.0+cu129 / torchvision 0.23.0 / torch_scatter / deps
pixi run bash scripts/setup_env.sh
pixi run bash scripts/build_torch_scatter.sh   # see §4 for why
pixi run bash scripts/fetch_dinov2_weights.sh  # see §4 for why

# 3.4 verify the environment
pixi run python scripts/check_env.py
```

Everything is installed **inside** `~/trackeverything/.pixi/envs/default`.
Nothing is installed globally, and nothing is written to the ComfyUI
installation.

### Using the environment

```bash
pixi run python ...        # run a command inside the env
pixi shell                 # interactive shell inside the env
```

`pixi shell` is the recommended way to work interactively.

## 4. Two upstream downloads that had to be re-routed

From this machine two upstream asset hosts are throttled to **~16 KiB per
connection** and then stall (measured ~0.8 KB/s, while
`download.pytorch.org`, `huggingface.co` and PyPI reach 40 MB/s from the same
host). Both are re-routed **without changing any algorithm**:

### 4.1 `torch_scatter` — built from source

The upstream README installs the prebuilt wheel:

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

`data.pyg.org` is throttled as described above. No mirror of the PyG wheel
index carries `torch_scatter`, and PyPI only ships an sdist for
`torch-scatter` 2.1.2.

`scripts/build_torch_scatter.sh` therefore compiles **the same upstream version
(2.1.2, unmodified sdist from PyPI)** against **the same Torch build
(2.8.0+cu129)** using `nvcc 12.9` from the pixi environment. The result is
numerically equivalent to the official wheel — it is the same C++/CUDA sources.

If `data.pyg.org` becomes reachable, the canonical command works unchanged and
`scripts/setup_env.sh` (step 2) will use it.

### 4.2 DINOv2 backbone — fetched from a Hugging Face mirror

The DINOv2 ViT-S/14 weights are pulled from
`dl.fbaipublicfiles.com` (same throttling). `scripts/fetch_dinov2_weights.sh`
fetches the identical file from the Hugging Face mirror
`hdtech/dinov2_vits14_pretrain` into the standard torch.hub cache and verifies
it against the upstream host (exact size + byte-identical prefix).

## 5. Model weights

| Model | Hub repo | Size | Access |
|---|---|---|---|
| TrackEverything tracker | `katefgroup/TrackEverything` (`model.pth`) | 546 MB | **gated** — requires accepting the licence + an HF token |
| Pi3 geometry | `yyfz233/Pi3` (`model.safetensors`) | 3.83 GB | public |

Weights are downloaded by `huggingface_hub` into the standard HF cache
(`$HF_HOME`, default `~/.cache/huggingface`) and are reused on subsequent runs
— nothing is re-downloaded and nothing is stored in Git.

For the gated repo, provide a read token **without** writing it into any file
in this repository:

```bash
export HF_TOKEN=hf_...            # or: hf auth login
```

The upstream CLI also accepts `--token`, but exporting `HF_TOKEN` is preferred.

## 6. Run standalone inference

```bash
# 6.1 make the test clip (16 frames, 384x256, synthetic)
pixi run python scripts/make_test_video.py

# 6.2a run the ORIGINAL upstream CLI directly
cd ~/trackeverything
PYTHONPATH="$PWD/external" pixi run python -m trackeverything.demo_hf \
  --video test_data/synthetic_16f.mp4 \
  --output_dir outputs/standalone_16f \
  --geometry_backend pi3 \
  --mode dense \
  --num_iters 3 \
  --voxel_size 0.02 \
  --mem_efficient \
  --max_frames 16

# 6.2b same thing through a wrapper that also records time and peak VRAM
pixi run python scripts/run_inference.py \
  --video test_data/synthetic_16f.mp4 \
  --output_dir outputs/standalone_16f \
  --geometry_backend pi3 --mode dense \
  --num_iters 3 --voxel_size 0.02 --max_frames 16
```

Notes:

* `PYTHONPATH` must contain `external/` — the upstream repo *is* the
  `trackeverything` package, so its **parent** directory has to be importable.
  `scripts/run_inference.py` sets this automatically.
* `--geometry_backend pi3` is used deliberately. The upstream default is
  `vggt_omega`, whose checkpoint path is hard-coded to the authors' private
  infrastructure (`/data/group_data/katefgroup/...`).
* `--save_dense` is **not** enabled on the first run: dense trajectories grow
  quadratically with the sequence length and easily exhaust VRAM.
* `--mem_efficient` is an upstream `BooleanOptionalAction` flag — it takes no
  value; `--no-mem_efficient` disables it.

Outputs written to `outputs/standalone_16f/`:

| File | Content |
|---|---|
| `tracks.pt` | `coords`, `visibs`, `dyn_logits`, `src_time`, `src_y`, `src_x`, `cell_to_token` |
| `geometry.pt` | `depths`, `intrinsics`, `extrinsics`, `norm_factor` |
| `run_report.json` | run metadata only (command, wall clock, peak VRAM) — no tensors |

## 7. Inspect the results

```bash
pixi run python scripts/inspect_results.py outputs/standalone_16f
```

The script prints shapes, dtypes and statistics for every tensor and then
validates:

1. the frame count agrees across `coords` / `visibs` / `dyn_logits` /
   `cell_to_token` / `depths` / `intrinsics` / `extrinsics`;
2. `cell_to_token` indices stay inside the valid token range;
3. there are no unexpected NaN/Inf values;
4. the data is numerically meaningful (non-constant, positive depths, valid
   intrinsics/extrinsics);
5. both files reload cleanly in a **separate** process.

The `-1000` sentinel used by upstream for "missing / not-yet-existing"
trajectory entries is excluded from the statistics — it is not a 3D coordinate.
Results are never converted to JSON or any other lossy format.

## 7.1 Tests

```bash
# end-to-end: env check -> synthetic clip -> inference -> validation
pixi run bash tests/test_e2e.sh

# ComfyUI node tests
pixi run test-nodes        # unit tests against a fake worker (no GPU)
pixi run test-nodes-e2e   # real inference through the Track node
pixi run test-save-data   # Save Data node
pixi run test-visualize   # Visualize node (synthetic scene)
pixi run test-visualize-e2e  # Visualize on the standalone results

# reproducibility: two runs must agree within float32 tolerance
pixi run python scripts/check_determinism.py outputs/det_a outputs/det_b
```

Note: the upstream tracker is **not** bit-reproducible on GPU (CUDA kernels
that reduce with atomics), but the geometry and all index tensors are
bit-identical and the float outputs agree to ~1e-7 relative. See
`docs/standalone-test.md` §8.

## 8. Re-downloading / reusing the models

```bash
# explicit pre-fetch (optional - inference downloads on demand too)
pixi run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('yyfz233/Pi3'); snapshot_download('katefgroup/TrackEverything')"

# where the weights live
pixi run python -c "import huggingface_hub as h; print(h.constants.HF_HUB_CACHE)"
```

Set `HF_HOME` to relocate the cache. The cache is outside the repository and is
git-ignored.

## 9. Docs

* [`docs/standalone-test.md`](docs/standalone-test.md) — versions, GPU, run
  parameters, timings, VRAM, results and known limitations (ТЗ №1).
* [`docs/vram-diagnostics.md`](docs/vram-diagnostics.md) — per-stage CUDA
  memory measurements of the worker.
* [`docs/comfyui-install.md`](docs/comfyui-install.md) — **installation guide
  for a ComfyUI user**: environment, models, node installation, configuration,
  first workflow and removal (ТЗ №2).

## 10. ComfyUI nodes

The nodes are in `comfy_nodes/` and can be installed into any ComfyUI:

```bash
ln -s ~/trackeverything/comfy_nodes \
      /path/to/ComfyUI/custom_nodes/comfy_nodes_trackeverything
```

| Node | Input | Output | GPU |
|---|---|---|---|
| **TrackEverything Track** | `images` (IMAGE batch), `num_iters` (3), `voxel_size` (0.02), `mem_efficient` (true) | `TRACK_DATA` | yes, in a worker process |
| **TrackEverything Load Data** | `result_dir` (STRING) | `TRACK_DATA` | no |
| **TrackEverything Save Data** | `track_data`, `save_mode` (`reference`/`copy`), `filename_prefix` | `TRACK_DATA` | no (`OUTPUT_NODE`) |
| **TrackEverything Visualize** | `track_data`, `max_points` (5000), `trail_length` (8), `visibility_threshold` (0.5), `dynamic_threshold` (0.5), `background_opacity` (0.7), optional `images` (IMAGE batch) | `images` (IMAGE batch) | no |

All four nodes exchange the same `TRACK_DATA` handle (file references +
metadata, never tensors), so a workflow can be rewired freely between them.

**Two minimal workflows**

```
Inference:
  Load Video ──▶ TrackEverything Track ──▶ TrackEverything Save Data

Visualization (no repeated inference):
  TrackEverything Load Data ──▶ TrackEverything Visualize ──▶ Preview Image
```

For the visualization workflow, connect the original `IMAGE` batch to the
optional `images` input of the Visualize node to draw the tracks on top of the
footage.

**Isolation.** The ComfyUI process never imports `trackeverything`,
`torch_scatter` or anything from the worker's Python 3.10 environment — an
isolation guard in `comfy_nodes/config.py` fails loudly if that ever happens.
The tracker runs in a separate process with its own CUDA context, so its VRAM
is released completely when the run ends; ComfyUI's own models are never
touched. The visualizer is CPU-only and loads no model at all.

**Configuration** (nothing is hard-coded to a user account):

```bash
export TRACKEVERYTHING_PROJECT_ROOT=~/trackeverything
export TRACKEVERYTHING_WORKER_PYTHON=~/trackeverything/.pixi/envs/default/bin/python
# or copy config.json.example to config.json in the project root
```

Full instructions: [`docs/comfyui-install.md`](docs/comfyui-install.md).

## 11. Status / known limitations

See `docs/standalone-test.md` for the ТЗ №1 state and
`docs/vram-diagnostics.md` for the memory figures.
