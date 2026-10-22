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
│   ├── nodes.py                # the five node classes
│   ├── bridge.py               # IMAGE batch -> PNG -> worker -> TRACK_DATA
│   ├── config.py               # runtime path resolution + isolation guard
│   └── types.py                # the TRACK_DATA / TRACK_SELECTION payloads
├── worker/                 # runs inside the isolated environment
│   ├── __init__.py
│   ├── runner.py               # calls the original upstream demo_hf
│   ├── visualize.py            # calls the original upstream renderer (CPU)
│   ├── select_object.py        # selects the tracks of one object (CPU)
│   ├── trackdata.py            # loading + the fingerprint of a track set
│   └── console.py              # stage reporting, progress bars and rates
├── external/
│   └── trackeverything/    # upstream, Git submodule (read-only, pinned)
├── scripts/                # setup / check / test / inspection helpers
│   ├── setup_env.sh            # install torch, torchvision, deps
│   ├── build_torch_scatter.sh  # build torch_scatter from source (see §4)
│   ├── build_curope.sh         # build the cuRoPE2D CUDA kernel (see §4.1)
│   ├── patch_curope.py         # relax one over-strict croco assertion
│   ├── check_curope.py         # verify the kernel and its numerics
│   ├── bench_curope.py         # CUDA kernel vs pytorch fallback
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
│   ├── test_select_object.py# Select Object node tests + real-data E2E
│   └── fake_worker.py       # stand-in worker used by the unit tests
├── test_data/              # test clips (git-ignored)
├── outputs/                # inference results (git-ignored)
├── docs/                   # reports
│   ├── standalone-test.md
│   ├── vram-diagnostics.md
│   ├── curope.md
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
pixi run bash scripts/build_curope.sh           # the CUDA RoPE kernel, see §4.1
pixi run bash scripts/fetch_dinov2_weights.sh  # see §4 for why

# 3.4 verify the environment
pixi run python scripts/check_env.py
pixi run python scripts/check_curope.py
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

### 4.1 The cuRoPE2D CUDA kernel

Pi3 can use a compiled CUDA implementation of its 2D rotary position
embedding. Without it Pi3 prints

```
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```

`scripts/build_curope.sh` builds the original kernel from
[croco](https://github.com/naver/croco) (`models/curope`, pinned commit
`5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb`) for **sm_86** (NVIDIA A40) and
**sm_89** (RTX 4060 Ti) and installs it in the layout the upstream expects
(`models.curope` as a namespace package). The upstream source is not modified.

Two things are worth knowing:

* croco's own `setup.py` derives the GPU architectures from *torch's* build
  list, which does not contain `sm_89` - so the build uses our own
  arch-explicit `scripts/curope_build/setup.py`.
* one over-strict assertion in croco's kernel rejects Pi3's attention memory
  layout; `scripts/patch_curope.py` relaxes exactly that check and leaves the
  kernel body untouched.

Details, measurements and the fallback behaviour:
[`docs/curope.md`](docs/curope.md).

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
pixi run test-console      # console reporting
pixi run test-curope      # the cuRoPE2D CUDA kernel
pixi run test-select      # the Select Object node
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
* [`docs/curope.md`](docs/curope.md) — the cuRoPE2D CUDA kernel: how it is
  built, what it changes numerically and how it compares to the fallback.
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
| **TrackEverything Save Data** | `track_data`, `save_mode` (`reference`/`copy`), `filename_prefix`, optional `selection` | `TRACK_DATA` | no (`OUTPUT_NODE`) |
| **TrackEverything Select Object** | `track_data`, `masks` (MASK), `mask_threshold` (0.5), `erode_pixels` (2), `min_inside_ratio` (0.6), `min_support_frames` (2) | `TRACK_SELECTION`, `INT`, `STRING` | no |
| **TrackEverything Visualize** | `track_data`, `max_points` (5000), `trail_length` (8), `visibility_threshold` (0.5), `dynamic_threshold` (0.5), `background_opacity` (0.7), optional `images` (IMAGE batch), optional `selection` | `images` — socket type **`IMAGE`** | no |

All nodes exchange file references plus metadata, never tensors (`TRACK_DATA`,
`TRACK_SELECTION`), so a workflow can be rewired freely between them.

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

**Selecting one object** — post-processing of a finished tracking, no new
inference:

```
Video ──▶ TrackEverything Track ──▶ TRACK_DATA ─┐
                                                  ├─▶ TrackEverything Select Object ──▶ TRACK_SELECTION
Video ──▶ SAM3 (or any segmenter) ──▶ MASK ───────┘

TRACK_DATA ────┐
               ├─▶ TrackEverything Visualize ──▶ Preview Image
TRACK_SELECTION ┘
```

The masks must cover exactly the same frames as the tracks. The result is a
**list of stable track ids** — the original track set is not modified and no
model is loaded. `TrackEverything Visualize` draws only those tracks
(`max_points` is applied *after* the selection), and `TrackEverything Save Data`
copies the selection along with the result.

The selection is stored in its own directory under ComfyUI's output directory
(`trackeverything_selection_<timestamp>/selection.json`) and records the
**fingerprint** of the track set it was made for — a selection of a different
inference is rejected instead of silently producing wrong tracks.

**Socket types.** The Visualize node declares its output with ComfyUI's
standard `IMAGE` type (the display name is still `images`). That is what
makes the socket blue and lets it be connected to the `images` input of the
standard nodes — *Create Video*, *Preview Image*, *Save Image*, … ComfyUI
links sockets by comparing the type names exactly, so a custom type name
would render as a grey, unconnectable socket.

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

## 10.1 Console output

Both workers print a stage-by-stage report, and the ComfyUI console streams it
live while the run is happening. The same text is kept in `worker.log` next to
the results.

```
==========================================================================
 TrackEverything - standalone inference
==========================================================================
   device          : NVIDIA GeForce RTX 4060 Ti
   model           : katefgroup/TrackEverything
   geometry        : pi3
   pi3             : yyfz233/Pi3
   frames          : 16 @ 384x256
   mode            : dense, 3 iterations
   voxel size      : 0.02
   mem efficient   : True
   output          : /home/egorkkk/ComfyUI/output/trackeverything_20250101-120000
==========================================================================

[1/4] loading the TrackEverything checkpoint
       demo_hf: Downloading model from HuggingFace Hub: katefgroup/TrackEverything
       done in 2.19 s   520.8 MiB on disk

[2/4] loading the pi3 geometry backend
       done in 6.71 s

[3/4] estimating geometry and tracking points
       demo_hf: Estimating geometry with pi3 (16 frames)...
       -> geometry estimation
       geometry                   elapsed    1.0s    8.11 frames/s
       chunk 1/1   elapsed    1.6s   0.61 chunks/s
       -> tracking (dense)
       tracking                   elapsed    0.6s   13.34 frames/s
       done in 3.01 s   16 frames, 2.65 frames/s, 1 geometry chunk(s)

[4/4] saving tracks.pt and geometry.pt
       done in 0.05 s   7.2 MiB written

==========================================================================
   result          : /home/egorkkk/ComfyUI/output/trackeverything_20250101-120000
   tracks          : 2703
   frames          : 16
   total           : 13.52 s
   peak VRAM       : 6.0 GiB allocated / 6.1 GiB reserved
==========================================================================
```

The geometry phase reports **real** progress: the upstream annotator processes
the video chunk by chunk, and the worker counts the finished chunks (the
upstream source is not modified). The heartbeat line shows the elapsed time
and the current rate in frames/s; on a TTY it is rewritten in place, in a pipe
it is emitted as a normal line every couple of seconds.

The visualization worker reports the same way (loading, preparing the scene,
rendering, writing the frames) and states that it runs on the CPU only.

## 11. Status / known limitations

See `docs/standalone-test.md` for the ТЗ №1 state and
`docs/vram-diagnostics.md` for the memory figures.
