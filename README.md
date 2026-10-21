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
├── external/
│   └── trackeverything/    # upstream, Git submodule (read-only, pinned)
├── scripts/                # setup / check / test / inspection helpers
│   ├── setup_env.sh            # install torch, torchvision, deps
│   ├── build_torch_scatter.sh  # build torch_scatter from source (see §4)
│   ├── check_env.py            # Stage B3 environment verification
│   ├── make_test_video.py      # Stage D1 synthetic test clip
│   └── inspect_results.py      # Stage E tracks.pt / geometry.pt validation
├── tests/                  # additional checks
├── test_data/              # test clips (git-ignored)
├── outputs/                # inference results (git-ignored)
├── docs/                   # reports
│   └── standalone-test.md
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

## 4. `torch_scatter` — why it is built from source

The upstream README installs the prebuilt wheel:

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

From this machine `data.pyg.org` is throttled to roughly **16 KiB per
connection** and then stalls (measured ~0.8 KB/s, while `download.pytorch.org`
reaches ~40 MB/s from the same host). No mirror of the PyG wheel index carries
`torch_scatter`, and PyPI only ships an sdist for `torch-scatter` 2.1.2.

`scripts/build_torch_scatter.sh` therefore compiles **the same upstream
version (2.1.2, unmodified sdist from PyPI)** against **the same Torch build
(2.8.0+cu129)** using `nvcc 12.9` from the pixi environment. The result is
numerically equivalent to the official wheel — it is the same C++/CUDA source.

If `data.pyg.org` becomes reachable, the canonical command works unchanged and
`scripts/setup_env.sh` (step 2) will use it.

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

# 6.2 run the ORIGINAL upstream CLI
cd ~/trackeverything
PYTHONPATH="$PWD/external" pixi run python -m trackeverything.demo_hf \
  --video test_data/synthetic_16f.mp4 \
  --output_dir outputs/standalone \
  --geometry_backend pi3 \
  --mode dense \
  --num_iters 3 \
  --voxel_size 0.02 \
  --mem_efficient \
  --max_frames 16
```

Notes:

* `PYTHONPATH` must contain `external/` — the upstream repo *is* the
  `trackeverything` package, so its **parent** directory has to be importable.
* `--geometry_backend pi3` is used deliberately. The upstream default is
  `vggt_omega`, whose checkpoint path is hard-coded to the authors' private
  infrastructure (`/data/group_data/katefgroup/...`).
* `--save_dense` is **not** enabled on the first run: dense trajectories grow
  quadratically with the sequence length and easily exhaust VRAM.

Outputs written to `outputs/standalone/`:

| File | Content |
|---|---|
| `tracks.pt` | `coords`, `visibs`, `dyn_logits`, `src_time`, `src_y`, `src_x`, `cell_to_token` |
| `geometry.pt` | `depths`, `intrinsics`, `extrinsics`, `norm_factor` |

## 7. Inspect the results

```bash
pixi run python scripts/inspect_results.py outputs/standalone
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
  parameters, timings, VRAM, results and known limitations.

## 10. Status / known limitations

See `docs/standalone-test.md` for the current state, including which acceptance
criteria of ТЗ №1 are met and which are outstanding.
