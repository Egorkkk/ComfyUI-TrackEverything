# Installing the TrackEverything nodes into ComfyUI

This guide is written for a **separate** ComfyUI user (in our case
`egorkkk`) who wants to install the TrackEverything nodes into their own
ComfyUI instance.

The installation is **fully isolated from the ComfyUI Python environment**:

* the ComfyUI process never imports `trackeverything`, `torch_scatter` or any
  package of the worker's Python 3.10 environment,
* the model runs in a **separate process** with its own CUDA context, so its
  VRAM is released completely when a run finishes,
* nothing is installed into, or removed from, ComfyUI's Python environment.

**No step below touches the ComfyUI installation.** The nodes live in
`custom_nodes/`, and removing them is a single `rm -rf`.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| A CUDA GPU | ~7 GiB of free VRAM per inference run (see `docs/vram-diagnostics.md`) |
| `git`, `curl`, `gcc`/`g++` | to build `torch_scatter` |
| ~25 GB free disk | torch + CUDA toolkit + model weights |
| A Hugging Face account | the tracker checkpoint is a **gated** repo |
| pixi | the isolated environment manager (optional, see §2) |

---

## 1. Get the project and its submodule

```bash
mkdir -p ~/$USER/trackeverything
cd ~/$USER/trackeverything

git clone <THIS_REPOSITORY> .
cd trackeverything

# the upstream TrackEverything lives in external/trackeverything
git submodule update --init --recursive

# verify the pinned upstream commit
git -C external/trackeverything log --oneline -1
# expected: f0c18dd code release
```

> If you received the project as an archive instead of a git clone, the
> submodule directory will be empty. In that case clone the upstream
> repository manually into it:
>
> ```bash
> git clone https://github.com/ayushjain1144/trackeverything.git external/trackeverything
> git -C external/trackeverything checkout f0c18dd9714b9f85825c45a1c3aae685da4e6222
> ```

## 2. Create your own isolated environment

Use **your own** account - do not reuse anybody else's environment and do not
install anything into the ComfyUI environment.

### 2.1 Install pixi (optional)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
# add ~/.pixi/bin to your PATH
```

Any other environment manager works as well; the requirements are simply
Python 3.10 with torch 2.8.0+cu129 (see §2.3).

### 2.2 Create the environment

The repository already contains `pixi.toml` and `pixi.lock`:

```bash
cd ~/$USER/trackeverything
pixi install
```

This creates `~/$USER/trackeverything/.pixi/envs/default` with **Python
3.10.22** and the CUDA 12.9 compiler. It is completely separate from
ComfyUI's environment.

### 2.3 Install the Python dependencies

```bash
cd ~/$USER/trackeverything

# torch 2.8.0+cu129 / torchvision 0.23.0+cu129 / the pinned upstream deps
pixi run bash scripts/setup_env.sh

# torch_scatter (see the note below)
pixi run bash scripts/build_torch_scatter.sh

# DINOv2 backbone (see the note below)
pixi run bash scripts/fetch_dinov2_weights.sh
```

> **Why are two of these built/fetched by a script?**
>
> * `torch_scatter` — the upstream README installs a prebuilt wheel from
>   `data.pyg.org`. On some networks that host is throttled to ~16 KiB per
>   connection and the download cannot finish. The script compiles the
>   **same upstream sources** (sdist 2.1.2 from PyPI) against the same torch
>   build with `nvcc 12.9`. If `data.pyg.org` is fast for you, the canonical
>   command works too:
>   `pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html`
> * the DINOv2 ViT-S/14 backbone — upstream pulls it from
>   `dl.fbaipublicfiles.com`, which is throttled on some networks. The script
>   fetches the identical file from a Hugging Face mirror and verifies it
>   against the upstream host.

### 2.4 Verify the environment

```bash
pixi run python scripts/check_env.py
```

All checks must pass: Python 3.10, torch 2.8.0+cu129, `cuda.is_available()`,
the GPU name, `torch_scatter` and the `trackeverything.demo_hf` import.

## 3. Hugging Face access and the models

Two checkpoints are needed:

| Model | Repo | Size | Access |
|---|---|---|---|
| Tracker | `katefgroup/TrackEverything` | 546 MB | **gated** — accept the licence |
| Geometry (Pi3) | `yyfz233/Pi3` | 3.83 GB | public |

### 3.1 Accept the licence for the tracker

Open <https://huggingface.co/katefgroup/TrackEverything>, log in and accept
the terms. Until that is done, downloads fail with
`401 GatedRepo`.

### 3.2 Create your **own** access token

Do **not** copy somebody else's token. Create your own at
<https://huggingface.co/settings/tokens> (a *read* token is enough).

Log in with the standard Hugging Face CLI — the token is then stored in your
own user profile and is never part of this project:

```bash
pixi run python -c "from huggingface_hub import login; login()"
# paste your token when asked
```

Alternatively export it for the session only:

```bash
export HF_TOKEN=hf_...          # your own token
```

> The token is only ever read from the environment / the Hugging Face
> credential store. It is never written into the project, never passed on a
> command line and never printed into a report.

### 3.3 Pre-fetch the models (optional)

```bash
pixi run python -c "
from huggingface_hub import snapshot_download
snapshot_download('katefgroup/TrackEverything')
snapshot_download('yyfz233/Pi3')
"
```

They are cached in `~/.cache/huggingface` and reused on every later run — they
are never re-downloaded.

## 4. Install the nodes into ComfyUI

```bash
# symlink (preferred - the project stays where it is and can be updated)
ln -s ~/$USER/trackeverything/comfy_nodes \
      /home/egorkkk/ComfyUI/custom_nodes/comfy_nodes_trackeverything

# ... or copy the directory
# cp -r ~/$USER/trackeverything/comfy_nodes \
#       /home/egorkkk/ComfyUI/custom_nodes/comfy_nodes_trackeverything
```

Only the `comfy_nodes/` directory is needed - it contains no binary weights
and no Python dependencies beyond what ComfyUI already has (numpy, Pillow,
torch).

Check that ComfyUI sees them:

```bash
ls -l /home/egorkkk/ComfyUI/custom_nodes/
```

The nodes appear in the node menu under the category **TrackEverything**.

## 5. Point the nodes at your worker environment

The nodes must know three things: where your project is, which python runs
the worker, and where the results go.

### 5.1 Option A - environment variables (recommended)

Set them in the shell that starts ComfyUI:

```bash
export TRACKEVERYTHING_PROJECT_ROOT=/home/egorkkk/trackeverything
export TRACKEVERYTHING_WORKER_PYTHON=/home/egorkkk/trackeverything/.pixi/envs/default/bin/python
# optional, defaults to ComfyUI's own output directory:
# export TRACKEVERYTHING_OUTPUT_ROOT=/home/egorkkk/ComfyUI/output
```

### 5.2 Option B - `config.json`

```bash
cp ~/$USER/trackeverything/config.json.example ~/$USER/trackeverything/config.json
$EDITOR ~/$USER/trackeverything/config.json
```

```json
{
  "worker_python": "/home/egorkkk/trackeverything/.pixi/envs/default/bin/python",
  "output_root": "/home/egorkkk/ComfyUI/output",
  "device": "cuda",
  "geometry_backend": "pi3"
}
```

Environment variables win over `config.json`.

### 5.3 Check the configuration

```bash
cd ~/$USER/trackeverything
pixi run python -c "
from comfy_nodes.config import Config
cfg = Config.resolve(); cfg.validate()
print(cfg.describe())
"
```

If anything is missing you get an explicit message naming the variable to
set.

## 6. Start ComfyUI as usual

```bash
cd /home/egorkkk/ComfyUI
python main.py          # or whatever start command you normally use
```

Nothing about the start command changes. The worker is started on demand by
the node, not at ComfyUI startup.

Watch the terminal: when a node runs, the worker's log is streamed to the
ComfyUI console and also saved as `worker.log` next to the results.

## 7. A minimal test workflow

1. Load a video (e.g. the *Load Video* / *VHS* node) or any node that
   outputs an `IMAGE` batch.
2. Add **TrackEverything Track**.
3. Connect the `IMAGE` output to the node's `images` input.
4. Leave the defaults: `num_iters = 3`, `voxel_size = 0.02`,
   `mem_efficient = true`.
5. Queue the workflow.
6. The node returns a `TRACK_DATA` output.

To re-use a previous run without running the model again, add
**TrackEverything Load Data** and paste the result directory
(for example `/home/egorkkk/ComfyUI/output/trackeverything_20250101-120000`).

## 8. Check the result

```bash
cd /home/egorkkk/trackeverything
ls -l /home/egorkkk/ComfyUI/output/trackeverything_*/
```

```
tracks.pt          scene tokens (coords, visibs, dyn_logits, src_*, cell_to_token)
geometry.pt        depths, intrinsics, extrinsics, norm_factor
metadata.json      parameters, versions, resolutions, coordinate transform
worker_report.json runtime facts (versions, timings, peak CUDA memory)
worker.log         the full worker log
```

Validate them with the upstream checker:

```bash
pixi run python scripts/inspect_results.py \
    /home/egorkkk/ComfyUI/output/trackeverything_20250101-120000
```

All checks must pass.

### Reading the coordinates

`metadata.json` → `coordinate_transform` describes exactly how the numbers
relate to your pixels:

* the model runs at a fixed resolution (`working_size`, e.g. 512×384) that is
  **not** your input resolution - use `scale_x` / `scale_y` to map back,
* `src_x` / `src_y` are **cell indices** of the feature grid
  (`feature_grid`, patch size 4), not pixels,
* the 3D coordinates live in the upstream scene frame (camera-0 aligned and
  normalised): multiply by `norm_factor` to get back to the original depth
  units,
* `-1000` marks a missing / not-yet-existing trajectory - it is **not** a
  coordinate.

## 9. Removing the nodes again

```bash
# symlink
rm /home/egorkkk/ComfyUI/custom_nodes/comfy_nodes_trackeverything

# copy
rm -rf /home/egorkkk/ComfyUI/custom_nodes/comfy_nodes_trackeverything
```

Restart ComfyUI. Nothing else was modified: no ComfyUI file was touched, no
ComfyUI package was installed, and the model weights stay in
`~/.cache/huggingface`.

To remove the weights as well:

```bash
rm -rf ~/.cache/huggingface/hub/models--katefgroup--TrackEverything
rm -rf ~/.cache/huggingface/hub/models--yyfz233--Pi3
```

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `TrackEverything nodes are not configured: ... worker python ... does not exist` | set `TRACKEVERYTHING_WORKER_PYTHON` (see §5) |
| `the worker environment's site-packages is on sys.path` | the isolation guard fired: the ComfyUI process must not have the worker's `site-packages` on `sys.path` |
| `The TrackEverything worker failed (exit code 1)` | read `worker.log` in the result directory - the worker's traceback is copied into the error message |
| `HTTP 401 / GatedRepo` | accept the licence for `katefgroup/TrackEverything` and log in with **your own** token |
| `CUDA out of memory` | the run needs ~7 GiB of free VRAM; close other GPU jobs or use a longer/shorter clip |
| `No frames left after start_frame=...` | the IMAGE batch was empty |
| results are not bit-identical between runs | expected: the upstream kernels reduce with atomics, so float outputs differ by ~1e-7 relative. Compare with a tolerance. |
