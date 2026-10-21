# cuRoPE2D: the CUDA RoPE kernel of Pi3

Pi3 (the geometry backend of TrackEverything) can use a compiled CUDA
implementation of its 2D rotary position embedding. When it is not available,
Pi3 falls back to a slow pure-pytorch implementation and prints

```
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```

This document describes how the CUDA kernel is built, how it is wired into the
worker, what it changes numerically and what it does **not** change.

---

## 1. How the upstream imports it

`external/trackeverything/third_party/pi3/models/layers/pos_embed.py`:

```python
try:
    from models.curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        ...   # pure pytorch fallback
```

The import is **absolute**: the upstream expects a top-level package `models`
that contains the `curope` sub-package of
[croco](https://github.com/naver/croco) (`models/curope`). The upstream source
is **not modified** - the integration is purely external.

## 2. What is installed

`scripts/build_curope.sh` clones croco at a pinned commit and installs this
layout into the isolated environment:

```
<site-packages>/models/curope/__init__.py     copied from croco
<site-packages>/models/curope/curope2d.py    copied from croco
<site-packages>/models/__init__.py            NOT installed - 'models' stays
                                              a namespace package, exactly as
                                              in croco
<site-packages>/curope...so                   the compiled CUDA extension
```

`models` has no `__init__.py` in croco either, so it is a namespace package and
nothing else in the environment is shadowed. The compiled extension is named
`curope`, which is what croco's `curope2d.py` imports.

| Item | Value |
|---|---|
| croco repository | `https://github.com/naver/croco` |
| pinned commit | `5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb` |
| croco licence | CC BY-NC-SA 4.0 (non-commercial) - only compiled locally, never redistributed |
| built extension | `curope-0.0.0-cp310-cp310-linux_x86_64.whl` |
| build script | `scripts/build_curope.sh` |
| our build file | `scripts/curope_build/setup.py` |
| verification | `scripts/check_curope.py` |
| benchmark | `scripts/bench_curope.py` |

Nothing compiled is committed: the `.so` lives in the git-ignored pixi
environment.

## 3. GPU architectures

Both target GPUs are covered:

| GPU | Compute capability | Flag |
|---|---|---|
| NVIDIA A40 (Ampere) | sm_86 | `-gencode arch=compute_86,code=sm_86` |
| RTX 4060 Ti (Ada Lovelace) | sm_89 | `-gencode arch=compute_89,code=sm_89` |
| newer GPUs | - | `-gencode arch=compute_89,code=compute_89` (PTX) |

croco's own `setup.py` derives the flags from
`torch.cuda.get_gencode_flags()`, which lists the architectures *torch itself*
was compiled for: `sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120,
compute_120`. That list **does not contain sm_89**, so the stock build does not
run on an RTX 4060 Ti. `scripts/curope_build/setup.py` therefore replaces
croco's `setup.py` with an explicit architecture list. The croco *sources* are
not touched.

## 4. The one documented deviation from the croco sources

croco's CUDA kernel starts with:

```cpp
TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D, "tokens are not contiguous");
```

The kernel receives the tokens as `(B, N, H, D)` and requires the **head**
stride to equal `D` - the memory layout of croco's own attention, where a
head's `D` values are adjacent.

Pi3 keeps the attention tensor contiguous in `(B, H, N, D)`, so after the
`transpose(1, 2)` the head stride is `N * D` and the assertion fires:

```
RuntimeError: tokens are not contiguous
```

The kernel *body* is layout agnostic: it reads and writes through
`torch::PackedTensorAccessor32`, which resolves the element offsets from the
strides. Only the assertion is too strict, so `scripts/patch_curope.py`
relaxes exactly that one check:

```cpp
TORCH_CHECK(tokens.stride(3) == 1, "the last dimension must be contiguous");
```

The patch is applied with an exact-match replacement and the build **fails
loudly** if the expected source text is not found, so a change in croco can
never silently produce a broken extension. The numerical algorithm, the
kernels and the python wrapper are untouched.

## 5. Numerical agreement

`scripts/check_curope.py` compares the CUDA kernel against the upstream
pure-pytorch fallback on random data:

| Layout | max abs difference | relative |
|---|---|---|
| croco's permuted view `(B, H, N, D)` | 2.38e-07 | 5.45e-08 |
| Pi3's contiguous `(B, H, N, D)` | 2.38e-07 | 5.71e-08 |

That is the float32 rounding level (~1 ULP): the two implementations compute
the same formula.

### `--use_fast_math` is deliberately not used

The first build used croco's `-O3 --use_fast_math`. `--use_fast_math` maps
`sinf`/`cosf` onto the fast intrinsic approximations, which shifted the rope
values by ~1e-5 (100x more than the precise math) and, amplified through the
Pi3 transformer, changed the resulting geometry noticeably. The kernel is
memory bound anyway, so the precise math costs nothing - the build flags are
`nvcc=["-O3"]`.

## 6. The in-place behaviour

croco's kernel writes into its input tensor:

```python
def forward(self, tokens, positions):
    cuRoPE2D_func.apply(tokens.transpose(1, 2), positions, self.base, self.F0)
    return tokens
```

`transpose` returns a view, so the kernel writes straight into the caller's
storage. The pure-pytorch fallback does not (it returns a new tensor).

**Does this change the result?** No. In Pi3's attention the rope is applied to
the output of `q_norm` / `k_norm`:

```python
q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
q = self.rope(q, qpos)
k = self.rope(k, kpos)
```

`q_norm`/`k_norm` produce fresh tensors, so the in-place write only touches
those temporaries - no tensor that is used elsewhere is modified. The module
returns the same tensor it was given, which now carries the roped values
(`scripts/check_curope.py` asserts both facts).

The side-effect difference is nevertheless documented and tested
(`tests/test_curope.py`): the CUDA kernel modifies its input, the fallback
does not.

## 7. Benchmark: CUDA kernel vs fallback

`scripts/bench_curope.py` runs the **same** upstream pipeline twice - once
with the compiled kernel and once with the fallback (forced through
`TRACKEVERYTHING_FORCE_ROPE_FALLBACK=1`, which only hides `models.curope`
from the import system; the upstream source is untouched). A warm-up pair runs
first, so the cold page cache of the 3.8 GB Pi3 weights does not bias the
measurement.

8 frames, RTX 4060 Ti, torch 2.8.0+cu129:

| Metric | CUDA kernel | fallback | speedup |
|---|---|---|---|
| geometry estimation (`inference_seconds`) | 2.97 s | 3.02 s | **1.02x** |
| total worker time | 13.54 s | 13.43 s | 0.99x |
| peak VRAM | 6112.16 MiB | 6112.23 MiB | 1.00x |
| model load | 1.99 s | 1.92 s | - |
| Pi3 load | 6.99 s | 6.90 s | - |

**The CUDA RoPE is not a meaningful speedup for Pi3** - the rope is a tiny
fraction of Pi3's compute (the geometry stage is dominated by the DINOv2
backbone and the transformer), and the fallback's vectorised pytorch op is
already efficient. The kernel is not slower either, so there is no reason to
prefer the fallback.

### Objective quality: is the result still good?

A numerical comparison is not conclusive here, because a 5e-08 difference in
the rope is amplified by the Pi3 transformer into a ~2.5e-03 difference in the
geometry - and the tracker's voxelisation is a *discrete* operation, so a
handful of points cross a voxel border.

The objective measure is the **multi-view photometric consistency** of the
estimated geometry: the depth maps are projected into the next frame with the
estimated cameras and compared with the real pixels (lower is better).

| Variant | Photometric error | Tracks |
|---|---|---|
| CUDA kernel | **0.141504** | 2409 |
| fallback | 0.141880 | 2415 |
| difference | **-0.27 %** | - |

The CUDA variant is marginally **better**, so the geometry quality is
unchanged. The 6 tracks that differ are the result of the voxel-boundary
flips, not of a degraded geometry.

### Run-to-run determinism

Two runs of the CUDA variant:

* geometry (`depths`, `intrinsics`, `extrinsics`): **bit-identical**,
* `src_time` / `src_x` / `src_y`: **bit-identical**,
* `cell_to_token`: **5 of 196 608 cells differ (0.003 %)** - points that sit
  exactly on a voxel border and flip because of the tracker's own
  non-deterministic CUDA reductions.

The fallback variant is bit-identical in `cell_to_token` as well; the CUDA
path slightly amplifies the tracker's non-determinism at voxel borders.

## 8. Fallback

If the extension is missing or fails to build, Pi3 uses its pure-pytorch
fallback and prints the original warning. Nothing breaks - the whole pipeline
keeps working, just without the CUDA rope.

`scripts/setup_env.sh` treats a failed build as non-fatal:

```
==> [4/4] the CUDA RoPE2D kernel (croco's cuRoPE2D)
    !! the cuRoPE2D kernel could not be built.
    !! Pi3 will use its pure-pytorch RoPE fallback (slower).
```

`scripts/check_curope.py` reports what the upstream actually bound:

```
[PASS] the upstream binds RoPE2D to the CUDA implementation
[PASS] the fallback warning is not printed
```

## 9. Reproducing

```bash
cd ~/trackeverything

# build + install the kernel (also part of scripts/setup_env.sh)
pixi run bash scripts/build_curope.sh

# verify the installation and the numerics
pixi run python scripts/check_curope.py

# benchmark against the fallback
pixi run python scripts/bench_curope.py --frames-dir outputs/_frames_test --max-frames 8

# the tests
pixi run python tests/test_curope.py
```
