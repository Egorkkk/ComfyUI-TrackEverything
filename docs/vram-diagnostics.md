# CUDA memory diagnostics (ТЗ №2, Stage 1)

Produced by `scripts/diagnose_vram.py`, which runs the pipeline **in-process**
so the torch caching-allocator counters are exact.

```bash
pixi run python scripts/diagnose_vram.py --max_frames 16 --out docs/vram_diagnostics/16f.json
```

## Headline numbers

The previously reported "1156 MiB" was **not** our process's memory: it is the
*baseline device usage* caused by the production ComfyUI process that was
already running on the GPU. The real footprint of a TrackEverything run is
**~6.5 GiB allocated / ~6.9 GiB reserved** on top of that.

Machine: RTX 4060 Ti, 16 379 MiB total, driver 610.88, torch 2.8.0+cu129.

| Run | Stage | max_memory_allocated | max_memory_reserved | peak active | segments |
|---|---|---|---|---|---|
| 16 frames | 1. load TrackEverything | 228.94 MiB | 254.00 MiB | 228.94 MiB | 28 |
| 16 frames | 2. load Pi3 geometry | 3 886.47 MiB | 3 918.00 MiB | 3 886.47 MiB | 262 |
| 16 frames | 3. inference (dense) | **6 518.96 MiB** | **6 938.00 MiB** | 6 518.96 MiB | 282 |
| 16 frames | 4. save outputs | 3 982.38 MiB | 6 938.00 MiB | 3 982.38 MiB | 269 |

| Run | max alloc (inference) | max reserved | device peak used | our delta over baseline |
|---|---|---|---|---|
| 4 frames | 5 910.15 MiB | 6 108.00 MiB | 7 338.38 MiB | 6 182 MiB |
| 8 frames | 6 112.23 MiB | 6 274.00 MiB | 7 504.38 MiB | 6 348 MiB |
| 16 frames | 6 518.96 MiB | 6 938.00 MiB | 8 170.38 MiB | 7 014 MiB |
| 16 frames, pure-noise input (worst case) | 7 707.87 MiB | 8 958.00 MiB | 10 190.38 MiB | 9 034 MiB |

Device baseline (the other ComfyUI process): 1 156.38 MiB.

## Where the memory goes

* **The Pi3 geometry backend dominates: 3.89 GiB.** That is essentially the
  model *weights* (`model.safetensors` is 3.83 GB) — the geometry model is the
  single biggest VRAM consumer, not the tracker.
* **The tracker checkpoint is small: 229 MiB.**
* **Inference activations add ~2.6 GiB** on top of the loaded models.
* Memory grows only mildly with the frame count (4 → 16 frames: +0.6 GiB),
  because `--mem_efficient` keeps per-window snapshots instead of the full
  dense tensor. The *track count* is what grows with scene complexity, not the
  frame count: the noise input produced 11 207 tracks instead of 2 703 and
  pushed the peak to 7.71 GiB allocated / 8.96 GiB reserved.

## NVML per-process figures

`pynvml.nvmlDeviceGetComputeRunningProcesses()` returns
`usedGpuMemory = None` for every process under WSL2 — the platform does not
expose per-process GPU memory attribution. The reports therefore contain:

```json
"nvml": {
  "available": true,
  "per_process_memory_supported": false,
  "note": "WSL2 does not expose per-process GPU memory; usedGpuMemory is None for every process."
}
```

The authoritative per-process numbers are the torch allocator counters
(`max_memory_allocated` / `max_memory_reserved`), which are exact and
process-local.

## Consequence for the ComfyUI integration

The worker process needs roughly **7 GiB of VRAM headroom** on top of whatever
ComfyUI itself is using. On this 16 GiB card, ComfyUI's own ~1.2 GiB leaves
enough room. The worker is a **separate process**, so its CUDA context and
memory are released completely when it exits — ComfyUI's own models are never
evicted.
