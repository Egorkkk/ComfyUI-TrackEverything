#!/usr/bin/env python
"""Stage B - environment verification for the isolated TrackEverything setup.

Checks (Stage B3 of the TZ):
  * Python version
  * PyTorch import, CUDA availability, device name, VRAM
  * torch_scatter import
  * a trivial GPU operation
  * `from trackeverything.demo_hf import TrackEverything`

Also prints isolation diagnostics so we can prove that the production ComfyUI
installation (/home/egorkkk/ComfyUI) is not part of this environment.

Run with:  pixi run python scripts/check_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_PARENT = str(REPO_ROOT / "external")

# The upstream repository *is* the `trackeverything` package, so its PARENT
# directory has to be importable. The upstream README prescribes exporting
# PYTHONPATH; we additionally self-bootstrap here so the script can be called
# directly (e.g. `pixi run python scripts/check_env.py`).
if UPSTREAM_PARENT not in sys.path:
    sys.path.insert(0, UPSTREAM_PARENT)
    os.environ["PYTHONPATH"] = (
        UPSTREAM_PARENT + os.pathsep + os.environ.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {label}" + (f"  ->  {detail}" if detail else ""))


def section(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 58 - len(title)))


# ---------------------------------------------------------------------------
section("Python")
# ---------------------------------------------------------------------------
print(f"executable      : {sys.executable}")
print(f"version         : {sys.version.replace(chr(10), ' ')}")
check(
    "python is 3.10.x",
    sys.version_info[:2] == (3, 10),
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
)

# ---------------------------------------------------------------------------
section("Isolation")
# ---------------------------------------------------------------------------
print(f"PYTHONPATH      : {os.environ.get('PYTHONPATH', '<unset>')}")

# The environment must not borrow anything from a ComfyUI installation.
# NOTE: the check looks for a path *component* named "ComfyUI" - a user
# account name (for example "egorkkk") is perfectly legitimate inside a path
# and must never be treated as an isolation violation.
comfy_paths = []
for p in sys.path:
    if not p:
        continue
    parts = Path(p).parts
    if any(part.lower() == "comfyui" for part in parts):
        comfy_paths.append(p)

for p in comfy_paths:
    check(f"no ComfyUI installation on sys.path: {p}", False)
if not comfy_paths:
    check("no ComfyUI installation on sys.path", True)

site = [p for p in sys.path if p.endswith("site-packages")]
for p in site:
    print(f"  site-packages : {p}")

# ---------------------------------------------------------------------------
section("PyTorch / CUDA")
# ---------------------------------------------------------------------------
try:
    import torch

    print(f"torch version   : {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"cudnn version   : {torch.backends.cudnn.version()}")
    check("torch imported", True, torch.__version__)
    check(
        "torch built for CUDA 12.9",
        torch.version.cuda == "12.9",
        str(torch.version.cuda),
    )
except Exception as exc:  # pragma: no cover
    check("torch imported", False, repr(exc))
    raise SystemExit(1)

cuda_available = torch.cuda.is_available()
check("torch.cuda.is_available()", cuda_available)

if cuda_available:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / 1024**3
    print(f"device index    : {idx}")
    print(f"device name     : {props.name}")
    print(f"capability      : sm_{props.major}{props.minor}")
    print(f"total VRAM      : {total_gb:.2f} GiB")
    check("GPU present", True, props.name)

    free_b, total_b = torch.cuda.mem_get_info()
    print(f"free VRAM now   : {free_b / 1024**3:.2f} GiB of {total_b / 1024**3:.2f} GiB")

    # Trivial GPU operation.
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    check(
        "GPU matmul works",
        bool(torch.isfinite(c).all().item()),
        f"result mean={c.mean().item():.6f}",
    )

    # A torch_scatter op on GPU would be an even stronger check; done below.
else:
    check("GPU present", False)

# ---------------------------------------------------------------------------
section("torch_scatter")
# ---------------------------------------------------------------------------
try:
    import torch_scatter

    ver = getattr(torch_scatter, "__version__", "unknown")
    print(f"torch_scatter   : {ver}")
    check("torch_scatter imported", True, str(ver))

    if cuda_available:
        src = torch.randn(64, 16, device="cuda")
        index = torch.randint(0, 8, (64,), device="cuda")
        out = torch_scatter.scatter_sum(src, index, dim=0)
        torch.cuda.synchronize()
        check(
            "torch_scatter GPU op works",
            tuple(out.shape) == (8, 16),
            f"scatter_sum -> {tuple(out.shape)}",
        )
except Exception as exc:
    check("torch_scatter imported", False, repr(exc))

# ---------------------------------------------------------------------------
section("TrackEverything package")
# ---------------------------------------------------------------------------
parent = UPSTREAM_PARENT
in_path = parent in sys.path or os.environ.get("PYTHONPATH", "").find(parent) >= 0
print(f"expected parent on sys.path: {parent}")
print(f"self-bootstrapped          : {in_path}")
try:
    from trackeverything.demo_hf import TrackEverything

    mod = sys.modules["trackeverything"]
    check(
        "import trackeverything.demo_hf",
        True,
        getattr(mod, "__file__", "?"),
    )
    check("TrackEverything.from_pretrained available", hasattr(TrackEverything, "from_pretrained"))
except Exception as exc:
    check("import trackeverything.demo_hf", False, repr(exc))

# ---------------------------------------------------------------------------
section("Result")
# ---------------------------------------------------------------------------
print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
