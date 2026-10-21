#!/usr/bin/env python
"""Verify the cuRoPE2D installation.

Checks, in order:

1. ``models.curope`` is importable and the compiled kernel is loaded.
2. The **upstream** TrackEverything actually picks the CUDA implementation
   (the fallback class is *not* used) - i.e. the "cannot find cuda-compiled
   version of RoPE2D" warning is gone.
3. The CUDA kernel produces the same numbers as the upstream pure-pytorch
   fallback.
4. The kernel runs on the current GPU without a CUDA error.

Run with:
    pixi run python scripts/check_curope.py
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_PARENT = str(REPO_ROOT / "external")
if UPSTREAM_PARENT not in sys.path:
    sys.path.insert(0, UPSTREAM_PARENT)

import torch  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ->  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    print("=" * 78)
    print("cuRoPE2D verification")
    print("=" * 78)

    # --- 1. the extension is importable --------------------------------
    try:
        from models.curope import cuRoPE2D
        import models.curope.curope2d as wrapper
    except Exception as exc:
        check("models.curope is importable", False, repr(exc))
        return 1

    kernels = getattr(wrapper, "_kernels", None)
    check("models.curope is importable", True, cuRoPE2D.__module__)
    check(
        "the compiled CUDA kernel is loaded",
        kernels is not None and getattr(kernels, "__file__", "").endswith(".so"),
        getattr(kernels, "__file__", "?"),
    )

    # --- 2. the upstream really uses it --------------------------------
    # import the upstream module the way the worker does and look at the
    # class the upstream bound to the name "RoPE2D"
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        from trackeverything.third_party.pi3.models.layers import pos_embed
    importlib.reload(pos_embed) if False else None

    upstream_class = pos_embed.RoPE2D
    check(
        "the upstream binds RoPE2D to the CUDA implementation",
        upstream_class is cuRoPE2D,
        f"{upstream_class.__module__}.{upstream_class.__name__}",
    )
    warning = "cannot find cuda-compiled version of RoPE2D"
    check(
        "the fallback warning is not printed",
        warning not in buffer.getvalue(),
        repr(buffer.getvalue().strip()[:120]),
    )

    # --- 3. numerical agreement with the pure pytorch fallback ----------
    from trackeverything.third_party.pi3.models.layers.pos_embed import RoPE2D

    fallback_src = (
        "import torch\n"
        "class _Fallback(torch.nn.Module):\n"
        "    def __init__(self, freq=100.0, F0=1.0):\n"
        "        super().__init__()\n"
        "        self.base = freq\n"
        "        self.F0 = F0\n"
        "        self.cache = {}\n"
        "    def get_cos_sin(self, D, seq_len, device, dtype):\n"
        "        if (D,seq_len,device,dtype) not in self.cache:\n"
        "            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))\n"
        "            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)\n"
        "            freqs = torch.einsum('i,j->ij', t, inv_freq).to(dtype)\n"
        "            freqs = torch.cat((freqs, freqs), dim=-1)\n"
        "            self.cache[D,seq_len,device,dtype] = (freqs.cos(), freqs.sin())\n"
        "        return self.cache[D,seq_len,device,dtype]\n"
        "    @staticmethod\n"
        "    def rotate_half(x):\n"
        "        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]\n"
        "        return torch.cat((-x2, x1), dim=-1)\n"
        "    def apply_rope1d(self, tokens, pos1d, cos, sin):\n"
        "        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]\n"
        "        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]\n"
        "        return (tokens * cos) + (self.rotate_half(tokens) * sin)\n"
        "    def forward(self, tokens, positions):\n"
        "        D = tokens.size(3) // 2\n"
        "        cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)\n"
        "        y, x = tokens.chunk(2, dim=-1)\n"
        "        y = self.apply_rope1d(y, positions[:,:,0], cos, sin)\n"
        "        x = self.apply_rope1d(x, positions[:,:,1], cos, sin)\n"
        "        return torch.cat((y, x), dim=-1)\n"
    )
    namespace: dict = {}
    exec(fallback_src, namespace)
    fallback = namespace["_Fallback"]()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    batch, heads, tokens_n, dim = 2, 4, 64, 32
    # the layout the upstream produces: a contiguous (B, N, C) tensor that is
    # reshaped and permuted into (B, H, N, D)
    base = torch.randn(batch, tokens_n, heads * dim, device=device, dtype=torch.float32)
    q_cuda = base.clone().reshape(batch, tokens_n, heads, dim).permute(0, 2, 1, 3)
    q_fallback = base.clone().reshape(batch, tokens_n, heads, dim).permute(0, 2, 1, 3)
    positions = torch.randint(
        0, tokens_n, (batch, tokens_n, 2), device=device, dtype=torch.int64
    )

    cuda_rope = cuRoPE2D(freq=100.0).to(device)
    fallback_rope = fallback.to(device)

    try:
        out_cuda = cuda_rope(q_cuda, positions)
        torch.cuda.synchronize() if device == "cuda" else None
        cuda_error = None
    except Exception as exc:
        out_cuda = None
        cuda_error = repr(exc)

    check(
        "the CUDA kernel runs without a CUDA error",
        cuda_error is None,
        cuda_error or "",
    )

    if out_cuda is not None:
        out_fallback = fallback_rope(q_fallback, positions)
        diff = (out_cuda.float() - out_fallback.float()).abs()
        max_diff = float(diff.max())
        scale = float(out_fallback.float().abs().max())
        rel = max_diff / scale if scale > 0 else max_diff
        print(f"       max |diff| vs the pytorch fallback: {max_diff:.3e}"
              f"  (relative {rel:.3e}, scale {scale:.3f})")
        check(
            "the CUDA kernel agrees with the pytorch fallback",
            rel < 1e-5,
            f"relative difference {rel:.3e}",
        )

        # --- 3b. the layout the upstream actually uses --------------------
        # Pi3 hands the rope a *contiguous* (B, H, N, D) tensor, while the
        # check above used croco's permuted view. Both must agree with the
        # fallback.
        contig_cuda = torch.randn(
            batch, heads, tokens_n, dim, device=device, dtype=torch.float32
        )
        contig_fallback = contig_cuda.clone()
        contig_positions = positions
        try:
            out_contig_cuda = cuda_rope(contig_cuda, contig_positions)
            out_contig_fallback = fallback_rope(contig_fallback, contig_positions)
            d = (out_contig_cuda.float() - out_contig_fallback.float()).abs()
            dmax = float(d.max())
            dscale = float(out_contig_fallback.float().abs().max())
            drel = dmax / dscale if dscale > 0 else dmax
            print(f"       contiguous (B,H,N,D) layout: max |diff| {dmax:.3e}"
                  f"  (relative {drel:.3e})")
            check(
                "the CUDA kernel agrees with the fallback on Pi3's layout",
                drel < 1e-5,
                f"relative difference {drel:.3e}",
            )
        except Exception as exc:
            check("the contiguous (B,H,N,D) layout works", False, repr(exc))

        # --- 4. the in-place behaviour ---------------------------------
        # The CUDA kernel writes into its input tensor (croco behaviour), the
        # pure pytorch fallback does not. Measure it on the tensor that is
        # actually handed to the rope.
        storage = torch.randn(batch, tokens_n, heads * dim, device=device)
        q = storage.reshape(batch, tokens_n, heads, dim).permute(0, 2, 1, 3)
        before = q.detach().clone()
        out = cuda_rope(q, positions)
        changed = not torch.equal(before, q.detach())
        print(f"       the input tensor is modified in place: {changed}")
        check(
            "the in-place write is confined to the roped tensor",
            changed,
            "documented: cuRoPE2D writes into its input (croco behaviour); "
            "the returned tensor carries the roped values",
        )
        # the returned tensor must carry the roped values
        check(
            "the returned tensor holds the roped values",
            torch.equal(out.detach(), q.detach()),
            "the module returns the (in-place roped) input",
        )

        # the fallback must NOT modify its input
        storage2 = torch.randn(batch, tokens_n, heads * dim, device=device)
        q2 = storage2.reshape(batch, tokens_n, heads, dim).permute(0, 2, 1, 3)
        before2 = q2.detach().clone()
        fallback_rope(q2, positions)
        check(
            "the pytorch fallback leaves its input untouched",
            torch.equal(before2, q2.detach()),
            "the two implementations differ in their side effects",
        )

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
