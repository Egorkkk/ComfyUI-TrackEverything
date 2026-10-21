#!/usr/bin/env python
"""Tests for the cuRoPE2D (croco) CUDA integration.

They cover

* the installation: the compiled kernel is importable and the upstream binds
  the CUDA implementation instead of the pure-pytorch fallback,
* the numerical agreement between the CUDA kernel and the fallback, on both
  the croco layout and the layout Pi3 actually uses,
* the in-place behaviour of the kernel (documented, and confined to the
  tensor that is handed to the rope),
* the ``TRACKEVERYTHING_FORCE_ROPE_FALLBACK`` switch,
* the build/patch helpers.

The tests that need the compiled extension are **skipped** when it has not
been built yet (``pixi run bash scripts/build_curope.sh``).

Run with:
    pixi run python tests/test_curope.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "external") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "external"))

import torch  # noqa: E402

FALLBACK_SOURCE = """
import torch


class _Fallback(torch.nn.Module):
    \"\"\"The upstream pure-pytorch RoPE (same formula as in pos_embed.py).\"\"\"

    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            self.cache[D, seq_len, device, dtype] = (freqs.cos(), freqs.sin())
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)
"""


def fallback_class():
    namespace: dict = {}
    exec(FALLBACK_SOURCE, namespace)
    return namespace["_Fallback"]()


def extension_available() -> bool:
    try:
        import models.curope  # noqa: F401
        import models.curope.curope2d as wrapper

        return getattr(wrapper, "_kernels", None) is not None
    except Exception:
        return False


requires_extension = unittest.skipUnless(
    extension_available(),
    "the cuRoPE2D extension is not built - run scripts/build_curope.sh",
)


class TestInstallation(unittest.TestCase):
    def test_the_extension_is_importable(self):
        if not extension_available():
            self.skipTest("the extension is not built")
        from models.curope import cuRoPE2D

        self.assertTrue(hasattr(cuRoPE2D, "forward"))

    def test_the_compiled_kernel_is_used(self):
        if not extension_available():
            self.skipTest("the extension is not built")
        import models.curope.curope2d as wrapper

        self.assertTrue(
            getattr(wrapper, "_kernels", None).__file__.endswith(".so"),
            "the pure python path was taken instead of the compiled kernel",
        )

    def test_the_upstream_binds_the_cuda_implementation(self):
        if not extension_available():
            self.skipTest("the extension is not built")
        from models.curope import cuRoPE2D

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            from trackeverything.third_party.pi3.models.layers import pos_embed

        self.assertIs(pos_embed.RoPE2D, cuRoPE2D)
        self.assertNotIn(
            "cannot find cuda-compiled version of RoPE2D", buffer.getvalue()
        )

    def test_the_upstream_source_is_not_modified(self):
        """The upstream must stay pristine - the integration is external."""
        upstream = PROJECT_ROOT / "external" / "trackeverything"
        proc = subprocess.run(
            ["git", "-C", str(upstream), "status", "--porcelain"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.stdout.strip(), "",
                         "the upstream submodule has local modifications")

    def test_no_compiled_binary_is_committed(self):
        """The .so lives in the (git-ignored) environment, never in git."""
        proc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "ls-files"],
            capture_output=True, text=True, timeout=60,
        )
        tracked = proc.stdout.split()
        for name in tracked:
            self.assertFalse(
                name.endswith((".so", ".pyd", ".dylib")) and not name.endswith(".py"),
                f"a compiled binary is tracked: {name}",
            )


class TestNumerics(unittest.TestCase):
    def _compare(self, make_tokens):
        if not extension_available():
            self.skipTest("the extension is not built")
        from models.curope import cuRoPE2D

        fallback = fallback_class()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch, heads, tokens_n, dim = 2, 4, 64, 32
        positions = torch.randint(
            0, tokens_n, (batch, tokens_n, 2), device=device, dtype=torch.int64
        )

        torch.manual_seed(0)
        cuda_in = make_tokens(batch, heads, tokens_n, dim, device)
        fallback_in = cuda_in.clone()
        out_cuda = cuRoPE2D(freq=100.0).to(device)(cuda_in, positions)
        out_fb = fallback.to(device)(fallback_in, positions)

        diff = (out_cuda.float() - out_fb.float()).abs().max().item()
        scale = out_fb.float().abs().max().item()
        rel = diff / scale if scale > 0 else diff
        return diff, rel

    def test_croco_layout_matches_the_fallback(self):
        def make(batch, heads, tokens_n, dim, device):
            base = torch.randn(batch, tokens_n, heads * dim, device=device)
            return base.reshape(batch, tokens_n, heads, dim).permute(0, 2, 1, 3)

        diff, rel = self._compare(make)
        self.assertLess(rel, 1e-5, f"relative difference {rel:.3e} (max {diff:.3e})")

    def test_pi3_layout_matches_the_fallback(self):
        """Pi3 hands the rope a contiguous (B, H, N, D) tensor."""

        def make(batch, heads, tokens_n, dim, device):
            return torch.randn(batch, heads, tokens_n, dim, device=device)

        diff, rel = self._compare(make)
        self.assertLess(rel, 1e-5, f"relative difference {rel:.3e} (max {diff:.3e})")

    def test_the_kernel_is_close_to_one_ulp(self):
        """Without --use_fast_math the kernel matches the fallback closely."""
        def make(batch, heads, tokens_n, dim, device):
            base = torch.randn(batch, tokens_n, heads * dim, device=device)
            return base.reshape(batch, tokens_n, heads, dim).permute(0, 2, 1, 3)

        _, rel = self._compare(make)
        self.assertLess(rel, 1e-6, f"relative difference {rel:.3e} is too large")

    @requires_extension
    def test_the_input_is_modified_in_place(self):
        """croco's kernel writes into its input - documented behaviour."""
        from models.curope import cuRoPE2D

        device = "cuda" if torch.cuda.is_available() else "cpu"
        storage = torch.randn(1, 4, 8, 16, device=device)
        q = storage.reshape(1, 8, 4, 16).permute(0, 2, 1, 3)
        before = q.detach().clone()
        # non-zero positions, otherwise the rope is the identity
        positions = torch.randint(1, 8, (1, 8, 2), dtype=torch.int64, device=device)
        out = cuRoPE2D(freq=100.0).to(device)(q, positions)
        self.assertFalse(torch.equal(before, q.detach()),
                         "the kernel is expected to write into its input")
        self.assertTrue(torch.equal(out.detach(), q.detach()),
                        "the returned tensor must carry the roped values")

    @requires_extension
    def test_the_fallback_leaves_its_input_untouched(self):
        fallback = fallback_class()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        storage = torch.randn(1, 4, 8, 16, device=device)
        q = storage.reshape(1, 8, 4, 16).permute(0, 2, 1, 3)
        before = q.detach().clone()
        positions = torch.randint(1, 8, (1, 8, 2), dtype=torch.int64, device=device)
        fallback.to(device)(q, positions)
        self.assertTrue(torch.equal(before, q.detach()))


class TestForcedFallback(unittest.TestCase):
    def test_the_switch_blocks_the_curope_import(self):
        sys.path.insert(0, str(PROJECT_ROOT / "worker"))
        import runner

        runner.force_rope_fallback()
        try:
            with self.assertRaises(ImportError):
                import models.curope  # noqa: F401
        finally:
            # remove the blocker again
            sys.meta_path = [
                f for f in sys.meta_path
                if not isinstance(f, runner._BlockCuropeFinder)
            ]
            for name in list(sys.modules):
                if name == "models.curope" or name.startswith("models.curope."):
                    del sys.modules[name]

    def test_the_switch_is_off_by_default(self):
        sys.path.insert(0, str(PROJECT_ROOT / "worker"))
        import runner

        self.assertFalse(any(
            isinstance(f, runner._BlockCuropeFinder) for f in sys.meta_path
        ))


class TestBuildHelpers(unittest.TestCase):
    def test_the_build_script_exists(self):
        self.assertTrue((PROJECT_ROOT / "scripts" / "build_curope.sh").is_file())

    def test_the_setup_script_builds_curope(self):
        text = (PROJECT_ROOT / "scripts" / "setup_env.sh").read_text()
        self.assertIn("build_curope.sh", text)

    def test_the_patch_is_idempotent(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import patch_curope

        with tempfile_dir() as tmp:
            path = tmp / "kernels.cu"
            path.write_text(
                "int main() {}\n"
                + patch_curope.ORIGINAL
                + "}\n"
            )
            self.assertTrue(patch_curope.patch(path))
            self.assertIn(patch_curope.PATCHED, path.read_text())
            # a second run must not patch twice
            self.assertFalse(patch_curope.patch(path))
            self.assertEqual(path.read_text().count(patch_curope.PATCHED), 1)

    def test_the_patch_fails_loudly_on_unexpected_sources(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import patch_curope

        with tempfile_dir() as tmp:
            path = tmp / "kernels.cu"
            path.write_text("int main() {}\n")
            with self.assertRaises(SystemExit):
                patch_curope.patch(path)

    def test_the_setup_declares_both_gpu_architectures(self):
        text = (PROJECT_ROOT / "scripts" / "cuda_env.sh").read_text()
        self.assertIn("8.6", text)     # NVIDIA A40
        self.assertIn("8.9", text)     # RTX 4060 Ti

    def test_the_build_uses_our_own_arch_explicit_setup(self):
        text = (PROJECT_ROOT / "scripts" / "curope_build" / "setup.py").read_text()
        self.assertIn("compute_86", text)
        self.assertIn("compute_89", text)
        # the fast-math intrinsics must not be enabled (they cost accuracy)
        flags_line = [ln for ln in text.splitlines() if "nvcc=" in ln]
        self.assertTrue(flags_line, "the nvcc flags were not found")
        for line in flags_line:
            self.assertNotIn("--use_fast_math", line,
                             "--use_fast_math degrades the rope accuracy")


class _tempfile_dir:
    def __enter__(self):
        import tempfile

        self._dir = tempfile.mkdtemp(prefix="te_curope_test_")
        return Path(self._dir)

    def __exit__(self, *exc):
        import shutil

        shutil.rmtree(self._dir, ignore_errors=True)
        return False


def tempfile_dir():
    return _tempfile_dir()


if __name__ == "__main__":
    unittest.main(verbosity=2)
