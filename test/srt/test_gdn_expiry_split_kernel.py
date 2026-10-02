"""Split expiry truncation: launcher dispatch and Triton-interpreter equivalence.

The interpreter cases need CPU torch + triton and set TRITON_INTERPRET before
loading the kernel module; they check math and masking, NOT GPU bitwise
equality or timing.
"""

import importlib
import os
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")

KERNELS = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/layers/attention/linear/kernels"
)

try:
    import torch  # noqa: F401
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def _load():
    package = types.ModuleType("gdn_kernels_under_test")
    package.__path__ = [str(KERNELS)]
    sys.modules.setdefault("gdn_kernels_under_test", package)
    return importlib.import_module("gdn_kernels_under_test.gdn_factored")


class _Recorder:
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, kwargs))

        return launch


@unittest.skipUnless(HAVE_TRITON, "needs torch + triton")
class SplitDispatchTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.gf = _load()
        self.calls = []
        self.saved = {}
        for name in (
            "_factored_expiry_truncate_kernel",
            "_expiry_directions_kernel",
            "_expiry_project_kernel",
        ):
            self.saved[name] = getattr(self.gf, name)
            setattr(self.gf, name, _Recorder(self.calls, name))
        self.saved["expiry_split_enabled"] = self.gf.expiry_split_enabled
        self.fu = torch.zeros(4, 3, 32, 128)
        self.fw = torch.zeros(4, 3, 32, 128)
        self.cnt = torch.zeros(4, 3, dtype=torch.int32)
        self.idx = torch.tensor([0, 1], dtype=torch.int64)

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(self.gf, name, value)

    def _truncate(self, split):
        self.gf.expiry_split_enabled = lambda batch: split
        self.gf.factored_expiry_truncate(
            self.fu, self.fw, self.cnt, self.idx, 16, 32, method="mgs"
        )

    def _with_envs(self, enabled, min_batch):
        # The real gate reads sglang.srt.environ; stub only that module.
        name = "sglang.srt.environ"
        env = types.ModuleType(name)
        env.envs = types.SimpleNamespace(
            SGLANG_GDN_EXPIRY_SPLIT_KERNEL=types.SimpleNamespace(get=lambda: enabled),
            SGLANG_GDN_EXPIRY_SPLIT_MIN_BATCH=types.SimpleNamespace(get=lambda: min_batch),
        )
        patcher = mock.patch.dict(sys.modules, {name: env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gf.expiry_split_enabled = self.saved["expiry_split_enabled"]

    def test_batch_below_threshold_launches_the_old_fused_kernel(self):
        import torch

        fused = [("_factored_expiry_truncate_kernel", (8 * 3,))]
        split = ["_expiry_directions_kernel", "_expiry_project_kernel"]
        cases = [(True, 16, 8, fused), (True, 16, 16, split), (True, 4, 8, split),
                 (False, 1, 64, [("_factored_expiry_truncate_kernel", (64 * 3,))])]
        for enabled, min_batch, batch, expected in cases:
            self._with_envs(enabled, min_batch)
            self.calls.clear()
            self.gf.factored_expiry_truncate(
                self.fu, self.fw, self.cnt, torch.zeros(batch, dtype=torch.int64),
                16, 32, method="mgs",
            )
            got = [c[0] if isinstance(expected[0], str) else c[:2] for c in self.calls]
            # One predicate launch per layer below the threshold, as with the switch off.
            self.assertEqual(got, expected, (enabled, min_batch, batch))

    def test_switch_off_launches_only_the_fused_kernel(self):
        self._truncate(False)
        self.assertEqual(
            [(name, grid) for name, grid, _ in self.calls],
            [("_factored_expiry_truncate_kernel", (6,))],
        )

    def test_switch_on_launches_directions_then_every_tile(self):
        self._truncate(True)
        names = [name for name, _, _ in self.calls]
        self.assertEqual(names, ["_expiry_directions_kernel", "_expiry_project_kernel"])
        (_, grid_a, kw_a), (_, grid_b, kw_b) = self.calls
        self.assertEqual(grid_a, (6, 1))
        # K/FT U tiles plus V/FT W tiles cover all 256 feature columns.
        self.assertEqual(grid_b, (6, 128 // kw_b["FT"] + 128 // kw_b["FT"], 1))
        self.assertEqual((kw_a["RK"], kw_b["RK"], kw_a["R"]), (16, 16, 16))


@unittest.skipUnless(HAVE_TRITON, "needs torch + triton")
class SplitInterpreterTest(unittest.TestCase):
    """Old fused kernel vs the two stages on the same inputs (interpreter)."""

    def _case(self, r, rfull):
        import torch

        gf = _load()
        g = torch.Generator().manual_seed(7)
        S, HV, K, V, rmax = 5, 2, 128, 128, rfull
        fu = torch.randn(S, HV, rmax, K, generator=g)
        fw = torch.randn(S, HV, rmax, V, generator=g) * torch.logspace(0, -3, rmax)[
            None, None, :, None
        ]
        cnt = torch.full((S, HV), rfull, dtype=torch.int32)
        cnt[3, 1] = rfull - 1  # one head not due
        idx = torch.tensor([0, 1, 3, -1], dtype=torch.int64)  # -1 = padding row
        old = [fu.clone(), fw.clone(), cnt.clone()]
        gf._factored_expiry_truncate_kernel[(idx.numel() * HV,)](
            *old, idx, stride_idx=1, HV=HV, K=K, V=V, RMAX=rmax, R=r, RFULL=rfull,
            ITERS=gf.TRUNC_ITERS, REL_TOL=gf.MGS_REL_TOL,
        )
        new = [fu.clone(), fw.clone(), cnt.clone()]
        gf.expiry_truncate_split(*new, idx, r, rfull)
        return (fu, fw, cnt), old, new

    def _check(self, r, rfull):
        import torch

        before, old, new = self._case(r, rfull)
        self.assertTrue(torch.equal(old[2], new[2]))  # counts: r where due
        for x, y in zip(old[:2], new[:2]):
            torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)
        for b, y in zip(before[:2], new[:2]):
            # Rows >= r are never written; not-due heads and slots outside
            # the batch keep every row.
            self.assertTrue(torch.equal(y[:, :, r:], b[:, :, r:]))
            self.assertTrue(torch.equal(y[3, 1], b[3, 1]))
            self.assertTrue(torch.equal(y[2], b[2]))

    def test_r16_w16(self):
        self._check(16, 32)

    def test_r8_w8(self):
        self._check(8, 16)


if __name__ == "__main__":
    unittest.main()
