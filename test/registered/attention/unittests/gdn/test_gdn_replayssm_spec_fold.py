"""GDN ReplaySSM fold-every-commit: fused ring-write + commit fold.

The production kernel targets bitwise parity with the recurrent verify and
per-draft snapshot baseline. These tests allow an absolute error up to FP32_ATOL
for committed/tracked state and downstream outputs, and verify that bound
through 256 chained commits. Ring-write output and untouched/null slots remain
exact.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

B, T = 3, 4
H, HV = 4, 8
K = V = 64
NUM_SLOTS = 8
DEVICE = "cuda"
# Absolute allowance for this deterministic regression case, not a general
# numerical-error guarantee for ReplaySSM.
FP32_ATOL = 2 * torch.finfo(torch.float32).eps


def _make_window(step_seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(step_seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    return {
        "q": rand(1, B * T, H, K),
        "k": rand(1, B * T, H, K),
        "v": rand(1, B * T, HV, V),
        "a": rand(B * T, HV),
        "b": rand(B * T, HV),
    }


def _run_verify(inputs, gating, state, slots, *, snapshots=None, rings=None):
    kwargs = {}
    if snapshots is not None:
        kwargs.update(
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=slots,
            cache_steps=T,
        )
    if rings is not None:
        # Per-layer views, matching the backend's mamba2_layer_cache slices.
        kwargs.update(
            cache_ring=True,
            replayssm_rawv=rings["rawv"][0],
            replayssm_rawk=rings["rawk"][0],
            replayssm_g=rings["g"][0],
            replayssm_beta=rings["beta"][0],
        )
    cu_seqlens = torch.arange(0, B * T + 1, step=T, dtype=torch.int32, device=DEVICE)
    return fused_sigmoid_gating_delta_rule_update(
        A_log=gating["A_log"],
        dt_bias=gating["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        b=inputs["b"],
        a=inputs["a"],
        initial_state_source=state,
        initial_state_indices=slots,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        disable_state_update=True,
        **kwargs,
    )


def _make_rings(dtype=torch.bfloat16):
    return {
        "rawv": torch.zeros(1, NUM_SLOTS, HV, T, V, device=DEVICE, dtype=dtype),
        "rawk": torch.zeros(1, NUM_SLOTS, H, T, K, device=DEVICE, dtype=dtype),
        "g": torch.zeros(1, NUM_SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
        "beta": torch.zeros(1, NUM_SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
    }


def _fold(state, rings, slots, accept_lens, track_slots=None, track_steps=None):
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=state,
        rawv_cache=rings["rawv"],
        rawk_cache=rings["rawk"],
        g_cache=rings["g"],
        beta_cache=rings["beta"],
        ssm_state_indices=slots,
        accept_lens=accept_lens,
        max_cache_len=T,
        num_k_heads=H,
        mamba_track_indices=track_slots,
        mamba_steps_to_track=track_steps,
        null_block_id=-1,
    )


class TestGdnReplayssmSpecFold(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.gating = {
            "A_log": torch.randn(HV, device=DEVICE) * 0.1,
            "dt_bias": torch.randn(HV, device=DEVICE) * 0.1,
        }
        self.slots = torch.tensor([5, 2, 7], dtype=torch.int32, device=DEVICE)
        self.accept_lens = torch.tensor([3, 1, 4], dtype=torch.int32, device=DEVICE)

    def _state(self, dtype):
        gen = torch.Generator(device=DEVICE).manual_seed(1)
        return torch.randn(
            NUM_SLOTS, HV, K, V, device=DEVICE, dtype=dtype, generator=gen
        )

    def test_ring_write_does_not_change_verify_output(self):
        for dtype in (torch.float32, torch.bfloat16):
            state = self._state(dtype)
            inputs = _make_window(11)
            out_plain = _run_verify(inputs, self.gating, state.clone(), self.slots)
            out_ring = _run_verify(
                inputs, self.gating, state.clone(), self.slots, rings=_make_rings()
            )
            self.assertTrue(torch.equal(out_plain, out_ring), f"{dtype=}")

    def test_ring_write_accepts_strided_qkv_views(self):
        inputs = _make_window(12)
        packed_qkv = torch.cat(
            [inputs[name].reshape(B * T, -1) for name in ("q", "k", "v")],
            dim=-1,
        )
        q, k, v = packed_qkv.split([H * K, H * K, HV * V], dim=-1)
        strided_qkv = {
            "q": q.view(1, B * T, H, K),
            "k": k.view(1, B * T, H, K),
            "v": v.view(1, B * T, HV, V),
        }
        self.assertTrue(
            all(not tensor.is_contiguous() for tensor in strided_qkv.values())
        )

        def run(qkv):
            state = self._state(torch.float32).unsqueeze(0).contiguous()
            rings = _make_rings()
            output = _run_verify(
                {**inputs, **qkv},
                self.gating,
                state[0],
                self.slots,
                rings=rings,
            )
            _fold(state, rings, self.slots, self.accept_lens)
            return {"output": output, "state": state, **rings}

        contiguous = run(
            {name: tensor.contiguous() for name, tensor in strided_qkv.items()}
        )
        strided = run(strided_qkv)
        for name in contiguous:
            self.assertTrue(torch.equal(contiguous[name], strided[name]), name)

    def test_fold_matches_snapshot_baseline(self):
        for dtype in (torch.float32, torch.bfloat16):
            state = self._state(dtype)
            inputs = _make_window(22)

            snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
            _run_verify(
                inputs, self.gating, state.clone(), self.slots, snapshots=snapshots
            )

            fold_state = state.clone().unsqueeze(0).contiguous()
            rings = _make_rings()
            _run_verify(inputs, self.gating, fold_state[0], self.slots, rings=rings)
            _fold(fold_state, rings, self.slots, self.accept_lens)

            for s, n in zip(self.slots.tolist(), self.accept_lens.tolist()):
                torch.testing.assert_close(
                    snapshots[s, n - 1],
                    fold_state[0, s],
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} slot={s} accept_len={n}",
                )
            untouched = set(range(NUM_SLOTS)) - set(self.slots.tolist())
            for s in untouched:
                self.assertTrue(torch.equal(fold_state[0, s], state[s]))

    def test_track_store_and_null_slots(self):
        dtype = torch.float32
        state = self._state(dtype)
        inputs = _make_window(33)

        snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
        _run_verify(inputs, self.gating, state.clone(), self.slots, snapshots=snapshots)

        fold_state = state.clone().unsqueeze(0).contiguous()
        rings = _make_rings()
        _run_verify(inputs, self.gating, fold_state[0], self.slots, rings=rings)

        track_slots = torch.tensor([1, 0, 3], dtype=torch.int64, device=DEVICE)
        track_steps = torch.tensor([1, -1, 2], dtype=torch.int64, device=DEVICE)
        slots_with_null = self.slots.clone()
        slots_with_null[1] = -1
        _fold(
            fold_state,
            rings,
            slots_with_null,
            self.accept_lens,
            track_slots=track_slots,
            track_steps=track_steps,
        )

        torch.testing.assert_close(
            fold_state[0, 1], snapshots[5, 1], rtol=0, atol=FP32_ATOL
        )
        torch.testing.assert_close(
            fold_state[0, 3], snapshots[7, 2], rtol=0, atol=FP32_ATOL
        )
        # Row 1's state slot is replaced with -1, and its track step is -1, so
        # neither its original state slot 2 nor tracking slot 0 is written.
        self.assertTrue(torch.equal(fold_state[0, 2], state[2]))
        self.assertTrue(torch.equal(fold_state[0, 0], state[0]))

    def test_long_chain_error_stays_bounded(self):
        """This regression case remains within FP32_ATOL through 256 commits."""
        num_iters = 256
        for dtype in (torch.float32, torch.bfloat16):
            base_state = self._state(dtype)
            fold_state = base_state.clone().unsqueeze(0).contiguous()
            snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
            gen = torch.Generator().manual_seed(7)
            for it in range(num_iters):
                inputs = _make_window(1000 + it)
                accept_lens = torch.randint(1, T + 1, (B,), generator=gen).to(
                    device=DEVICE, dtype=torch.int32
                )

                out_base = _run_verify(
                    inputs, self.gating, base_state, self.slots, snapshots=snapshots
                )
                for s, n in zip(self.slots.tolist(), accept_lens.tolist()):
                    base_state[s] = snapshots[s, n - 1]

                rings = _make_rings()
                out_fold = _run_verify(
                    inputs, self.gating, fold_state[0], self.slots, rings=rings
                )
                _fold(fold_state, rings, self.slots, accept_lens)

                torch.testing.assert_close(
                    out_base,
                    out_fold,
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} {it=}",
                )
                torch.testing.assert_close(
                    base_state,
                    fold_state[0],
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} {it=}",
                )


def _cuda_commit_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        return False
    try:
        import cuda.bindings.nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_cuda_commit_available(), "needs SM90+ and cuda.bindings")
class TestGdnReplayssmCommitCuda(CustomTestCase):
    """SGLANG_Q35MF_COMMIT_CUDA: the CUDA compact commit against the Triton one on
    TP8 shapes (V = K = 128, ring 16), with track rows, residuals, a null row and a
    window longer than 8 entries. FP32 sums run in another order, so the BF16
    checkpoint may differ by one ulp; cursors and untouched slots stay exact."""

    def test_cuda_commit_matches_triton(self):
        from sglang.kernels.ops.attention.fla import gdn_replayssm_commit_cuda
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_circular,
        )

        layers, slots, h, hv, d, ring = 2, 12, 2, 8, 128, 16
        gen = torch.Generator(device=DEVICE).manual_seed(3)

        def rand(*shape, scale):
            x = torch.randn(*shape, device=DEVICE, generator=gen) * scale
            return x.to(torch.bfloat16)

        state = rand(layers, slots, hv, d, d, scale=0.05)
        caches = dict(
            d_cache=rand(layers, slots, hv, ring, d, scale=0.05),
            k_cache=rand(layers, slots, h, ring, d, scale=0.09),
            g_cache=-torch.rand(layers, slots, hv, ring, device=DEVICE, generator=gen)
            * 0.5,
            d_residual_cache=rand(layers, slots, hv, ring, d, scale=1e-4),
            k_residual_cache=rand(layers, slots, h, ring, d, scale=1e-4),
        )
        state_rows = torch.tensor([3, 7, -1, 1], dtype=torch.int32, device=DEVICE)
        replay_rows = torch.tensor([0, 5, 9, 2], dtype=torch.int64, device=DEVICE)
        accept_lens = torch.tensor([3, 7, 2, 1], dtype=torch.int32, device=DEVICE)
        write_pos = torch.zeros(slots, dtype=torch.int32, device=DEVICE)
        write_pos[replay_rows] = torch.tensor([3, 12, 2, 1], device=DEVICE).int()
        cache_base = torch.randint(
            0, ring, (slots,), device=DEVICE, generator=gen
        ).int()
        is_flush = torch.ones(slots, dtype=torch.int8, device=DEVICE)
        track_rows = torch.tensor([10, 11, -1, 4], dtype=torch.int64, device=DEVICE)
        track_steps = torch.tensor([1, 6, -1, 0], dtype=torch.int32, device=DEVICE)

        outs = []
        for use_cuda in (False, True):
            cursors = [t.clone() for t in (write_pos, cache_base, is_flush)]
            out = state.clone()
            commit_gdn_replayssm_circular(
                checkpoint_state=out,
                **caches,
                state_batch_indices=state_rows,
                replay_indices=replay_rows,
                write_pos=cursors[0],
                cache_base=cursors[1],
                is_flush=cursors[2],
                accept_lens=accept_lens,
                mamba_track_indices=track_rows,
                mamba_steps_to_track=track_steps,
                null_block_id=-1,
                use_cuda=use_cuda,
            )
            outs.append((out, cursors))
        self.assertTrue(gdn_replayssm_commit_cuda._KERNELS)

        (want, want_cursors), (got, got_cursors) = outs
        for a, b in zip(want_cursors, got_cursors):
            self.assertTrue(torch.equal(a, b))
        written = [3, 7, 1, 10, 11, 4]
        torch.testing.assert_close(
            got[:, written].float(), want[:, written].float(), rtol=2**-7, atol=1e-6
        )
        self.assertFalse(torch.equal(want[:, written], state[:, written]))
        untouched = [s for s in range(slots) if s not in written]
        self.assertTrue(torch.equal(got[:, untouched], state[:, untouched]))


@unittest.skipUnless(_cuda_commit_available(), "needs SM90+ and cuda.bindings")
class TestGdnCoreCudaTrtFold(CustomTestCase):
    """SGLANG_PST_GDN_TRT_FOLD: gdn_core_cuda fold 2 / 3 on two 16-entry history buffers (a ring of 32)
    against fold 0 on the active buffer and the CUDA commit of the history, bit for bit. A step whose rows
    do not fit after its history folds it into the checkpoint and writes its rows from entry 0 of the other
    buffer; fold 2 replays the history for its outputs, fold 3 runs on the folded checkpoint."""

    H, HV, T, RING = 2, 4, 7, 32

    def _inputs(self, gen):
        h, hv, t, d = self.H, self.HV, self.T, 128
        n, slots = 6, 9

        def rand(*shape, scale=1.0):
            return (torch.randn(*shape, device=DEVICE, generator=gen) * scale).to(
                torch.bfloat16
            )

        rows = n * t
        ab = rand(rows, 2 * hv)
        x = dict(
            qkv=rand(rows, (2 * h + hv) * d),
            z=rand(rows, hv, d),
            a=ab[:, :hv],
            b=ab[:, hv:],
            A_log=torch.rand(hv, device=DEVICE, generator=gen),
            dt_bias=torch.rand(hv, device=DEVICE, generator=gen),
            norm_weight=rand(d),
            input_scale=torch.tensor([0.02], device=DEVICE),
        )
        state = rand(slots, hv, d, d, scale=0.05)
        ring = [
            rand(slots, hv, self.RING, d, scale=0.05),
            rand(slots, h, self.RING, d, scale=0.09),
            -torch.rand(slots, hv, self.RING, device=DEVICE, generator=gen) * 0.5,
        ]
        states = torch.tensor([4, 0, 7, -1, 2, 5], dtype=torch.int32, device=DEVICE)
        replay = torch.tensor([3, 8, 1, 6, 0, 5], dtype=torch.int32, device=DEVICE)
        wp = torch.zeros(slots, dtype=torch.int32, device=DEVICE)
        wp[replay.long()] = torch.tensor([0, 5, 9, 12, 10, 16], device=DEVICE).int()
        base = torch.zeros(slots, dtype=torch.int32, device=DEVICE)
        base[replay.long()] = torch.tensor([16, 0, 16, 0, 0, 16], device=DEVICE).int()
        return x, state, ring, states, replay, wp, base

    def _core(self, x, state, ring, states, replay, wp, base, fold):
        from sglang.kernels.ops.attention.fla import gdn_core_cuda

        rows = x["qkv"].shape[0]
        y = torch.empty(
            rows, self.HV * 128, dtype=torch.float8_e4m3fn, device=DEVICE
        )
        o = torch.zeros(rows, self.HV, 128, dtype=torch.bfloat16, device=DEVICE)
        self.assertTrue(
            gdn_core_cuda.try_core(
                **x,
                checkpoint=state,
                ring_d=ring[0],
                ring_k=ring[1],
                ring_g=ring[2],
                ring_dlo=None,
                ring_klo=None,
                state_indices=states,
                replay_indices=replay,
                write_pos=wp,
                cache_base=base,
                is_flush=torch.zeros_like(wp, dtype=torch.int8),
                eps=1e-6,
                out=y,
                rows_per_request=self.T,
                num_k_heads=self.H,
                o_out=o,
                fold=fold,
            )
        )
        return y, o

    def _check(self, fold):
        from sglang.kernels.ops.attention.fla import gdn_replayssm_commit_cuda

        gen = torch.Generator(device=DEVICE).manual_seed(11 + fold)
        x, state, ring, states, replay, wp, base = self._inputs(gen)
        folds = (wp + self.T > 16).to(torch.int8)
        got_state, got_ring = state.clone(), [r.clone() for r in ring]
        y, o = self._core(x, got_state, got_ring, states, replay, wp, base, fold)

        # want: the commit of each folding history, then fold 0 on the active buffer (fold 3: a folding
        # row on the committed checkpoint with no history)
        committed = state.clone().unsqueeze(0)
        gdn_replayssm_commit_cuda.commit(
            checkpoint_state=committed,
            d_cache=ring[0].unsqueeze(0),
            k_cache=ring[1].unsqueeze(0),
            g_cache=ring[2].unsqueeze(0),
            d_residual_cache=None,
            k_residual_cache=None,
            state_batch_indices=states,
            replay_indices=replay,
            write_pos=wp,
            cache_base=base,
            is_flush=folds,
            accept_lens=wp[: states.numel()].contiguous(),
            null_block_id=-1,
        )
        committed = committed.squeeze(0)
        wp_ref = torch.where(folds.bool(), torch.zeros_like(wp), wp) if fold == 3 else wp
        active = [torch.zeros_like(r[:, :, :16]) for r in ring]
        for rid in range(wp.numel()):
            b0 = int(base[rid])
            for r16, r32 in zip(active, ring):
                r16[rid] = r32[rid, :, b0 : b0 + 16]
        ref_state = committed.clone() if fold == 3 else state.clone()
        y_ref, o_ref = self._core(
            x, ref_state, active, states, replay, wp_ref, torch.zeros_like(base), 0
        )
        torch.cuda.synchronize()

        live = (states >= 0).repeat_interleave(self.T)
        self.assertTrue(torch.equal(o[live], o_ref[live]))
        self.assertTrue(torch.equal(y.view(torch.uint8)[live], y_ref.view(torch.uint8)[live]))
        self.assertFalse(y[~live].float().any())
        self.assertTrue(torch.equal(got_state, committed))
        want_ring = [r.clone() for r in ring]
        for n in range(states.numel()):
            if int(states[n]) < 0:
                continue
            rid = int(replay[n])
            w, b0, f = int(wp[rid]), int(base[rid]), bool(folds[rid])
            first = (b0 ^ 16) if f else b0 + w
            for t in range(self.T):
                src = int(wp_ref[rid]) + t
                if src >= 16:
                    # fold 2's reference (fold 0, ring of 16) drops a folding row's late entries
                    for want, got in zip(want_ring, got_ring):
                        want[rid, :, first + t] = got[rid, :, first + t]
                    continue
                for want, act in zip(want_ring, active):
                    want[rid, :, first + t] = act[rid, :, src]
        for want, got in zip(want_ring, got_ring):
            self.assertTrue(torch.equal(want, got))

    def test_fold2_replays_history_and_commits_into_the_other_buffer(self):
        self._check(2)

    def test_fold3_runs_on_the_committed_checkpoint(self):
        self._check(3)


if __name__ == "__main__":
    unittest.main()
