# SPDX-License-Identifier: Apache-2.0
"""PERF-ST comm-time L2 prefetch of the routed MoE FC1 weights (SGLANG_PST_L2PF).

In a TEP8 verify layer the attention exit (out_proj, then the AllReduce + residual + RMSNorm kernels, then the router,
top-k and routing kernels) leaves the local HBM almost idle for 30-40 us, and the routed FC1 that follows streams
every local expert's NVFP4 gate / up weights at the HBM rate (B48: 288 MB in about 45 us). From B32 up nearly every
local expert has rows, so the first experts' weights can be fetched into L2 before routing is known.

A small kernel on a side stream issues L2 prefetches for the first N local experts of FC1 (weights and block scales,
expert-major) while the main stream runs the exit; the main stream's kernels and their PDL edges are unchanged, the
side stream joins once after the last decoder layer. N, the instruction kind and the two launch points read a
pinned host word at every launch, so one CUDA graph serves every setting (``SGLANG_PST_L2PF_CTL``: a file a thread
re-reads every 0.5 s, "<n_pre_ar> <n_pre_mlp> [mode]"; mode 0 = cp.async.bulk.prefetch.L2, 1 = prefetch.global.L2 per
128-byte line, 2 = bulk with an evict_last policy).
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SRC = r"""
typedef unsigned long long u64;
struct Region { const char* ptr; u64 unit; u64 units; };

__device__ __forceinline__ void bulk_pf(const char* p, unsigned sz) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(sz) : "memory");
}
__device__ __forceinline__ void bulk_pf_last(const char* p, unsigned sz, u64 pol) {
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(p), "r"(sz), "l"(pol) : "memory");
}
__device__ __forceinline__ void line_pf(const char* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p) : "memory");
}

// ctl[slot]: experts' worth of bytes to prefetch (<= 0: nothing); ctl[3]: mode = instruction (bits 0-1: 0 bulk
// cp.async.bulk.prefetch.L2, 1 prefetch.global.L2 per 128-byte line, 2 bulk + evict_last) | 4: the last experts instead
// of the first | 8: spread, the same bytes as the first n / units of every expert. One region pair (weights, block scales).
__device__ __forceinline__ void pf_range(const char* p, u64 bytes, u64 tid, u64 nth, int ins, u64 pol) {
  if (ins == 1) {
#pragma unroll 4
    for (u64 off = tid * 128; off < bytes; off += nth * 128) line_pf(p + off);
  } else {
    const u64 chunk = 32768;
#pragma unroll 1
    for (u64 off = tid * chunk; off < bytes; off += nth * chunk) {
      u64 sz = bytes - off < chunk ? bytes - off : chunk;
      sz &= ~(u64)15;
      if (sz == 0) continue;
      if (ins == 2) bulk_pf_last(p + off, (unsigned)sz, pol); else bulk_pf(p + off, (unsigned)sz);
    }
  }
}

extern "C" __global__ void pst_l2pf(Region w, Region s, const volatile int* ctl, int slot) {
  const int n = ctl[slot];
  if (n <= 0) return;
  const int mode = ctl[3];
  const int ins = mode & 3;
  const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const u64 nth = (u64)gridDim.x * blockDim.x;
  u64 pol = 0;
  if (ins == 2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
#pragma unroll 1
  for (int r = 0; r < 2; ++r) {
    const Region& g = r == 0 ? w : s;
    if (g.units == 0) continue;
    const u64 cnt = (u64)n < g.units ? (u64)n : g.units;
    if (mode & 8) {
      // the first cnt / units of every expert: one range an expert, experts spread over the threads
      const u64 per = ((g.unit * cnt / g.units) + 15) & ~(u64)15;
      const u64 tpe = nth >= g.units ? nth / g.units : 1;  // threads an expert
#pragma unroll 1
      for (u64 e = tid / tpe; e < g.units; e += (nth / tpe > 0 ? nth / tpe : 1))
        pf_range(g.ptr + e * g.unit, per, tid % tpe, tpe, ins, pol);
    } else {
      const u64 first = (mode & 4) ? g.units - cnt : 0;
      pf_range(g.ptr + first * g.unit, cnt * g.unit, tid, nth, ins, pol);
    }
  }
}
"""


class _Region(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("unit", ctypes.c_uint64), ("units", ctypes.c_uint64)]


def _check(res, what):
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    err, *rest = res
    ok = err in (cu.CUresult.CUDA_SUCCESS, nvrtc.nvrtcResult.NVRTC_SUCCESS)
    if not ok:
        raise RuntimeError(f"st_l2_prefetch: {what} failed: {err}")
    return rest[0] if len(rest) == 1 else rest


class _State:
    def __init__(self, device: torch.device):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        self.device = device
        prog = _check(nvrtc.nvrtcCreateProgram(_SRC.encode(), b"pst_l2pf.cu", 0, [], []), "create")
        major, minor = torch.cuda.get_device_capability(device)
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-default-device"]
        if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("st_l2_prefetch: NVRTC failed:\n" + log.decode(errors="replace"))
        cubin = b" " * _check(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
        _check(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
        nvrtc.nvrtcDestroyProgram(prog)
        self.module = _check(cu.cuModuleLoadData(cubin), "module")
        self.fn = _check(cu.cuModuleGetFunction(self.module, b"pst_l2pf"), "function")
        self.ctas = max(1, envs.SGLANG_PST_L2PF_CTAS.get())
        self.min_rows = envs.SGLANG_PST_L2PF_MIN_ROWS.get()
        self.threads = 32
        # ctl: [experts at the attention exit, experts before the MoE, unused, mode]; pinned, read by every launch
        self.ctl = torch.zeros(4, dtype=torch.int32, pin_memory=True)
        self.ctl[0] = envs.SGLANG_PST_L2PF_PRE_AR.get()
        self.ctl[1] = envs.SGLANG_PST_L2PF_PRE_MLP.get()
        self.ctl[3] = envs.SGLANG_PST_L2PF_MODE.get()
        self.stream = torch.cuda.Stream(device=device)
        self.pending = False
        self.launches = 0
        path = envs.SGLANG_PST_L2PF_CTL.get()
        if path:
            threading.Thread(target=self._ctl_loop, args=(path,), daemon=True, name="pst-l2pf-ctl").start()
        logger.info(
            "PERF-ST L2 prefetch (SGLANG_PST_L2PF): FC1 experts pre_ar %d / pre_mlp %d, mode %d, %d CTAs, ctl file %r",
            int(self.ctl[0]), int(self.ctl[1]), int(self.ctl[3]), self.ctas, path,
        )

    def _ctl_loop(self, path: str) -> None:
        import time

        last = None
        while True:
            try:
                with open(path) as f:
                    words = f.read().split()
                vals = [int(x) for x in words[:3]]
                if vals and vals != last:
                    self.ctl[0] = vals[0]
                    if len(vals) > 1:
                        self.ctl[1] = vals[1]
                    if len(vals) > 2:
                        self.ctl[3] = vals[2]
                    last = vals
                    logger.info("PERF-ST L2 prefetch ctl: %s", vals)
            except (OSError, ValueError):
                pass
            time.sleep(0.5)

    def launch(self, w: torch.Tensor, s: Optional[torch.Tensor], slot: int) -> None:
        from cuda.bindings import driver as cu

        main = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(main)
        rw = _Region(w.data_ptr(), w[0].numel() * w.element_size(), w.shape[0])
        rs = _Region(s.data_ptr(), s[0].numel() * s.element_size(), s.shape[0]) if s is not None else _Region(0, 0, 0)
        ctl_ptr = ctypes.c_void_p(self.ctl.data_ptr())
        slot_c = ctypes.c_int(slot)
        args = [rw, rs, ctl_ptr, slot_c]
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = self.ctas, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = self.threads, 1, 1
        cfg.sharedMemBytes = 0
        cfg.hStream = cu.CUstream(self.stream.cuda_stream)
        cfg.attrs, cfg.numAttrs = [], 0
        addresses = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        _check(cu.cuLaunchKernelEx(cfg, self.fn, ctypes.addressof(addresses), 0), "launch")
        if self.launches == 0:
            logger.info(
                "PERF-ST L2 prefetch engaged: FC1 %s (%d B an expert) + scales %s (%d B an expert), slot %d",
                tuple(w.shape), rw.unit, None if s is None else tuple(s.shape), rs.unit, slot,
            )
        self.pending = True
        self.launches += 1


_STATES: dict = {}
_LOCK = threading.Lock()
_ENABLED: Optional[bool] = None


def _enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = envs.SGLANG_PST_L2PF.get()
    return _ENABLED


def _state() -> Optional[_State]:
    if not _enabled():
        return None
    dev = torch.cuda.current_device()
    st = _STATES.get(dev)
    if st is None:
        # NVRTC + module load are not capturable; the graph runner's eager warmups build it first.
        if torch.cuda.is_current_stream_capturing():
            logger.warning("PERF-ST L2 prefetch: first call under graph capture, skipped")
            return None
        with _LOCK:
            st = _STATES.get(dev)
            if st is None:
                st = _State(torch.device("cuda", dev))
                _STATES[dev] = st
    return st


def _fc1_tensors(experts) -> Optional[tuple]:
    # NVFP4 trtllm-gen layout: [local experts, 2 x intermediate, hidden / 2] uint8, one contiguous slice an expert
    w = experts.w13_weight
    if w.dim() != 3 or w.dtype != torch.uint8 or not w.is_contiguous():
        return None
    s = experts.w13_weight_scale
    if s.dim() < 2 or s.shape[0] != w.shape[0] or not s.is_contiguous():
        s = None
    return w, s


def prefetch_fc1(experts, forward_batch, slot: int, rows: int) -> None:
    """Launch the FC1 weight prefetch of this layer's routed MoE on the side stream (slot 0: at the attention exit,
    before its AllReduce; slot 1: before the MoE, after the AllReduce + RMSNorm). Decode / target-verify batches of
    at least SGLANG_PST_L2PF_MIN_ROWS rows only."""
    st = _state()
    if st is None or rows < st.min_rows:
        return
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return
    t = _fc1_tensors(experts)
    if t is None:
        return
    st.launch(t[0], t[1], slot)


def prefetch_layer(layer, forward_batch, slot: int, rows: int) -> None:
    """prefetch_fc1 for a Qwen3.5 decoder layer whose MLP is a routed MoE block (the megakernel executor's hook)."""
    if not _enabled():
        return
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
        prefetch_fc1(layer.mlp.experts, forward_batch, slot, rows)


def join() -> None:
    """Join the side stream after the last decoder layer (no-op when nothing was launched)."""
    if not _STATES:
        return
    st = _STATES.get(torch.cuda.current_device())
    if st is None or not st.pending:
        return
    torch.cuda.current_stream(st.device).wait_stream(st.stream)
    st.pending = False
