"""ctypes glue for mkh_hop.cu: workspaces, launches and bitwise checks for hop_bench.py."""

import ctypes
import hashlib
import os
import subprocess
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = 3
CTAS = 4
FRAGS = 512

# Variant codes, mirrored from mkh_hop.cu.
(V_ONE_LL, V_ONE_FIN, V_PROD_PUSH, V_PROD_PLAIN, V_CONSUME, V_LDR1, V_LDR2, V_ONE_TU, V_ONE_FIN_FMA,
 V_ONE_LL_UNI, V_PROD_PUSH_UNI, V_ONE_LL_WEAK, V_PUBLISH, V_PUBLISH_FIN, V_FUSED,
 V_PUBLISH_FIN_QUAD, V_PUBLISH_FIN_QUAD4, V_PUBLISH_FIN_K10) = range(1, 19)


class Params(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "partial", "routed", "expert_w", "permuted", "shared", "residual_in", "gamma",
        "fp8_scale", "prod_src", "residual_out", "norm_out", "fp8_out", "prod_out",
        "mb_mc", "mb_uc", "aux_mc", "aux_uc", "ctrl", "stamps")] + [
        (n, ctypes.c_int) for n in ("rank", "world", "m", "m_cap", "top_k", "stamp_calls")] + [
        ("eps", ctypes.c_float), ("prod_coeff", ctypes.c_float),
        ("timeout_ns", ctypes.c_ulonglong), ("peer_uc", ctypes.c_uint64 * 8),
        ("poll_sleep_ns", ctypes.c_uint), ("self_only", ctypes.c_int), ("active_rank", ctypes.c_int),
        ("early_trigger", ctypes.c_int), ("poll_mode", ctypes.c_int)]


class OprojArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("w", "x", "sx", "sw", "out")]


def pack_rows32(w_fp8, k_steps_per_warp):
    """[4096, K] e4m3 -> [cta 128][warp 8][step S][block 2][lane 32][16 B] mma A fragments
    (same packing as the fused attention lane's o_proj core)."""
    w = w_fp8.view(torch.uint8)
    n, k = w.shape
    S = k_steps_per_warp
    assert k == 8 * S * 32
    v = w.view(n // 32, 2, 2, 8, 8, S, 2, 4, 4)
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).contiguous().view(-1)


def _build(rank):
    src = os.path.join(HERE, "mkh_hop.cu")
    digest = hashlib.md5(open(src, "rb").read()).hexdigest()[:12]
    out_dir = os.path.join(HERE, "bin")
    os.makedirs(out_dir, exist_ok=True)
    lib = os.path.join(out_dir, f"libmkh_hop.{digest}.so")
    if rank == 0 and not os.path.exists(lib):
        tmp = f"{lib}.tmp.{os.getpid()}"
        cmd = ["nvcc", "-O3", "-std=c++17", "-arch=sm_103a", "-shared", "-Xcompiler", "-fPIC",
               "-o", tmp, src]
        res = subprocess.run(cmd, capture_output=True, text=True)
        with open(os.path.join(out_dir, f"build.{digest}.log"), "w") as f:
            f.write(" ".join(cmd) + "\n" + res.stdout + res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"nvcc failed: {res.stderr[-2000:]}")
        os.rename(tmp, lib)
    for _ in range(900):
        if os.path.exists(lib):
            break
        time.sleep(1)
    return ctypes.CDLL(lib), digest


def _symm(shape, dtype, device, group, peers=False):
    import torch.distributed._symmetric_memory as symm_mem
    from flashinfer.comm.mnnvl_cutedsl.symmetric_buffer import SymmetricBuffer

    if symm_mem.get_backend(device) is None:
        symm_mem.set_backend("NCCL")
    return SymmetricBuffer.allocate(shape, dtype, device, group, require_multicast=True,
                                    materialize_peer_addresses=peers)


class _Instance:
    """One transport workspace (mailboxes + control block) for one (variant, M)."""

    def __init__(self, hop, kind, m):
        self.hop, self.kind, self.m = hop, kind, m
        dev, w, cap, h = hop.device, hop.world, hop.m_cap, hop.hidden
        self.ctrl = torch.zeros(4, dtype=torch.int32, device=dev)
        self.aux = None
        self.peers = [0] * 8
        if kind == "ll":
            self.mb = _symm((STAGES, w, cap, h), torch.int16, dev, hop.group, peers=True)
            self.mb.tensor.fill_(-32768)  # bf16 -0.0
            self.peers[:w] = [int(a) for a in self.mb.peer_addresses.cpu().tolist()]
        elif kind == "tu":
            self.mb = _symm((STAGES, w, cap, FRAGS, 4), torch.int64, dev, hop.group)
            self.mb.tensor.zero_()
        else:  # ldr1 / ldr2
            self.mb = _symm((STAGES, cap, h), torch.int16, dev, hop.group)
            self.mb.tensor.fill_(0x7FC0)  # bf16 NaN
            if kind == "ldr2":
                self.aux = _symm((STAGES, cap, h), torch.int16, dev, hop.group)
                self.aux.tensor.fill_(-32768)
        torch.cuda.synchronize()
        dist.barrier(group=hop.group)


class MkhHop:
    def __init__(self, *, hidden, max_m, rank, world, device, group, timeout_s=2.0):
        self.lib, self.digest = _build(rank)
        if self.lib.mkh_params_size() != ctypes.sizeof(Params):
            raise RuntimeError("Params layout mismatch")
        if self.lib.mkh_hidden() != hidden:
            raise RuntimeError(f"mkh_hop.cu is built for hidden {self.lib.mkh_hidden()}")
        self.hidden, self.m_cap, self.rank, self.world = hidden, max_m, rank, world
        self.device, self.group = device, group
        self.timeout_ns = int(timeout_s * 1e9)
        self.instances = {}
        self.outputs = {}
        self.stamps = None
        self.poll_sleep_ns = int(os.environ.get("MKH_POLL_SLEEP_NS", "0"))
        self.self_only = int(os.environ.get("MKH_SELF_ONLY", "0"))
        self.active_rank = int(os.environ.get("MKH_ACTIVE_RANK", "-1"))
        self.early_trigger = int(os.environ.get("MKH_EARLY_TRIGGER", "0"))
        self.poll_mode = int(os.environ.get("MKH_POLL_MODE", "0"))

    def _instance(self, kind, m, tag=""):
        key = (kind + tag, m)
        if key not in self.instances:
            self.instances[key] = _Instance(self, kind, m)
        return self.instances[key]

    def _outputs(self, key, m):
        if key not in self.outputs:
            h = self.hidden
            self.outputs[key] = dict(
                residual=torch.empty(m, h, dtype=torch.bfloat16, device=self.device),
                norm=torch.empty(m, h, dtype=torch.bfloat16, device=self.device),
                fp8=torch.empty(m, h, dtype=torch.uint8, device=self.device),
                prod=torch.empty(m, h, dtype=torch.bfloat16, device=self.device),
            )
        return self.outputs[key]

    def _params(self, inst, *, m, outs, x=None, residual=None, gamma=None, scale=None,
                fin=None, prod_src=None, quant=True, norm=True, coeff=1.0):
        p = Params()
        ptr = lambda t: 0 if t is None else t.data_ptr()
        p.partial = ptr(x)
        if fin is not None:
            p.routed, p.expert_w, p.permuted, p.shared = (ptr(fin["routed"]), ptr(fin["weights"]),
                                                         ptr(fin["permuted"]), ptr(fin["shared"]))
            p.top_k = fin["permuted"].shape[1]
        p.residual_in, p.gamma, p.fp8_scale = ptr(residual), ptr(gamma), ptr(scale)
        p.prod_src = ptr(prod_src)
        p.residual_out = ptr(outs["residual"])
        p.norm_out = ptr(outs["norm"]) if norm else 0
        p.fp8_out = ptr(outs["fp8"]) if quant else 0
        p.prod_out = ptr(outs["prod"])
        p.mb_mc, p.mb_uc = inst.mb.multicast_address, inst.mb.tensor.data_ptr()
        if inst.aux is not None:
            p.aux_mc, p.aux_uc = inst.aux.multicast_address, inst.aux.tensor.data_ptr()
        p.ctrl = inst.ctrl.data_ptr()
        p.stamps = 0 if self.stamps is None else self.stamps.data_ptr()
        p.stamp_calls = 0 if self.stamps is None else self.stamps.shape[0]
        p.rank, p.world, p.m, p.m_cap = self.rank, self.world, m, self.m_cap
        p.eps, p.prod_coeff, p.timeout_ns = 1e-6, coeff, self.timeout_ns
        for i, a in enumerate(inst.peers):
            p.peer_uc[i] = a
        p.poll_sleep_ns = self.poll_sleep_ns
        p.self_only = self.self_only
        p.active_rank = self.active_rank
        p.early_trigger = self.early_trigger
        p.poll_mode = self.poll_mode
        return p

    def _launch(self, variant, order, p):
        rc = self.lib.mkh_launch(variant, order, ctypes.byref(p),
                                 ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"mkh_launch({variant}) returned {rc}")

    def make_call(self, name, *, m, x, residual, gamma, scale, fin=None, order=0):
        """name: one | one_nq | fin | fin_fma | tu | ldr1 | ldr2 (single hop calls)."""
        kind = {"one": "ll", "one_nq": "ll", "one_uni": "ll", "one_rlx": "ll", "fin": "ll",
                "fin_fma": "ll", "tu": "tu", "ldr1": "ldr1", "ldr2": "ldr2"}[name]
        variant = {"one": V_ONE_LL, "one_nq": V_ONE_LL, "one_uni": V_ONE_LL_UNI, "one_rlx": V_ONE_LL_WEAK,
                   "fin": V_ONE_FIN, "fin_fma": V_ONE_FIN_FMA, "tu": V_ONE_TU, "ldr1": V_LDR1,
                   "ldr2": V_LDR2}[name]
        inst = self._instance(kind, m)
        outs = self._outputs((name, m, order), m)
        p = self._params(inst, m=m, outs=outs, x=x, residual=residual, gamma=gamma, scale=scale,
                         fin=fin if name.startswith("fin") else None, quant=name != "one_nq")

        def call():
            self._launch(variant, order, p)

        return call

    def make_split(self, name, *, m, x, residual, gamma, scale, fin=None, order=0):
        """split | split_nq | split_fin | split_finq[4] | split_finq[4]_nq: publish launch + consume launch."""
        inst = self._instance("ll", m)
        outs = self._outputs((name, m, order), m)
        is_fin = name.startswith("split_fin")
        p = self._params(inst, m=m, outs=outs, x=x, residual=residual, gamma=gamma, scale=scale,
                         fin=fin if is_fin else None, quant=not name.endswith("_nq"))
        pub = {"split_fin": V_PUBLISH_FIN, "split_finq": V_PUBLISH_FIN_QUAD,
               "split_finq_nq": V_PUBLISH_FIN_QUAD, "split_finq4": V_PUBLISH_FIN_QUAD4,
               "split_finq4_nq": V_PUBLISH_FIN_QUAD4, "split_fink": V_PUBLISH_FIN_K10,
               "split_fink_nq": V_PUBLISH_FIN_K10}.get(name, V_PUBLISH)

        def call():
            self._launch(pub, 0, p)
            self._launch(V_CONSUME, order, p)

        return call

    def make_chain(self, name, *, m, x, residual, gamma, scale, fi_call=None, order=0):
        """Producer + hop pairs; the producer reads the previous hop's norm output.

        push      prod(push) -> consume                       (M2)
        plain_one prod(plain) -> one-kernel hop on prod_out   (M1 behind a producer)
        plain_fi  prod(plain) -> fi_call(prod_out)            (today's path behind a producer)
        prod      prod(plain) only, reading its own previous output (subtrahend)
        """
        inst = self._instance("ll", m)
        outs = self._outputs((f"chain-{name}", m, order), m)
        src = outs["norm"]
        if name == "prod":
            src = outs["prod"]
        src.copy_(x)
        pp = self._params(inst, m=m, outs=outs, x=outs["prod"], residual=residual, gamma=gamma,
                          scale=scale, prod_src=src, coeff=1.0)
        if name == "push":
            return lambda: (self._launch(V_PROD_PUSH, 0, pp), self._launch(V_CONSUME, order, pp))
        if name == "push_uni":
            return lambda: (self._launch(V_PROD_PUSH_UNI, 0, pp), self._launch(V_CONSUME, order, pp))
        if name == "plain_one":
            return lambda: (self._launch(V_PROD_PLAIN, 0, pp), self._launch(V_ONE_LL, order, pp))
        if name == "plain_fi":
            return lambda: (self._launch(V_PROD_PLAIN, 0, pp), fi_call(outs["prod"], outs["norm"]))
        if name == "prod":
            return lambda: self._launch(V_PROD_PLAIN, 0, pp)
        if name in ("fused", "fused_nq"):
            # Own mailbox: the chain always leaves one pushed, unconsumed hop behind.
            fi = self._instance("ll", m, tag=f"-{name}")
            pf = self._params(fi, m=m, outs=outs, x=outs["prod"], residual=residual, gamma=gamma,
                              scale=scale, prod_src=src, coeff=1.0, quant=name == "fused")
            self._launch(V_PROD_PUSH, 0, pf)  # prime hop 0 for the first fused launch
            return lambda: self._launch(V_FUSED, order, pf)
        raise ValueError(name)

    def pingpong(self, *, peer, iters, unicast, x, residual, gamma, scale, gap_ns=0):
        """One-way latency (ns) between rank 0 and `peer` through the ll mailbox; rank 0 reports."""
        inst = self._instance("ll", 1)
        outs = self._outputs(("pingpong", 1, 0), 1)
        p = self._params(inst, m=1, outs=outs, x=x, residual=residual, gamma=gamma, scale=scale)
        out = torch.zeros(1, dtype=torch.int64, device=self.device)
        # Clear the probe slot (stage 2, token 0, frag 0) on every rank before the bounce.
        inst.mb.tensor[2, :, 0, :8].fill_(0)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        rc = self.lib.mkh_pingpong(ctypes.byref(p), peer, iters, int(unicast),
                                   ctypes.c_void_p(out.data_ptr()),
                                   ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                                   ctypes.c_ulonglong(gap_ns))
        torch.cuda.synchronize()
        inst.mb.tensor[2, :, 0, :8].fill_(-32768)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        if rc != 0:
            raise RuntimeError(f"mkh_pingpong returned {rc}")
        return int(out.item()) / (2 * iters)

    def make_oproj(self, name, *, m, x_fp8, w_packed, k_steps, sx, sw, residual, gamma, scale, out):
        """oproj (GEMM only, writes out) | oproj_push (GEMM + push, then consume) | oproj_push_out."""
        inst = self._instance("ll", m, tag="-oproj")
        outs = self._outputs((name, m, 0), m)
        p = self._params(inst, m=m, outs=outs, x=None, residual=residual, gamma=gamma, scale=scale,
                         quant=False)
        o = OprojArgs(w_packed.data_ptr(), x_fp8.data_ptr(), sx.data_ptr(), sw.data_ptr(),
                      0 if name == "oproj_push" else out.data_ptr())
        stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def gemm(which):
            rc = self.lib.mkh_oproj(which, k_steps, ctypes.byref(p), ctypes.byref(o), stream())
            if rc != 0:
                raise RuntimeError(f"mkh_oproj failed {rc}")

        if name == "oproj":
            return lambda: gemm(0)
        return lambda: (gemm(1), self._launch(V_CONSUME, 1, p))

    def outputs_of(self, name, m, order=0):
        return self.outputs[(name, m, order)]

    def errors(self):
        return {f"{k[0]}@m{k[1]}": int(v.ctrl[3].item()) for k, v in self.instances.items()}
