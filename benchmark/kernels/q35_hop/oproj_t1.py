"""Single-GPU timing of the Qwen3.5 GatedDeltaNet out_proj at bs1 verify sizes.

Arms: the served path (flashinfer bmm_fp8 through apply_fp8_linear_bmm_flashinfer) and the two
FP8 GEMV cores in oproj_t1.cu, alone and behind the served gated-RMSNorm FP8 producer
(_layer_norm_fwd), in CUDA graphs. Weight banks rotate so every call reads its weights cold.
Per-call time is the slope between graphs of N1 and N2 calls; one JSON line per measurement.
"""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
N = 4096
HEAD_V = 128
L2_BYTES = 126 * 2**20


class T1Args(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_void_p),
        ("x", ctypes.c_void_p),
        ("sx", ctypes.c_void_p),
        ("sw", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("box", ctypes.c_void_p),
        ("m", ctypes.c_int),
    ]


def emit(**kv):
    print(json.dumps(kv), flush=True)


def build():
    src = os.path.join(HERE, "oproj_t1.cu")
    digest = hashlib.md5(open(src, "rb").read()).hexdigest()[:12]
    out_dir = os.environ.get("OPROJ_T1_BUILD_DIR", "/tmp/oproj-t1-build")
    os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, f"liboproj_t1.{digest}.so")
    if not os.path.exists(so):
        tmp = f"{so}.{os.getpid()}.tmp"
        subprocess.check_call(
            ["nvcc", "-O3", "-std=c++17", "-arch=sm_103a", "-shared", "-Xcompiler", "-fPIC", "-o", tmp, src]
        )
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.t1_oproj.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    assert lib.t1_args_size() == ctypes.sizeof(T1Args)
    emit(event="build", digest=digest)
    return lib


def pack_cluster(w_nk, units, steps):
    # Same layout as the fused attention epilogue: [unit][rank 8][warp 8][step][block 2][lane 32][16 B].
    w = w_nk.contiguous().view(torch.uint8)
    v = w.view(units, 8, 2, 2, 8, 8, steps, 2, 4, 4)
    return v.permute(0, 5, 1, 6, 2, 4, 8, 7, 3, 9).contiguous().view(-1)


class Shape:
    def __init__(self, k, m, banks, dev, seed):
        from sglang.srt.layers.q35_hop import pack_out_proj

        g = torch.Generator(device="cpu").manual_seed(seed)
        self.k, self.m, self.banks = k, m, banks
        self.nv = k // HEAD_V
        self.sx = torch.tensor([0.02], dtype=torch.float32, device=dev)
        self.sw = torch.tensor([0.004], dtype=torch.float32, device=dev)
        self.w_nk, self.w_kn, self.w_rows32, self.w_cluster = [], [], [], []
        for _ in range(banks):
            w = (torch.randn(N, k, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn).to(dev)
            self.w_nk.append(w)
            self.w_kn.append(w.t())  # served layout: [K, N] column-major view
            self.w_rows32.append(pack_out_proj(w))
            self.w_cluster.append(pack_cluster(w, 16, k // 256))
        rows = m * self.nv
        self.core = [torch.randn(rows, HEAD_V, generator=g).to(torch.bfloat16).to(dev) for _ in range(2)]
        self.z = [torch.randn(rows, HEAD_V, generator=g).to(torch.bfloat16).to(dev) for _ in range(2)]
        self.gamma = (1.0 + 0.1 * torch.randn(HEAD_V, generator=g)).to(torch.bfloat16).to(dev)
        xb = torch.randn(m, k, generator=g) * 8
        self.x = (xb / 0.02).clamp(-448, 448).to(torch.float8_e4m3fn).to(dev)
        self.out = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
        self.box = torch.empty(m, N, dtype=torch.bfloat16, device=dev)


def producer(sh, i, early=False):
    from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd

    kw = {"early_dependents": True} if early else {}
    (fp8, _), _, _ = _layer_norm_fwd(
        sh.core[i % 2],
        sh.gamma,
        None,
        1e-6,
        z=sh.z[i % 2],
        is_rms_norm=True,
        activation="swish",
        static_quant_scale=sh.sx,
        **kw,
    )
    return fp8.reshape(sh.m, sh.k)


def served(sh, x, i):
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer

    b = i % sh.banks
    return apply_fp8_linear_bmm_flashinfer(
        input=x,
        weight=sh.w_kn[b],
        weight_scale=sh.sw,
        input_scale=sh.sx,
        pre_quant_output_dtype=torch.bfloat16,
    )


def core_call(lib, sh, core, push, x, i, out=True):
    b = i % sh.banks
    w = (sh.w_rows32 if core == 0 else sh.w_cluster)[b]
    a = T1Args(
        w.data_ptr(), x.data_ptr(), sh.sx.data_ptr(), sh.sw.data_ptr(),
        sh.out.data_ptr() if out else 0, sh.box.data_ptr() if push else 0, sh.m,
    )
    rc = lib.t1_oproj(core, sh.k // 256, push, ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"t1_oproj core={core} rc={rc}")
    return sh.out


def arm_step(lib, sh, arm):
    # Returns step(i) that enqueues one layer's worth of work for this arm.
    if arm == "producer":
        return lambda i: producer(sh, i)
    chain, _, name = arm.partition(":")
    use_prod = chain == "chain"

    def step(i):
        x = producer(sh, i, early=name.endswith("_early")) if use_prod else sh.x
        if name == "served":
            served(sh, x, i)
        elif name == "rows32":
            core_call(lib, sh, 0, 0, x, i)
        elif name in ("rows32_push", "rows32_push_early"):
            core_call(lib, sh, 0, 1, x, i)
        elif name == "cluster":
            core_call(lib, sh, 1, 0, x, i)
        elif name == "cluster_push":
            core_call(lib, sh, 1, 1, x, i)
        else:
            raise ValueError(arm)

    return step


def slope(step, n1, n2, replays, stream):
    graphs = {}
    for n in (n1, n2):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for i in range(n):
                step(i)
        graphs[n] = g
    for n in (n1, n2):
        for _ in range(3):
            graphs[n].replay()
    torch.cuda.synchronize()
    t = {n1: [], n2: []}
    for _ in range(replays):
        for n in (n1, n2):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(stream)
            graphs[n].replay()
            e1.record(stream)
            torch.cuda.synchronize()
            t[n].append(e0.elapsed_time(e1) * 1e3)
    a, b = statistics.median(t[n1]), statistics.median(t[n2])
    return (b - a) / (n2 - n1), a, b, (max(t[n2]) - min(t[n2]))


def check(lib, sh):
    x = sh.x
    ref = (x.double() @ sh.w_nk[0].double().t()) * (sh.sx.double() * sh.sw.double())
    s = served(sh, x, 0).double()
    res = {"served_vs_ref_max": float((s - ref).abs().max())}
    for core, name in ((0, "rows32"), (1, "cluster")):
        for push in (0, 1):
            sh.out.fill_(0)
            sh.box.fill_(0)
            core_call(lib, sh, core, push, x, 0).double()
            o = sh.out.double()
            res[f"{name}_push{push}_mismatch_vs_served"] = int((o != s).sum())
            res[f"{name}_push{push}_max_vs_served"] = float((o - s).abs().max())
            res[f"{name}_push{push}_max_vs_ref"] = float((o - ref).abs().max())
            if push:
                res[f"{name}_box_equals_out"] = bool(torch.equal(sh.box, sh.out))
    torch.cuda.synchronize()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--k", default="1024,2048")
    ap.add_argument("--m", default="7,8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--n1", type=int, default=100)
    ap.add_argument("--n2", type=int, default=500)
    ap.add_argument(
        "--arms",
        default="producer,solo:served,solo:rows32,solo:rows32_push,solo:cluster,solo:cluster_push,"
        "chain:served,chain:rows32_push,chain:cluster_push",
    )
    args = ap.parse_args()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(dev)
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_per_tensor_fp8_supported

    emit(
        event="start", tag=args.tag, gpu=torch.cuda.get_device_name(dev), torch=torch.__version__,
        served_flashinfer_bmm=bool(flashinfer_per_tensor_fp8_supported()), argv=vars(args),
        host=os.uname().nodename, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    lib = build()
    stream = torch.cuda.Stream(device=dev)
    arms = args.arms.split(",")
    for k in (int(v) for v in args.k.split(",")):
        banks = max(48, -(-3 * L2_BYTES // (N * k)))
        for m in (int(v) for v in args.m.split(",")):
            sh = Shape(k, m, banks, dev, seed=1000 + k + m)
            emit(event="check", k=k, m=m, banks=banks, **check(lib, sh))
            for rep in range(args.reps):
                for arm in (arms if rep % 2 == 0 else arms[::-1]):
                    with torch.cuda.stream(stream):
                        per, t1, t2, spread = slope(arm_step(lib, sh, arm), args.n1, args.n2, args.replays, stream)
                    emit(event="t1", tag=args.tag, k=k, m=m, banks=banks, arm=arm, rep=rep,
                         per_call_us=round(per, 4), t_n1_us=round(t1, 2), t_n2_us=round(t2, 2),
                         spread_n2_us=round(spread, 2))
            del sh
            torch.cuda.empty_cache()
    emit(event="done", tag=args.tag)


if __name__ == "__main__":
    main()
