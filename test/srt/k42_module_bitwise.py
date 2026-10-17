"""K4.2 module-level bitwise check (docs/223): AGG prompt-end k31 publication on the side stream.

Builds a served-shape FactoredGDNPool (R16 m16 -> RMAX 32, fp16 factors, k31 init, strict_chunk + factored_prefix,
36 GDN layers, k = v = 128, 480 + 1 slots, TP1 h 48 or TP2 rank-0 h 24) without a server, prewarms the whole-prefix
k31 batch graph through pool.prewarm_k31_batch_graph(), and compares per rep, from identical pool bytes:
  F  today: forward stand-in work, prompt-end states written, graph.run on the current stream, post work, reader;
  S  K4.2:  same, but PDBatchPublication(agg, pool_route).submit + start_after_forward (side stream, after the
            "forward" event), post work overlaps it, pool.pside_join() makes the current stream wait, then the reader.
The reader is factored_packed_decode on all 36 layers for the target slots and the final-copy destinations.
Every pool tensor (a, U, W, count, stale, dense_of, prefix_valid, dense_required, dense_ring) and every reader output
is compared bitwise. Exit 0 only if every S-vs-F comparison is equal. Diagnostics (never gate): F-vs-F determinism
baseline; --negative-control = S without pside_join before the reader (and no post work), i.e. a missing join.

Run in the sglang container on one GPU from the fork worktree root:
  PYTHONPATH=python python3 test/srt/k42_module_bitwise.py --heads 24 --batch 1,2,4 --reps 3 --out k42.json

--route E (P1, docs/229 §3 (i)): the AGG full-N publication (compact FullNWorkspace + whole-prefix PrefillBatchGraph
prewarmed as gdn_prefill_agg_contract.prewarm does), driven through the real BatchCollector and
pool.commit_extend_batched by a 36-layer forward stand-in (per-layer work, one producer buffer reused by every layer,
the workspace snapshot in add, then a ~2 ms tail whose output stands in for the logits). Each rep is two back-to-back
extends (slab lifetime: the second extend's snapshots overwrite the first's slab rows), then the 36-layer reader:
  L   today (K4.2): launched after the tail (publication.start_after_forward); the runner hook joins before a forward;
  E   P1: launched from the last layer's add (controls bound on the first add); start_after_forward is a no-op;
  N   negative control: E with the fork event recorded after layer 45's add (before layer 46's snapshot).
Pass = on every rep E vs L bitwise on a / U / W / count / stale / dense_of / prefix_valid / dense_required / dense_ring,
the reader output and both tail outputs, and N detected; L vs L2 is the determinism diagnostic. States come from
--dumps (k31 state dumps; h48 dumps split into two h24 halves) or are synthetic; untracked and tracked variants run in
one invocation. The served env (K31_FUSED / MIXED_EIGH_ANY_BATCH / K42_FAST_BIND ...) comes from the job.
  PYTHONPATH=python python3 test/srt/k42_module_bitwise.py --route E --heads 24 --batch 1,2,4 --reps 3 \\
      --dumps '/work/.../k31dump-r0-*.pt' --out k42e.json
"""
import argparse
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace

# Served C+ recipe (docs/223 §3.1); set before sglang reads them. The deferred flag stays off: this script installs
# the publication itself instead of the server's ModelRunner hook.
os.environ.setdefault("SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31", "1")
os.environ.setdefault("SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31_MAX_BATCH", "4")
os.environ.setdefault("SGLANG_GDN_K31_MIXED_EIGH", "1")
os.environ.setdefault("SGLANG_GDN_K31_CHOLQR_MIXED", "1")
os.environ.pop("SGLANG_GDN_AGG_BATCH_PUBLISH_DEFERRED", None)
os.environ.pop("SGLANG_GDN_K42_CONTROL", None)

import torch  # noqa: E402

from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf  # noqa: E402
from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402
from sglang.srt.mem_cache.gdn_pd_publication import CudaPublicationRuntime, PDBatchPublication  # noqa: E402

DEV = "cuda"
K = V = 128
GDN_LAYERS = [i for i in range(48) if i % 4 != 3]  # 36 GDN layers of the 48-layer hybrid
HEADS_TOTAL = 48
SIZE = 480  # pool slots = SIZE + 1 = 481


def build_pool(h, tmpdir, vbar_source=None, pool_route=True):
    gen = torch.Generator(device="cpu").manual_seed(4242)
    vbar = {lid: torch.randn(HEADS_TOTAL, V, generator=gen) * 0.1 for lid in GDN_LAYERS}
    if vbar_source is not None:  # (36, H, V) of a served dump, tiled to the 48 model heads
        tiled = vbar_source.float().repeat(1, HEADS_TOTAL // vbar_source.shape[1], 1)
        vbar = {lid: tiled[i].clone() for i, lid in enumerate(GDN_LAYERS)}
    path = os.path.join(tmpdir, "consts.pt")
    torch.save({"vbar": vbar}, path)
    cfg = gp.FactoredGDNConfig.parse(
        f"r=16,m=16,dtype=fp16,ring=16,init_method=k31,decode_method=iter,strict_chunk=1,factored_prefix=1,vbar={path}")
    params = SimpleNamespace(shape=SimpleNamespace(temporal=(h, V, K)))
    pool = gp.FactoredGDNPool(size=SIZE, cache_params=params, mamba_layer_ids=list(GDN_LAYERS), device=DEV,
                              cfg=cfg, tp_rank=0)
    graph = None
    if pool_route:
        pool.prewarm_k31_batch_graph()
        graph = pool._k31_batch_graph
        if graph is None or not graph.warmed or graph.include_tail:
            raise SystemExit("prewarm_k31_batch_graph did not build a warmed include_tail=False graph "
                             f"(k31_graph_safe={gp.k31_graph_safe(pool.device)}, K31_EIGH={gp.K31_EIGH})")
    # non-trivial bytes everywhere so untouched slots and partial writes are visible
    g = torch.Generator(device=DEV).manual_seed(7)
    pool.a.copy_(torch.randn(pool.a.shape, generator=g, device=DEV) * 0.05)
    for t in (pool.U, pool.W):
        for l in range(t.shape[0]):
            t[l].copy_(torch.randn(t[l].shape, generator=g, device=DEV) * 0.1)
    pool.count.copy_(torch.randint(pool.cfg.r, pool.cfg.rfull, pool.count.shape, generator=g, device=DEV))
    pool.dense_ring.copy_(torch.randn(pool.dense_ring.shape, generator=g, device=DEV) * 0.01)
    torch.cuda.synchronize()
    return pool, graph


def pool_tensors(pool):
    out = dict(a=pool.a, U=pool.U, W=pool.W, count=pool.count, stale=pool.stale, dense_of=pool.dense_of,
               prefix_valid=pool.prefix_valid, dense_ring=pool.dense_ring)
    if pool.dense_required is not None:
        out["dense_required"] = pool.dense_required
    return out


def snapshot(pool):
    return {k: v.clone() for k, v in pool_tensors(pool).items()}


def restore(pool, snap):
    for k, v in pool_tensors(pool).items():
        v.copy_(snap[k])
    torch.cuda.synchronize()


def make_rep(pool, b, tracked, seed):
    """Prompt-end dense states (low rank + noise), plan, controls and decode-reader inputs for one rep."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = pool.hv

    def states():
        rank = 24
        sv = 2.0 * 0.8 ** torch.arange(rank, device=DEV, dtype=torch.float32)
        u = torch.randn(len(GDN_LAYERS), b, h, V, rank, generator=g, device=DEV) / rank ** 0.5
        w = torch.randn(len(GDN_LAYERS), b, h, rank, K, generator=g, device=DEV) / K ** 0.5
        x = (u * sv) @ w + 1e-3 * torch.randn(len(GDN_LAYERS), b, h, V, K, generator=g, device=DEV)
        return x.contiguous()

    perm = torch.randperm(SIZE, generator=g, device=DEV)[:3 * b] + 1  # distinct live slots (0 = padding)
    slots, extra, final_dst = perm[:b].long(), perm[b:2 * b].long(), perm[2 * b:].long()
    src = dict(normal=states(), tracked=states() if tracked else None)
    ring_dst = torch.where(torch.arange(b, device=DEV) % 2 == 0, torch.randperm(pool.cfg.ring, generator=g,
                           device=DEV)[:b], torch.full((b,), -1, device=DEV)).long()
    plan = SimpleNamespace(slots=slots, ring_dst=ring_dst, dense_required_after_commit=torch.randint(
        0, 2, (b,), generator=g, device=DEV, dtype=torch.int32))
    controls = (extra if tracked else None, slots.clone(), final_dst)  # track_slots, final_src, final_dst
    read_slots = torch.cat([slots, final_dst]).to(torch.int32)
    B, Hq = read_slots.numel(), h // 3
    reader = dict(
        slots=read_slots,
        mixed=(torch.randn(B, 2 * Hq * K + h * V, generator=g, device=DEV) * 0.5).to(torch.bfloat16),
        a=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16),
        b=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16),
        A_log=(torch.rand(len(GDN_LAYERS), h, generator=g, device=DEV) * 2 - 4).float(),
        dt_bias=torch.randn(len(GDN_LAYERS), h, generator=g, device=DEV).float(),
    )
    return src, plan, controls, reader


def run_reader(pool, rd):
    L, B, h = len(GDN_LAYERS), rd["slots"].numel(), pool.hv
    out = torch.empty(L, B, 1, h, V, dtype=torch.bfloat16, device=DEV)
    for l in range(L):
        gf.factored_packed_decode(
            rd["mixed"], rd["a"], rd["b"], A_log=rd["A_log"][l], dt_bias=rd["dt_bias"][l], scale=K ** -0.5,
            vbar=pool.vbar[l], fa=pool.a[l], fu=pool.U[l], fw=pool.W[l], fcount=pool.count[l], stale=pool.stale,
            ssm_state_indices=rd["slots"], num_q_heads=h // 3, num_v_heads=h, head_k_dim=K, head_v_dim=V,
            r=pool.cfg.r, rfull=pool.cfg.rfull, out=out[l], truncate=True, **pool.cfg.kernel_kwargs())
    return out


class StandIn:
    """Logits/sampler stand-in on the current stream (large bf16 matmuls), so the side stream truly overlaps."""

    def __init__(self, n, iters):
        g = torch.Generator(device=DEV).manual_seed(3)
        self.x = torch.randn(n, n, generator=g, device=DEV).to(torch.bfloat16)
        self.y = torch.empty_like(self.x)
        self.iters = iters

    def __call__(self):
        for _ in range(self.iters):
            torch.matmul(self.x, self.x, out=self.y)


def variant(kind, pool, graph, src, plan, controls, reader, live, standin, runtime):
    """kind: F (today), S (K4.2 + join), N (K4.2 without the reader join). Returns reader out, elapsed ms."""
    eager = gp.factorize_layers
    policy = (gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE, gp.factorize_dense)
    for t in live.values():  # garbage until the "forward" writes them: catches a side stream that does not wait
        if t is not None:
            t.zero_()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    standin()  # rest of the forward
    for k, t in live.items():
        if t is not None:
            t.copy_(src[k])  # last GDN layer outputs the prompt-end states on the forward stream
    tracked = live["tracked"]
    states = [(live["normal"][l], None if tracked is None else tracked[l]) for l in range(len(GDN_LAYERS))]
    publication = None
    if kind == "F":
        pool._pd_batch_publication = None
        graph.run(pool, plan, states, *controls, eager=eager, policy=policy)
        standin()
    else:
        publication = PDBatchPublication(pool, runtime, agg=True, pool_route=True)
        pool._pd_batch_publication = publication
        publication.submit(graph, plan, states, controls, eager=eager, policy=policy)
        publication.start_after_forward()  # what the ModelRunner.forward hook does after logits
        if kind == "S":
            standin()
            pool.pside_join()  # must make the current stream wait for the publication
            if publication.pending is not None:
                raise RuntimeError("pside_join left the publication pending")
    out = run_reader(pool, reader)
    t1.record()
    torch.cuda.synchronize()
    if publication is not None:
        publication.join()  # N: clear the pending ticket after the device sync
        pool._pd_batch_publication = None
    return out, t0.elapsed_time(t1)


def bits(t):
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def compare(x, y):
    eq = torch.equal(bits(x), bits(y))
    res = dict(equal=eq, max_abs=0.0, n_diff=0, numel=x.numel())
    if not eq:  # chunk over dim 0: U/W are 7 GB at TP1, a full fp64 diff would not fit comfortably
        for xi, yi in zip(x, y) if x.dim() > 1 else ((x, y),):
            d = (xi.double() - yi.double()).abs().nan_to_num(nan=float("inf"))
            res["max_abs"] = max(res["max_abs"], float(d.max()))
            res["n_diff"] += int((bits(xi) != bits(yi)).sum())
    return res


def compare_all(ref_pool, ref_out, pool, out):
    rows = {k: compare(ref_pool[k], v) for k, v in pool_tensors(pool).items()}
    rows["reader_out"] = compare(ref_out, out)
    return rows


# ------------------------------------------------------------------------------------------------ route E (P1)
CAPACITY = 4  # compact workspace rows (bucket 4 = the largest --batch)
CHUNK_TOKENS = 16384
TOKENS = 8192  # per extend; rows <= min(CAPACITY, TOKENS // 2)


class ForkRuntime(CudaPublicationRuntime):
    """CudaPublicationRuntime whose next launch can wait on an earlier, armed fork event (negative control)."""

    def __init__(self, device):
        super().__init__(device)
        self.fork = None

    def arm(self):
        self.fork = torch.cuda.Event()
        self.fork.record(torch.cuda.current_stream(self.device))

    def launch_after_forward(self, operation, inputs):
        if self.fork is None:
            return super().launch_after_forward(operation, inputs)
        forward_done, self.fork = self.fork, None
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(forward_done)
            try:
                operation()
            finally:
                for tensor in inputs:
                    if tensor is not None and tensor.is_cuda:
                        tensor.record_stream(self.stream)
                done.record(self.stream)
        return done


def build_fulln_graph(pool):
    from sglang.srt.mem_cache.gdn_fulln_workspace import FullNWorkspace
    from sglang.srt.mem_cache.gdn_prefill_batch_graph import PrefillBatchGraph

    workspace = FullNWorkspace(pool, CAPACITY, CHUNK_TOKENS)
    graph = PrefillBatchGraph(include_tail=False, workspace=workspace, shared=workspace.shared)
    graph.prewarm(pool, eager=gp.factorize_layers, policy=(gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE, gp.factorize_dense))
    return graph, workspace


def load_events(patterns, h):
    """Served prompt-end states as h-head rows (36, h, V, K) on the CPU; h48 dumps give two h24 halves."""
    import glob

    events, vbar = [], None
    for path in sorted(p for pattern in patterns for p in glob.glob(pattern)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        heads = payload["S"].shape[2]
        if heads % h:
            raise SystemExit(f"{path}: {heads} heads do not split into {h}")
        for half in range(heads // h):
            sl = slice(half * h, (half + 1) * h)
            if vbar is None:
                vbar = payload["vbar"][:, sl]
            for row in range(payload["S"].shape[0]):
                tracked = payload.get("S_tracked")
                events.append(dict(name=f"{os.path.basename(path)}:h{half}:r{row}", S=payload["S"][row, :, sl],
                                   T=None if tracked is None else tracked[row, :, sl]))
    return events, vbar


class EventSource:
    def __init__(self, events, seed, h):
        self.events, self.cursor, self.h = events, 0, h
        self.gen = torch.Generator(device=DEV).manual_seed(seed)

    def synthetic(self):
        rank, h = 24, self.h
        sv = 2.0 * 0.8 ** torch.arange(rank, device=DEV, dtype=torch.float32)
        u = torch.randn(len(GDN_LAYERS), h, V, rank, generator=self.gen, device=DEV) / rank ** 0.5
        w = torch.randn(len(GDN_LAYERS), h, rank, K, generator=self.gen, device=DEV) / K ** 0.5
        return (u * sv) @ w + 1e-3 * torch.randn(len(GDN_LAYERS), h, V, K, generator=self.gen, device=DEV)

    def rows(self, n, tracked=False):
        """n rows of (36, h, V, K); tracked rows prefer a dump's own tracked checkpoint."""
        out, names = [], []
        for _ in range(n):
            if not self.events:
                out.append(self.synthetic())
                names.append("synthetic")
                continue
            event = self.events[self.cursor % len(self.events)]
            self.cursor += 1
            source = event["T"] if tracked and event["T"] is not None else event["S"]
            out.append(source.to(DEV, torch.float32))
            names.append(event["name"] + (":T" if tracked and event["T"] is not None else ""))
        return torch.stack(out, dim=1).contiguous(), names  # (36, n, h, V, K)


def make_extend(pool, source, b, tracked_rows, slots, extra, final_dst, gen):
    src, names = source.rows(b)
    tsrc, tnames = source.rows(tracked_rows, tracked=True) if tracked_rows else (None, [])
    ring_dst = torch.where(torch.arange(b, device=DEV) % 2 == 0, torch.randperm(pool.cfg.ring, generator=gen,
                           device=DEV)[:b], torch.full((b,), -1, device=DEV)).long()
    required = torch.randint(0, 2, (b,), generator=gen, device=DEV, dtype=torch.int32)
    controls = (extra[:tracked_rows] if tracked_rows else None, slots.clone(), final_dst)
    return dict(b=b, src=src, tsrc=tsrc, slots=slots, ring_dst=ring_dst, required=required, controls=controls,
                events=names + tnames)


def make_plan(ext):
    b = ext["b"]
    zeros = torch.zeros(b, dtype=torch.long, device=DEV)
    return gp.FactoredExtendPlan(slots=ext["slots"], use_ring=torch.zeros(b, dtype=torch.bool, device=DEV),
                                 ring_src=zeros, ring_dst=ext["ring_dst"],
                                 ring_dst_rows=torch.nonzero(ext["ring_dst"] >= 0).flatten(),
                                 dense_required_after_commit=ext["required"], next_layer=0,
                                 last_layer=len(GDN_LAYERS) - 1)


class ForwardStandIn:
    """Per-layer GDN work, a spin before the last layer's state (so a fork recorded too early has a window on any
    GPU), and the trunk tail after layer 46 (out-proj / MoE / layer 47 / lm_head, ~2 ms) whose output is compared."""

    def __init__(self, layer_n, tail_n, tail_iters, spin_cycles):
        g = torch.Generator(device=DEV).manual_seed(5)
        self.lx = torch.randn(layer_n, layer_n, generator=g, device=DEV).to(torch.bfloat16)
        self.ly = torch.empty_like(self.lx)
        self.tx = torch.randn(tail_n, tail_n, generator=g, device=DEV).to(torch.bfloat16)
        self.ty = torch.empty_like(self.tx)
        self.tail_iters, self.spin_cycles = tail_iters, spin_cycles

    def layer(self, last):
        torch.matmul(self.lx, self.lx, out=self.ly)
        if last and self.spin_cycles:
            torch.cuda._sleep(self.spin_cycles)

    def tail(self):
        for _ in range(self.tail_iters):
            torch.matmul(self.tx, self.tx, out=self.ty)
            self.tx.add_(self.ty, alpha=1e-6)  # the next forward's tail differs
        return self.ty.clone()


def run_forwards(kind, pool, graph, extends, reader, standin, runtime, producers):
    """kind L / E / N over back-to-back extends, then the reader. Returns reader out, tail outputs, ms, stats."""
    from sglang.srt.mem_cache.gdn_prefill_batch_graph import BatchCollector

    publication = PDBatchPublication(pool, runtime, agg=True)
    pool._pd_batch_publication = publication
    standin.tx.copy_(standin.tx0)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    tails = []
    for ext in extends:
        pool.pside_join()  # install_forward_join: unconditional join before every runner.forward
        plan = make_plan(ext)
        b, normal, tracked = ext["b"], producers["normal"], producers["tracked"]
        collector = BatchCollector(pool, plan, graph=graph, token_count=TOKENS, publication=publication,
                                   launch_at_last_layer=kind in ("E", "N"))
        with collector:
            for i, lid in enumerate(GDN_LAYERS):
                last = i == len(GDN_LAYERS) - 1
                standin.layer(last)
                normal[:b].copy_(ext["src"][i])  # the layer's prompt-end state, same storage every layer
                t = None
                if ext["tsrc"] is not None:
                    t = tracked[:ext["tsrc"].shape[1]]
                    t.copy_(ext["tsrc"][i])
                pool.commit_extend_batched(lid, plan, normal[:b], t, *ext["controls"])
                if kind == "N" and i == len(GDN_LAYERS) - 2:
                    runtime.arm()  # fork event after layer 45's add, before layer 46's state
            tails.append(standin.tail())
        if collector.published is not True:
            raise RuntimeError("collector did not publish")
        publication.start_after_forward()  # gdn_prefill_agg_contract.forward after logits
    pool.pside_join()
    out = run_reader(pool, reader)
    t1.record()
    torch.cuda.synchronize()
    stats = dict(publication.stats)
    pool._pd_batch_publication = None
    return out, tails, t0.elapsed_time(t1), stats


def route_e(args, result):
    patterns = [p for p in (args.dumps or "").split(",") if p]
    events, vbar = load_events(patterns, args.heads) if patterns else ([], None)
    tmpdir = tempfile.mkdtemp(prefix="k42e_bitwise_")
    t_start = time.time()
    pool, _ = build_pool(args.heads, tmpdir, vbar_source=vbar, pool_route=False)
    graph, workspace = build_fulln_graph(pool)
    from sglang.srt.mem_cache import gdn_pd_publication as pubmod

    result["meta"].update(route="E", events=len(events), dumps=patterns, fast_bind=pubmod._FAST_BIND,
                          graph_entries=sorted(str(k[:2]) for k in graph.entries), capacity=CAPACITY,
                          spin_cycles=args.spin_cycles, device=torch.cuda.get_device_name(),
                          cfg=pool.cfg.raw, pool_file=gp.__file__,
                          env={k: v for k, v in os.environ.items() if k.startswith("SGLANG_GDN")})
    standin = ForwardStandIn(args.layer_n, args.standin_n, args.standin_iters, args.spin_cycles)
    standin.tx0 = standin.tx.clone()
    runtime = ForkRuntime(pool.a.device)
    producers = dict(normal=torch.empty(CAPACITY, pool.hv, V, K, device=DEV),
                     tracked=torch.empty(CAPACITY, pool.hv, V, K, device=DEV))
    source = EventSource(events, args.seed, pool.hv)
    print(f"pool + full-N graph prewarm ready in {time.time() - t_start:.0f}s on {result['meta']['device']}; "
          f"{len(events)} event rows; entries {result['meta']['graph_entries']}", flush=True)

    def extends_for(b, tracked, rep, seed):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        perm = torch.randperm(SIZE, generator=gen, device=DEV)[:6 * b] + 1  # distinct live slots (0 = padding)
        tracked_rows = 0 if not tracked else (b if rep % 2 == 0 else 1)
        return [make_extend(pool, source, b, tracked_rows, perm[k * 3 * b:k * 3 * b + b].long(),
                            perm[k * 3 * b + b:k * 3 * b + 2 * b].long(),
                            perm[k * 3 * b + 2 * b:(k + 1) * 3 * b].long(), gen) for k in range(2)]

    def reader_for(extends, seed):
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        slots = [e["slots"] for e in extends] + [e["controls"][2] for e in extends]
        slots += [e["controls"][0] for e in extends if e["controls"][0] is not None]
        read_slots = torch.cat(slots).to(torch.int32)
        B, h, Hq = read_slots.numel(), pool.hv, pool.hv // 3
        return dict(slots=read_slots,
                    mixed=(torch.randn(B, 2 * Hq * K + h * V, generator=g, device=DEV) * 0.5).to(torch.bfloat16),
                    a=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16),
                    b=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16),
                    A_log=(torch.rand(len(GDN_LAYERS), h, generator=g, device=DEV) * 2 - 4).float(),
                    dt_bias=torch.randn(len(GDN_LAYERS), h, generator=g, device=DEV).float())

    # warm-up (reader JIT, side stream, first replays); discarded
    init = snapshot(pool)
    extends = extends_for(1, True, 0, args.seed)
    for kind in ("L", "E"):
        run_forwards(kind, pool, graph, extends, reader_for(extends, args.seed), standin, runtime, producers)
        restore(pool, init)
    del init, extends
    ok = True
    for tracked in (False, True):
        for b in [int(x) for x in args.batch.split(",") if x]:
            for rep in range(args.reps):
                seed = args.seed * 100003 + b * 1009 + rep * 7 + int(tracked)
                extends = extends_for(b, tracked, rep, seed)
                reader = reader_for(extends, seed)
                init = snapshot(pool)
                row = dict(batch=b, tracked=tracked, rep=rep, seed=seed, events=[e["events"] for e in extends],
                           tracked_rows=[0 if e["tsrc"] is None else e["tsrc"].shape[1] for e in extends], ms={})
                runs = {}
                for kind in ("L", "L2", "E", "N"):
                    restore(pool, init)
                    out, tails, row["ms"][kind], stats = run_forwards(kind[0], pool, graph, extends, reader, standin,
                                                                      runtime, producers)
                    runs[kind] = (snapshot(pool) if kind == "L" else None, out, tails, stats)
                    if kind != "L":
                        ref, ref_out, ref_tails, _ = runs["L"]
                        rows = compare_all(ref, ref_out, pool, out)
                        for i, (x, y) in enumerate(zip(ref_tails, tails)):
                            rows[f"tail_out_{i}"] = compare(x, y)
                        row[f"{kind}_vs_L"] = rows
                    row[f"stats_{kind}"] = {k: v for k, v in stats.items() if v}
                row["publication_wrote"] = not torch.equal(bits(init["U"][:, extends[1]["slots"]]),
                                                           bits(runs["L"][0]["U"][:, extends[1]["slots"]]))
                row["E_equal"] = all(v["equal"] for v in row["E_vs_L"].values())
                row["L2_equal"] = all(v["equal"] for v in row["L2_vs_L"].values())
                row["negative_control_detected"] = not all(v["equal"] for v in row["N_vs_L"].values())
                row["early_launches"] = row["stats_E"].get("launched_at_last_layer", 0)
                row["pass"] = (row["E_equal"] and row["negative_control_detected"] and row["publication_wrote"]
                               and row["early_launches"] == len(extends)
                               and not row["stats_E"].get("joined_in_forward"))
                ok &= row["pass"]
                restore(pool, init)
                del init, runs
                torch.cuda.empty_cache()
                bad = [k for k, v in row["E_vs_L"].items() if not v["equal"]]
                print(f"tracked={int(tracked)} b{b} rep{rep}: E_vs_L {'EQUAL' if row['E_equal'] else 'DIFF ' + ','.join(bad)}"
                      f" | L2_vs_L {'equal' if row['L2_equal'] else 'DIFF'}"
                      f" | negative control detected={row['negative_control_detected']}"
                      f" | early launches {row['early_launches']}"
                      " | ms " + " ".join(f"{k}={v:.3f}" for k, v in row["ms"].items()), flush=True)
                result["reps"].append(row)
                result["pass"] = ok
                with open(args.out, "w") as f:
                    json.dump(result, f, indent=1)
    result["pass"] = ok
    result["meta"]["wall_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{'PASS' if ok else 'FAIL'}: route E {len(result['reps'])} reps, result {args.out}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=int, choices=(24, 48), default=24)
    ap.add_argument("--batch", default="1,2,4", help="comma list of prompt-end rows per forward (<= 4)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=223)
    ap.add_argument("--tracked", action="store_true", help="also publish a tracked (stale=1) checkpoint branch")
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--standin-n", type=int, default=8192)
    ap.add_argument("--standin-iters", type=int, default=6)
    ap.add_argument("--route", choices=("S", "E"), default="S",
                    help="S = K4.2 pool route (default, docs/223 §3.6); E = P1 launch at the last layer (docs/229)")
    ap.add_argument("--dumps", default="", help="route E: comma list of k31 state dump globs (default synthetic)")
    ap.add_argument("--layer-n", type=int, default=2048, help="route E: per-layer stand-in matmul size")
    ap.add_argument("--spin-cycles", type=int, default=2_000_000,
                    help="route E: torch.cuda._sleep before the last layer's state (all variants)")
    args = ap.parse_args()
    batches = [int(x) for x in args.batch.split(",") if x]
    if any(not 1 <= b <= 4 for b in batches):
        raise SystemExit("--batch values must be in 1..4 (k31 graph MAX_BATCH 4)")
    torch.manual_seed(args.seed)
    if args.route == "E":
        return route_e(args, dict(meta=dict(argv=vars(args), torch=torch.__version__,
                                            orth=(gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE)), reps=[]))
    tmpdir = tempfile.mkdtemp(prefix="k42_bitwise_")
    t_start = time.time()
    pool, graph = build_pool(args.heads, tmpdir)
    standin = StandIn(args.standin_n, args.standin_iters)
    runtime = CudaPublicationRuntime(pool.a.device)
    result = dict(meta=dict(argv=vars(args), device=torch.cuda.get_device_name(), torch=torch.__version__,
                            pool_file=gp.__file__, orth=(gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE),
                            cfg=pool.cfg.raw, kernel_kwargs={k: str(v) for k, v in pool.cfg.kernel_kwargs().items()},
                            graph_entries=sorted(str(k[:2]) for k in graph.entries),
                            env={k: v for k, v in os.environ.items() if k.startswith("SGLANG_GDN")}),
                  reps=[])
    # warm-up (Triton JIT / autotune of the reader, side-stream creation); discarded
    init = snapshot(pool)
    src, plan, controls, reader = make_rep(pool, batches[0], args.tracked, args.seed)
    live = {k: None if v is None else torch.empty_like(v) for k, v in src.items()}
    for kind in ("F", "S"):
        variant(kind, pool, graph, src, plan, controls, reader, live, standin, runtime)
        restore(pool, init)
    del init, src, live
    ok = True
    print(f"pool + k31 graph prewarm ready in {time.time() - t_start:.0f}s on {result['meta']['device']}", flush=True)
    for b in batches:
        for rep in range(args.reps):
            seed = args.seed * 100003 + b * 1009 + rep
            src, plan, controls, reader = make_rep(pool, b, args.tracked, seed)
            live = {k: None if v is None else torch.empty_like(v) for k, v in src.items()}
            init = snapshot(pool)
            row = dict(batch=b, rep=rep, seed=seed, timing_ms={})
            out_f, row["timing_ms"]["F"] = variant("F", pool, graph, src, plan, controls, reader, live, standin, runtime)
            ref = snapshot(pool)
            restore(pool, init)
            out_f2, row["timing_ms"]["F2"] = variant("F", pool, graph, src, plan, controls, reader, live, standin,
                                                     runtime)
            row["F_vs_F"] = compare_all(ref, out_f, pool, out_f2)
            restore(pool, init)
            out_s, row["timing_ms"]["S"] = variant("S", pool, graph, src, plan, controls, reader, live, standin, runtime)
            row["S_vs_F"] = compare_all(ref, out_f, pool, out_s)
            # the publication must have written something: the target slots differ from the initial bytes
            row["publication_wrote"] = not torch.equal(bits(init["U"][:, plan.slots]), bits(pool.U[:, plan.slots]))
            row["pass"] = all(v["equal"] for v in row["S_vs_F"].values()) and row["publication_wrote"]
            ok &= row["pass"]
            if args.negative_control:
                restore(pool, init)
                out_n, row["timing_ms"]["N"] = variant("N", pool, graph, src, plan, controls, reader, live, standin,
                                                       runtime)
                row["N_vs_F"] = compare_all(ref, out_f, pool, out_n)
                row["negative_control_detected"] = not all(v["equal"] for v in row["N_vs_F"].values())
            restore(pool, init)  # every rep starts from the same pool bytes
            del init, ref
            torch.cuda.empty_cache()
            bad = [k for k, v in row["S_vs_F"].items() if not v["equal"]]
            print(f"b{b} rep{rep}: S_vs_F {'EQUAL' if row['pass'] else 'DIFF ' + ','.join(bad)}"
                  f" | F_vs_F {'equal' if all(v['equal'] for v in row['F_vs_F'].values()) else 'DIFF'}"
                  + (f" | negative control detected={row['negative_control_detected']}"
                     if args.negative_control else "")
                  + " | ms " + " ".join(f"{k}={v:.3f}" for k, v in row["timing_ms"].items()), flush=True)
            result["reps"].append(row)
            result["pass"] = ok
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
    result["pass"] = ok
    result["meta"]["wall_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{'PASS' if ok else 'FAIL'}: {len(result['reps'])} reps, result {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
