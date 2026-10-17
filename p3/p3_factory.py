"""docs/229 §3 P3 / P5 holder-side workload factory for the P3 line's route_micro.py (contract in its docstring):
`--factory p3_factory:build` builds ONE PD P-role model per TP rank in-process (torch.distributed.run, 2 procs) and returns a
workload with prepare(prefix_tokens, extend_tokens, seed) / forward() / join() / counters().

Real (the served P-role code paths):
  * ServerArgs from the frozen P0 worker argv (fp32-pd j4529207 actual-launch.json: C+ = -Cp for P5, PC+ = -PCp for P3),
    disaggregation mode prefill, --disable-overlap-schedule, --disable-cuda-graph as frozen; environment from
    p3_frozen_env.py (exported by the job before this process starts, so module-level switches see it at import).
  * ModelRunner via sglang.benchmark.one_batch.load_model (alloc pools, attention backends, init_cuda_graphs -> exact-tail +
    PD full-N contract install, k31 dual-graph prewarm under SGLANG_GDN_K31_AB_DUAL=1).
  * Extend batches through ScheduleBatch.init_new -> prepare_for_extend -> ForwardBatch.init_new -> ModelRunner.forward ->
    sample (the scheduler's extend path; the contract hooks on ScheduleBatch / ForwardBatch / backend metadata run).
  * Prefix hits: the prefix is seeded once per prefix length by cold extends (16384-token pieces, each piece a prefix hit
    on the previous one) into slots that stay allocated; each measured request then gets prefix_indices = the seed's KV
    indices and req.kv.mamba_cow_src_index = the seed's mamba slot, i.e. the radix-cache prefix-hit route: ModelRunner
    applies the deferred COW (MambaPool.copy_from -> factored-pool copy_slots, stale = 1 factored-only) at forward start,
    inside the measured forward, as in serving.
The frozen NVLink custom memory pool remains enabled: Flash-Next PD staging reserves its buffers from it before KV sizing,
even though this microbenchmark does not create a sender or register the staging buffers with a transfer engine.
Not exercised / stubbed (recorded in factory.json): no scheduler, no tree cache (a TreeCacheNamespace as in one_batch), so no
radix insertion / track donation of the measured request (common to every route); no bootstrap / KV sender / transfer
engine (FactorStateHandoff.before_send is never called); no HTTP, no D, no
tokenizer manager.  TP group init: SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=env:// -> torch rendezvous on torchrun's
MASTER_ADDR / MASTER_PORT (the agent store torchrun --standalone hosts on a free port per run: both ranks agree, concurrent
runs on one node never collide).  Without it the fork builds tcp://<--host>:<--nccl-port> and, because torchrun sets
TORCHELASTIC_USE_AGENT_STORE, every rank becomes a TCPStore *client* of that address -> nobody serves it -> 600 s timeout
(holder C r2 smokes 10-17 03:03Z, tcp://0.0.0.0:<MASTER_PORT+7>).  --host is also rewritten to 127.0.0.1 and --nccl-port to
MASTER_PORT + 7 (unused by the env:// rendezvous; kept distinct from the frozen 65095).
counters(): 'fallbacks' = full-N summary fallbacks (runner.model._agg_fulln_summary) + emitter-graph fallback_batch +
fallback_capacity (helper twinstar_sgl.pd_emitter_graph.EmitterGraph.stats on owner.pd_emitter_graph; absent on C+ / P5 ->
0, reported emitter_graph=absent), each also as its own key; an explicit route 0n is not a fallback (the contract records no
full-N fallback for shallow 0n and C+ 0n stays full-N); + PD publication stats (pub_*); + last_pub_graph_ms = the publication
graph span of the last forward (+ join): PrefillBatchGraph.run timed with CUDA events -- on the forward stream when it runs
inline (route 0n: forward-stream k31), inside the side-stream launch when it is deferred (route 1: from the side stream's
start after the forward-done wait to the replay end).  The helper CUDA timeline has no publication event, so the span is
measured here; last_pub_graph_ms is a per-forward reading, not monotonic (route_micro records it in every row's after).
Helper staging rule: P3_HELPER_DIR must be a complete helper tree (the shared qualified base, 3e9a55951 blobs, with the P3
line's pd_shallow.py + pd_emitter_graph.py overlaid in its own dir), because it is prepended on PYTHONPATH and the
twinstar_sgl package resolves from the first entry that has it.

Smoke (first holder run, <= 10 min):  torchrun --nproc-per-node 2 p3_factory.py --smoke --stage p5|p3 --out DIR
  -> build, one forward per route on (0, 4096) and on (16384, 1024), counters, smoke-r<rank>.json.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

PIECE = 16384  # frozen --chunked-prefill-size: prefix seeds are extended in pieces of this size
SEED_RID_BASE = 10_000_000


def frozen_argv(command, nccl_port):
    """'… python -m sglang.launch_server ARGS' -> ARGS with the TP NCCL port replaced (only in-process change)."""
    toks = command.split() if isinstance(command, str) else list(command)
    i = toks.index('sglang.launch_server') + 1
    args = toks[i:]
    if '--host' in args:
        args[args.index('--host') + 1] = '127.0.0.1'
    if '--nccl-port' in args:
        args[args.index('--nccl-port') + 1] = str(nccl_port)
    else:
        args += ['--nccl-port', str(nccl_port)]
    return args


def tp_nccl_port():
    return int(os.environ.get('MASTER_PORT', '29500')) + 7


def dist_env(environ):
    """Point the fork's TP init at torchrun's rendezvous (env://); returns the receipt fields.  Raises outside torchrun."""
    missing = [k for k in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE') if not environ.get(k)]
    if missing:
        raise RuntimeError(f'p3_factory must run under torch.distributed.run (missing {missing})')
    environ.setdefault('SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE', 'env://')
    return dict(dist_init=environ['SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE'], master_addr=environ['MASTER_ADDR'],
                master_port=int(environ['MASTER_PORT']), world_size=int(environ['WORLD_SIZE']),
                agent_store=environ.get('TORCHELASTIC_USE_AGENT_STORE'))


def token_ids(n, seed, vocab):
    rng = random.Random(seed)
    return [rng.randrange(1000, vocab) for _ in range(n)]


class Workload:
    def __init__(self, runner, out, rank):
        import torch
        self.torch, self.r, self.out, self.rank = torch, runner, out, rank
        self.pool = runner.req_to_token_pool.factored_gdn_pool
        self.vocab = min(int(runner.model_config.vocab_size), 150_000)
        self.seeds = {}  # prefix length -> (ids, kv indices (device), mamba slot tensor (1,), live seed reqs)
        self.live = None  # the prepared request: (req, forward_batch)
        self.prepared = None
        self.rid = 0
        self.spans = []  # (begin, end) CUDA events of PrefillBatchGraph.run in the current forward
        self.pub_total_ms = 0.0
        self._instrument_publication()

    def _instrument_publication(self):
        from sglang.srt.mem_cache import gdn_prefill_batch_graph as g
        orig = g.PrefillBatchGraph.run
        if getattr(orig, '_p3_timed', False):
            return
        torch, spans = self.torch, self.spans

        def run(gself, *a, launch_replay=None, **k):
            if torch.cuda.is_current_stream_capturing():
                return orig(gself, *a, launch_replay=launch_replay, **k)
            b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if launch_replay is None:  # inline: on whatever stream runs it (forward stream, or the side stream's operation)
                b.record()
                out = orig(gself, *a, **k)
                e.record()
            else:  # join offload: only the replay moves to the publication stream

                def lr(replay):
                    def timed():
                        b.record()
                        r = replay()
                        e.record()
                        return r
                    return launch_replay(timed)
                out = orig(gself, *a, launch_replay=lr, **k)
            spans.append((b, e))
            return out

        run._p3_timed = True
        g.PrefillBatchGraph.run = run

    # -- request plumbing (scheduler extend path) --------------------------------------------------------------------
    def _req(self, ids, prefix_len=0, prefix_kv=None, cow_src=None):
        from array import array
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
        self.rid += 1
        req = Req(rid=f'p3micro-{self.rank}-{self.rid}', origin_input_text='', origin_input_ids=array('q', ids),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        if prefix_len:
            req.prefix_indices = prefix_kv[:prefix_len].to(self.torch.int64)
        if cow_src is not None:
            req.kv.mamba_cow_src_index = cow_src  # radix prefix hit: deferred COW applied by ModelRunner at forward start
        req.set_extend_range(prefix_len, len(ids))
        return req

    def _forward_batch(self, req):
        from sglang.benchmark.one_batch import TreeCacheNamespace, _maybe_prepare_mlp_sync_batch
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.runtime_context import get_schedule
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        r = self.r
        tree = TreeCacheNamespace(page_size=get_schedule().page_size, device=r.device,
                                  token_to_kv_pool_allocator=r.token_to_kv_pool_allocator)
        batch = ScheduleBatch.init_new(reqs=[req], req_to_token_pool=r.req_to_token_pool,
                                       token_to_kv_pool_allocator=r.token_to_kv_pool_allocator, tree_cache=tree,
                                       model_config=r.model_config, enable_overlap=False,
                                       spec_algorithm=SpeculativeAlgorithm.NONE)
        batch.prepare_for_extend()
        _maybe_prepare_mlp_sync_batch(batch, r)
        if batch.input_ids is None and getattr(batch, 'prefill_input_ids_cpu', None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
            batch.prefill_input_ids_cpu = None
        return ForwardBatch.init_new(batch, r, return_hidden_states_before_norm=False)

    def _run(self, fb):
        out = self.r.forward(fb).logits_output
        self.r.sample(out, fb)

    def _kv_of(self, req, n):
        return self.r.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :n].clone()

    def _release(self, req, keep_prefix):
        """Free the measured request: its extend KV pages, mamba slot and req row (the seed's prefix pages stay)."""
        rp, alloc = self.r.req_to_token_pool, self.r.token_to_kv_pool_allocator
        n = len(req.origin_input_ids)
        kv = rp.req_to_token[req.kv.req_pool_idx, keep_prefix:n]
        if kv.numel():
            alloc.free(kv.to(self.torch.int64))
        if getattr(req.kv, 'mamba_pool_idx', None) is not None and hasattr(rp, 'free_mamba_cache'):
            rp.free_mamba_cache(req)
        rp.free(req)

    def _seed(self, prefix_len):
        """Cold extends of the prefix in 16384-token pieces (each a prefix hit on the previous piece); slots stay allocated."""
        if prefix_len in self.seeds:
            return self.seeds[prefix_len]
        ids = token_ids(prefix_len, 7 + prefix_len, self.vocab)
        kv, slot, held, pos = None, None, [], 0
        while pos < prefix_len:
            end = min(pos + PIECE, prefix_len)
            req = self._req(ids[:end], pos, kv, slot)
            fb = self._forward_batch(req)
            self._run(fb)
            self.join()
            self.torch.cuda.synchronize()
            kv, slot, pos = self._kv_of(req, end), req.kv.mamba_pool_idx.view(1).clone(), end
            held.append(req)
        self.seeds[prefix_len] = (ids, kv, slot, held)
        return self.seeds[prefix_len]

    # -- route_micro contract ------------------------------------------------------------------------------------------
    def prepare(self, prefix_tokens, extend_tokens, seed):
        if self.live is not None:
            req, _ = self.live
            self.torch.cuda.synchronize()
            self._release(req, self.prepared[0])
            self.live = None
        ext = token_ids(extend_tokens, seed, self.vocab)
        if prefix_tokens:
            ids, kv, slot, _ = self._seed(prefix_tokens)
            req = self._req(ids + ext, prefix_tokens, kv, slot)
        else:
            req = self._req(ext)
        self.live = (req, self._forward_batch(req))
        self.prepared = (prefix_tokens, extend_tokens, seed)

    def forward(self):
        self.spans.clear()
        self._run(self.live[1])

    def join(self):
        pub = getattr(self.pool, '_pd_batch_publication', None)
        if pub is not None:
            pub.join()

    def _emitter_graph(self):
        m = self.r.model
        for owner in (m, getattr(m, 'model', None)):
            eg = (getattr(owner, 'pd_fulln_emitter_graph', None)
                  or getattr(owner, 'pd_emitter_graph', None)) if owner is not None else None
            if eg is not None:
                return eg
        return None

    def last_pub_graph_ms(self):
        done = [(b, e) for b, e in self.spans if e.query()]
        return round(sum(b.elapsed_time(e) for b, e in done), 4) if done else 0.0

    def counters(self):
        s = getattr(self.r.model, '_agg_fulln_summary', None)
        if s is None:
            raise RuntimeError('PD full-N contract not installed (no _agg_fulln_summary on the model)')
        eg = self._emitter_graph()
        est = dict(getattr(eg, 'stats', {})) if eg is not None else {}
        eb, ec = int(est.get('fallback_batch', 0)), int(est.get('fallback_capacity', 0))
        c = dict(forwards=s.forwards, trunk_replays=s.trunk_replays, batch_publications=s.batch_publications,
                 fulln_fallbacks=s.fallbacks, emitter_fallback_batch=eb, emitter_fallback_capacity=ec,
                 emitter_graph='present' if eg is not None else 'absent',
                 emitter_replayed=int(est.get('replayed', 0)), fallbacks=s.fallbacks + eb + ec,
                 last_pub_graph_ms=self.last_pub_graph_ms(), pub_graph_spans=len(self.spans))
        pub = getattr(self.pool, '_pd_batch_publication', None)
        if pub is not None:
            c.update({f'pub_{k}': v for k, v in dict(getattr(pub, 'stats', {})).items()})
        return c


def _receipt(out, rank, extra):
    Path(out).mkdir(parents=True, exist_ok=True)
    rec = dict(rank=rank, time=time.time(), helper_dir=os.environ.get('P3_HELPER_DIR') or None,
               helper_sha=os.environ.get('P3_HELPER_SHA') or None, src=os.environ.get('P3_SRC'),
               helper_files_vs_shared=(os.environ.get('P3_HELPER_DIFF') or '').split(),
               launch=os.environ.get('P3_LAUNCH_JSON'), stage=os.environ.get('P3_STAGE'),
               nccl_port=tp_nccl_port(), not_exercised=['scheduler', 'tree cache insertion / track donation',
                                                        'bootstrap / KV sender / transfer', 'HTTP / tokenizer manager'],
               **extra)
    Path(out, f'factory-r{rank}.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')


def build():
    # WS-2 10-17: PD receipt mounts can shadow /work/src (stable-src) -> sglang silently resolves to the image's stock copy;
    # assert the imported sglang lives in the declared tree before anything else
    import sglang as _sg
    _src = os.environ.get('P3_SRC', '')
    if _src and not os.path.realpath(_sg.__file__).startswith(os.path.realpath(_src)):
        raise RuntimeError(f'sglang imported from {_sg.__file__}, not from P3_SRC={_src}')
    print(f'P3_FACTORY_SGLANG {_sg.__file__}', flush=True)
    import logging
    rank = int(os.environ['RANK'])
    local = int(os.environ.get('LOCAL_RANK', rank))
    out = os.environ['P3_OUT']
    launch = json.loads(Path(os.environ['P3_LAUNCH_JSON']).read_text())
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, prepare_server_args
    from sglang.srt.utils import configure_logger
    dist = dist_env(os.environ)
    print(f'P3_FACTORY_DIST rank={rank} {json.dumps(dist)}', flush=True)
    argv = frozen_argv(launch['command'], tp_nccl_port())
    server_args = prepare_server_args(argv)
    if not getattr(server_args, '_resolution_finished', False):
        server_args.resolve_once()
    configure_logger(server_args, prefix=f' TP{rank}')
    logging.getLogger().setLevel(logging.INFO)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    # the fork's runtime context must be published by the process entry before any constructor reads it
    # (one_batch's own main does this; load_model alone does not -> 'config not published for role scheduler')
    from sglang.srt.runtime_context import publish
    publish(server_args, role="scheduler")
    t0 = time.time()
    bench_runner, _ = load_model(server_args, port_args, local, rank)
    runner = getattr(bench_runner, 'torch_runner', bench_runner)
    _receipt(out, rank, dict(argv=argv, load_s=round(time.time() - t0, 1), dist=dist,
                             disaggregation_mode=server_args.disaggregation_mode))
    return Workload(runner, out, rank)


def smoke(stage, out):
    import torch
    from sglang.srt.mem_cache.gdn_prefill_agg_contract import set_pd_micro_route
    os.environ.setdefault('P3_OUT', out)
    rank = int(os.environ['RANK'])
    w = build()
    pairs = ('0n', '1') if stage == 'p5' else ('0n', '0', '1')
    rows = []
    for prefix, extend in ((0, 4096), (16384, 1024)):
        for route in pairs:
            set_pd_micro_route(route)
            w.prepare(prefix, extend, 42)
            torch.cuda.synchronize()
            before = w.counters()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            w.forward()
            w.join()
            b.record()
            b.synchronize()
            after = w.counters()
            rows.append(dict(prefix=prefix, extend=extend, route=route, gpu_ms=a.elapsed_time(b), before=before,
                             after=after, pub_graph_ms=after['last_pub_graph_ms']))
            print('P3_FACTORY_SMOKE', json.dumps(rows[-1]), flush=True)
    ok = all(r['after']['fallbacks'] == r['before']['fallbacks'] for r in rows)
    Path(out, f'smoke-r{rank}.json').write_text(json.dumps(dict(stage=stage, rows=rows, no_fallback=ok), indent=1) + '\n')
    print('P3_FACTORY_SMOKE_DONE', 'no_fallback', ok, flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--stage', choices=('p5', 'p3'), default='p5')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not a.smoke:
        ap.error('only --smoke runs standalone; measurements go through route_micro.sh --factory p3_factory:build')
    sys.exit(smoke(a.stage, a.out))
