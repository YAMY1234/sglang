"""CPU control-flow replay of the actual runner/model/backend functions.

Tensor arithmetic and graph bodies are deliberately stubs. This establishes
planning call order, NOT numerical equivalence or measured CUDA latency.
"""

import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import types
from contextlib import nullcontext

H = Path(__file__).resolve().parent
REPO = H.parents[1]
SHA = "working-tree"
sources = {}
events = []


def source(path):
    raw = (REPO / path).read_bytes()
    sources[path] = dict(sha256=hashlib.sha256(raw).hexdigest())
    return ast.parse(raw)


def methods(tree, names):
    nodes = []
    for name in names:
        matches = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        assert len(matches) == 1, (name, len(matches))
        n = copy.deepcopy(matches[0])
        n.decorator_list = []
        nodes.append(n)
    return nodes


def build(name, base, nodes, scope):
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name=name,
                bases=[ast.Name(id=base, ctx=ast.Load())] if base else [],
                keywords=[],
                body=nodes,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(tree), "<pinned-control-flow>", "exec"), scope
    )
    return scope[name]


class Vector:
    def __init__(self, values, dtype="int64", device="cpu"):
        self.values = list(values)
        self.dtype, self.device = dtype, device
        self.shape = (len(self.values),)

    def __len__(self):
        return len(self.values)

    def numel(self):
        return len(self.values)

    def to(self, device):
        return Vector(self.values, self.dtype, device)

    def __getitem__(self, ix):
        return Vector([self.values[i] for i in ix.values], self.dtype, self.device)


class Mode:
    def is_extend(self, **kw):
        return True

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False

    def is_mixed(self):
        return False


class Batch(types.SimpleNamespace):
    def needs_forward_metadata_init(self):
        return not self.forward_metadata_ready


class Output:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class BaseGDN:
    def init_forward_metadata(self, fb):
        self.forward_metadata = types.SimpleNamespace(
            mamba_cache_indices=fb.req_pool_indices, has_mamba_track_mask=False
        )


class Pool:
    def plan_extend(self, slots, lengths, **kw):
        events.append(
            dict(
                kind="factor_plan",
                lengths=list(lengths),
                prefixes=list(kw["prefix_lens"]),
                final=list(kw["prompt_final"]),
            )
        )
        return object()


class FullAttention:
    def init_forward_metadata(self, fb):
        events.append(
            dict(kind="full_attention_plan", lengths=list(fb.extend_seq_lens_cpu))
        )


def main():
    runner_tree = source("python/sglang/srt/model_executor/runner/eager_runner.py")
    model_tree = source("python/sglang/srt/models/flash_next_duet/model.py")
    model_tree = next(
        n
        for n in model_tree.body
        if isinstance(n, ast.ClassDef)
        and any(
            isinstance(m, ast.FunctionDef) and m.name == "_twinstar_prefill"
            for m in n.body
        )
    )
    hybrid_tree = source(
        "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
    )
    hybrid_tree = next(
        n
        for n in hybrid_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HybridLinearAttnBackend"
    )
    gdn_tree = source("python/sglang/srt/layers/attention/linear/gdn_backend.py")
    scope = dict(
        BaseGDN=BaseGDN,
        copy=copy,
        itertools=itertools,
        os=types.SimpleNamespace(environ={"TWINSTAR_BOUNDARY_REPLAN": "0"}),
        _is_hip=False,
        is_cp_active=lambda fb: False,
        get_is_capture_mode=lambda: False,
        device_timer_ctx=lambda *a: nullcontext(),
        maybe_publish_prefill_shared_read_done=lambda *a: None,
        get_attn_tp_context=lambda: types.SimpleNamespace(
            maybe_input_scattered=lambda fb: nullcontext()
        ),
        get_global_expert_distribution_recorder=lambda: None,
        optional=lambda name: None,
        LogitsProcessorOutput=Output,
        logger=types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        _dev=lambda x, dtype, dev: Vector(
            x.values if isinstance(x, Vector) else x, dtype, dev
        ),
        torch=types.SimpleNamespace(
            long="int64",
            int64="int64",
            float32="float32",
            arange=lambda a, b: Vector(range(a, b)),
            cat=lambda vs: Vector(itertools.chain.from_iterable(v.values for v in vs)),
            tensor=lambda x, dtype=None, device="cpu": Vector(x, dtype, device),
            zeros=lambda n, *a, **kw: Vector([0] * n),
            get_device_module=lambda d: None,
        ),
    )
    serving = types.ModuleType("r13_fixture.serving")
    serving.embedding_streams = lambda *a: None
    sys.modules["r13_fixture.serving"] = serving
    scope["__package__"] = "r13_fixture"
    Eager = build("Eager", None, methods(runner_tree, ["_execute_extend"]), scope)
    Native = build(
        "Native",
        None,
        methods(
            model_tree,
            [
                "forward",
                "_is_twinstar_prefill",
                "_boundary_lens",
                "_sub_batch",
                "_twinstar_prefill",
                "prepare_forward_batch",
            ],
        ),
        scope,
    )
    Hybrid = build(
        "Hybrid", None, methods(hybrid_tree, ["init_forward_metadata"]), scope
    )
    GDN = build("GDN", "BaseGDN", methods(gdn_tree, ["init_forward_metadata", "_join_deferred_final"]), scope)
    parser_scope = dict(os=types.SimpleNamespace(environ={}))
    parse_node = next(
        n
        for n in source("python/sglang/srt/models/flash_next_duet/model.py").body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_defer_shallow_factor_plan_enabled"
    )
    exec(
        compile(
            ast.Module(body=[parse_node], type_ignores=[]),
            "<actual-env-parser>",
            "exec",
        ),
        parser_scope,
    )
    parse_flag = parser_scope["_defer_shallow_factor_plan_enabled"]
    assert parse_flag() is True
    for value, expected in [("0", False), ("1", True)]:
        parser_scope["os"].environ["SGLANG_GDN_DEFER_SHALLOW_FACTOR_PLAN"] = value
        assert parse_flag() is expected
    parser_scope["os"].environ["SGLANG_GDN_DEFER_SHALLOW_FACTOR_PLAN"] = "all"
    try:
        parse_flag()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid flag accepted")
    # Execute the actual startup eligibility assignment: enabling the default
    # must not enable this path for PD, embeddings, or pipeline parallelism.
    model_tree = source("python/sglang/srt/models/flash_next_duet/model.py")
    startup_assignment = next(
        n
        for n in ast.walk(model_tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "defer_shallow_factor_plan"
            for t in n.targets
        )
    )
    startup_code = compile(
        ast.Module(body=[startup_assignment], type_ignores=[]),
        "<actual-startup-eligibility>",
        "exec",
    )
    startup_cases = 0
    for value, mode, embedding, pp_size in itertools.product(
        [None, "0", "1"], ["null", "prefill", "decode"], [False, True], [1, 2]
    ):
        parser_scope["os"].environ.clear()
        if value is not None:
            parser_scope["os"].environ["SGLANG_GDN_DEFER_SHALLOW_FACTOR_PLAN"] = value
        instance = types.SimpleNamespace()
        exec(  # noqa: S102 - replay the trusted repository's startup assignment.
            startup_code,
            {
                "self": instance,
                "args": types.SimpleNamespace(
                    disaggregation_mode=mode, is_embedding=embedding, pp_size=pp_size
                ),
                "_defer_shallow_factor_plan_enabled": parse_flag,
            },
        )
        assert instance.defer_shallow_factor_plan is (
            value != "0" and mode == "null" and not embedding and pp_size == 1
        ), (value, mode, embedding, pp_size)
        startup_cases += 1
    assert startup_cases == 36
    rows = []
    shapes = [
        ([32768], [0], [False]),
        ([4096], [32768], [True]),
        ([1], [4095], [True]),
        ([32768, 4096], [0, 32768], [False, True]),
        ([1, 4096], [4095, 32768], [True, True]),
    ]
    for lengths, prefixes, final in shapes:
        for enabled, shallow, factor in itertools.product([False, True], repeat=3):
            events.clear()
            B, T = len(lengths), sum(lengths)
            fb = Batch(
                extend_seq_lens_cpu=lengths,
                extend_prefix_lens_cpu=prefixes,
                twinstar_prompt_final=final,
                forward_metadata_ready=False,
                forward_mode=Mode(),
                spec_info=None,
                return_logprob=False,
                capture_hidden_mode=types.SimpleNamespace(is_full=lambda: False),
                batch_size=B,
                seq_lens_sum=sum(prefixes) + T,
                orig_seq_lens=None,
                extend_start_loc=None,
                extend_logprob_start_lens_cpu=None,
                input_embeds=None,
                multi_item_delimiter_indices=None,
                input_ids=Vector(range(T)),
                positions=Vector(range(T)),
                req_pool_indices=Vector(range(B)),
                out_cache_loc=Vector(range(T)),
                seq_lens=Vector([p + l for p, l in zip(prefixes, lengths)]),
                seq_lens_cpu=Vector([p + l for p, l in zip(prefixes, lengths)]),
                extend_seq_lens=Vector(lengths),
                extend_prefix_lens=Vector(prefixes),
            )
            gdn = GDN()
            gdn.factored = Pool() if factor else None
            gdn._final_boundary_dense = False
            if gdn.factored is not None:
                gdn.factored._final_factor_deferred = None
            gdn._model_runner = types.SimpleNamespace(
                server_args=types.SimpleNamespace(disaggregation_mode="null")
            )
            hybrid = Hybrid()
            hybrid.attn_backend_list = [FullAttention(), gdn]
            hybrid.linear_attn_backend = gdn
            if gdn.factored is not None:
                gdn.factored.launch_pending_tracked = lambda: events.append(
                    dict(kind="boundary_done_hook")
                )
            hybrid.prepare_prefill_shared_read_snapshot = lambda *a, **k: None
            scope["get_attn_backend"] = lambda: hybrid
            model = Native()
            model.defer_shallow_factor_plan = enabled
            model._final_factor_deferred = None
            model.twinstar = {} if shallow else None
            model.fullstack = (
                dict(prefill_layer_trim=True, gdn_rank=8 if factor else 0, latent="on")
                if shallow
                else None
            )
            model.alloff_prefill_graph = False
            model.fullstack_v3_latent = False
            model.fullstack_code = True
            model.ratio = 256
            model.strict = True
            model.profile = False
            model.dump_dir = None
            model.state_audit_dir = None
            model.n_layers = 48
            model.p_layer_ids = list(range(31))
            model.emitter_ids = [31]
            model.bridges = []
            model.boundary_mode = "graph"
            model.config = types.SimpleNamespace(
                hc_count=4, hidden_size=2560, vocab_size=10
            )
            model.n_twinstar = model.n_prefix = model.n_fallback = (
                model.n_graph_fallback
            ) = 0
            model.n_graph_trunk = model.n_graph_emitters = 0

            def arithmetic(kind, result=None):
                def f(*a, **kw):
                    events.append(dict(kind=kind))
                    return result

                return f

            streams = types.SimpleNamespace(shape=(T, 10240))
            trunk = types.SimpleNamespace(
                can_run=lambda fb: True, run=arithmetic("P_trunk", streams)
            )
            emit = types.SimpleNamespace(
                can_run=lambda fb: True, run=arithmetic("emitter")
            )
            model._prefill_runners = dict(trunk=trunk, emitters=emit)
            model.latent_codec = types.SimpleNamespace(
                reconstruct=arithmetic("codec", streams)
            )
            model._publish_qsa_prefix = arithmetic("publish")
            model._boundary_graph = arithmetic("boundary", Output())
            model.model = types.SimpleNamespace(
                model=None, forward=arithmetic("full_model", Output())
            )
            runner = Eager()
            runner.enable_pdmux = False
            runner.load_batch = lambda fb, _: fb
            runner.model_runner = types.SimpleNamespace(
                _extend_forward_kwargs=lambda *a: {},
                ps=types.SimpleNamespace(attn_dcp_size=1),
                model=model,
                attn_backend=hybrid,
                device="cpu",
                device_timer=None,
                prefill_cuda_graph_runner=None,
            )
            runner._execute_extend(fb)
            hooks = [i for i, event in enumerate(events)
                     if event["kind"] == "boundary_done_hook"]
            assert len(hooks) == int(shallow and factor), events
            if hooks:
                boundaries = [i for i, event in enumerate(events)
                              if event["kind"] == "boundary"]
                assert all(i < hooks[0] for i in boundaries), events
            plans = [e for e in events if e["kind"] == "factor_plan"]
            expected = (
                (1 + int(any(l - int(f) > 0 for l, f in zip(lengths, final))))
                if shallow
                else 1
            )
            if enabled and shallow:
                expected -= 1
            assert len(plans) == (expected if factor else 0), (
                lengths,
                shallow,
                factor,
                events,
            )
            assert not getattr(fb, "_twinstar_defer_factor_plan", False)
            assert not hasattr(fb, "_twinstar_prefill_eligible")
            fa = [e for e in events if e["kind"] == "full_attention_plan"]
            assert len(fa) == (
                1 + int(shallow and any(l - int(f) > 0 for l, f in zip(lengths, final)))
            )
            rows.append(
                dict(
                    enabled=enabled,
                    lengths=lengths,
                    prefixes=prefixes,
                    final=final,
                    shallow=shallow,
                    factor=factor,
                    plans=len(plans),
                    events=copy.deepcopy(events),
                )
            )
    result = dict(
        source=SHA,
        sources=sources,
        cases=len(rows),
        startup_cases=startup_cases,
        rows=rows,
        limitations="Pinned complete control-flow functions; vector indexing shim, fake graph bodies and fake pool plan. No real tensor arithmetic, no CUDA, no pool ownership equivalence or timing assertion. Generic cases exercise Native dispatch fallback, not the external model implementation.",
    )
    if os.environ.get("PFACTOR4_TEST_OUTPUT"):
        Path(os.environ["PFACTOR4_TEST_OUTPUT"]).write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(
        json.dumps(
            dict(
                cases=len(rows),
                startup_cases=startup_cases,
                result="PASS",
                scope=result["limitations"],
            )
        )
    )


if __name__ == "__main__":
    main()
