"""Installed TP1 producer tests; CPU tensors and explicit CUDA/provider doubles."""

from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.plugin_kernels.nvidia import numeric as api
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding
from sglang.srt.kernel_plugins.nvidia import pool_attention as policy
from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen4_exp as model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class DeviceMetadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.delenv("SGLANG_QSA_POOL_DECODE", raising=False)
    monkeypatch.delenv("Q38_ATTN_CORE_V1", raising=False)
    monkeypatch.delenv("SGLANG_Q38_ATTN_CORE_TP4", raising=False)
    policy.pool_decode_enabled.cache_clear()
    policy.pool_device_sms.cache_clear()
    with ExitStack() as stack:
        stack.enter_context(
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True)
        )
        stack.enter_context(
            patch.object(model, "get_parallel", return_value=NS(tp_size=1))
        )
        stack.enter_context(
            patch.object(backend, "is_sm100_supported", return_value=True)
        )
        stack.enter_context(
            patch.object(binding, "megakernel_enabled", return_value=True)
        )
        stack.enter_context(
            patch.object(
                policy.torch.cuda, "is_current_stream_capturing", return_value=False
            )
        )
        stack.enter_context(
            patch.object(
                policy.torch.cuda,
                "get_device_properties",
                return_value=NS(multi_processor_count=148),
            )
        )
        yield
    policy.pool_decode_enabled.cache_clear()
    policy.pool_device_sms.cache_clear()


def attention():
    obj = object.__new__(model.Qwen4ExpAttentionDecoderLayer)
    torch.nn.Module.__init__(obj)
    from capability_fixtures import contract_spec

    obj._numeric_capability_spec = contract_spec()
    obj.is_qsa = True
    obj.num_heads, obj.num_kv_heads, obj.head_dim = 24, 2, 256
    obj.rotary_emb = NS(rotary_dim=64)
    obj.indexer = NS(block_topk=512, compress_ratio=4, token_topk=2048)
    return obj


def state(rows=32, width=4):
    obj = object.__new__(backend.QwenSparseAttnBackend)
    obj._fused_kv_pool_eligible = True
    obj.compress_ratio = 4
    obj.qsa_profile = NS(block_topk=512)
    q = torch.zeros((rows, 24, 256), dtype=torch.bfloat16).as_subclass(DeviceMetadata)
    packed = torch.zeros((rows, 4, 256), dtype=torch.bfloat16)
    k, v = packed[:, :2], packed[:, 2:]
    kp = torch.zeros((max(128, rows + 1), 2, 256), dtype=torch.bfloat16)
    vp = torch.zeros_like(kp)
    indices = torch.zeros((rows, 512), dtype=torch.int64)
    loc = torch.arange(rows, dtype=torch.int64)
    requests = torch.arange(rows // width, dtype=torch.int64)
    row_requests = requests.repeat_interleave(width)
    lengths = torch.full((rows,), 4097, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int64)
    slots = torch.zeros((max(1, rows // width), 8192), dtype=torch.int64)
    obj.req_to_token_pool = NS(req_to_token=slots)
    obj.token_to_kv_pool = NS(
        layer_router=NS(read=lambda layer_id: NS(key=lambda: kp, value=lambda: vp))
    )
    metadata = NS(
        indexer_metadata=NS(decode_logical_positions=positions),
        row_req_pool_indices=row_requests,
        sequence_lengths=lengths,
        is_cuda_graph=True,
    )
    obj._resolve_metadata = lambda batch: metadata
    mode = ForwardMode.TARGET_VERIFY if width == 4 else ForwardMode.DECODE
    batch = NS(
        forward_mode=mode,
        spec_info=NS(draft_token_num=width, topk=1),
        req_pool_indices=requests,
        out_cache_loc=loc,
    )
    layer = NS(tp_q_head_num=24, head_dim=256, scaling=0.0625, layer_id=0)
    args = [q, k, v, layer, batch, indices, kp, vp, loc, width, metadata]
    return obj, args


def install(rows, width, run, variant=None):
    registry = api.NumericRegistry()
    signatures = numeric_binding.collect_numeric_signatures(
        (attention(),), (rows,), width
    )
    signature = next(
        s
        for s in signatures
        if s.operation == api.NumericOperation.DIRECT_POOL_ATTENTION
        and s.variant == (variant or (1 if width == 4 else 2))
    )
    registry.register(api.NumericOffer(signature, run))
    return patch.object(numeric_binding, "numeric_registry", registry), signature


@pytest.mark.parametrize(
    "width,expected", [(4, (4, 8, 32, 64)), (1, (1, 4, 8, 32, 64))]
)
def test_installed_startup_declares_only_finite_source_rows(width, expected):
    declarations = numeric_binding.collect_numeric_signatures(
        (attention(),), (0, 1, 4, 8, 32, 64, 65, 1024), width
    )
    selected = [
        s
        for s in declarations
        if s.operation == api.NumericOperation.DIRECT_POOL_ATTENTION
        and s.variant == (1 if width == 4 else 2)
    ]
    assert tuple(s.rows for s in selected) == expected
    assert all(
        (
            s.query_heads,
            s.kv_heads,
            s.head_dim,
            s.selected_count,
            s.compress_ratio,
            s.scalar_type,
            s.rotary_dim,
            s.context_bound,
            s.variant,
        )
        == (24, 2, 256, 512, 4, 1, 0, 0, 1 if width == 4 else 2)
        for s in selected
    )


def test_merge_gate_declares_gated_verify_rows_only():
    with envs.SGLANG_ENABLE_QSA_POOL_MERGE_GATE.override(True):
        declarations = numeric_binding.collect_numeric_signatures(
            (attention(),), (1, 4, 32, 64), 4
        )
    assert {
        (s.rows, s.variant)
        for s in declarations
        if s.operation == api.NumericOperation.DIRECT_POOL_ATTENTION
    } == {(1, 2), (4, 2), (4, 5), (32, 2), (32, 5), (64, 2), (64, 5)}


@pytest.mark.parametrize("layout", ["flat", "strided"])
def test_merge_gate_rides_the_merge_and_says_so(layout):
    rows = 32
    obj, args = state(rows, 4)
    q_gate = torch.zeros((rows, 24, 512), dtype=torch.bfloat16)
    gate = (
        q_gate[:, :, 256:]
        if layout == "strided"
        else torch.zeros((rows, 6144), dtype=torch.bfloat16)
    )
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(7)

    with envs.SGLANG_ENABLE_QSA_POOL_MERGE_GATE.override(True):
        guard, sig = install(rows, 4, run, variant=5)
        with guard:
            output, gated = obj._try_numeric_pool_attention(*args, gate=gate)
    assert sig.variant == 1 | policy.POOL_MERGE_GATE and gated is True
    assert torch.equal(output, torch.full((rows, 6144), 7, dtype=torch.bfloat16))
    io = seen[0]
    assert io.gate.shape == (rows, 24, 256) and io.gate.data_ptr() == gate.data_ptr()
    assert io.gate.stride() == (
        (24 * 512, 512, 1) if layout == "strided" else (6144, 256, 1)
    )


@pytest.mark.parametrize("change", ["missing", "float", "rows"])
def test_merge_gate_without_a_usable_gate_leaves_the_call(change):
    rows = 32
    obj, args = state(rows, 4)
    gate = {
        "missing": None,
        "float": torch.zeros((rows, 6144)),
        "rows": torch.zeros((rows - 4, 6144), dtype=torch.bfloat16),
    }[change]
    seen = []
    with envs.SGLANG_ENABLE_QSA_POOL_MERGE_GATE.override(True):
        guard, _ = install(rows, 4, seen.append, variant=5)
        with guard:
            assert obj._try_numeric_pool_attention(*args, gate=gate) is None
    assert not seen


def test_merge_gate_off_keeps_the_ungated_verify():
    rows = 32
    obj, args = state(rows, 4)
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(7)

    guard, sig = install(rows, 4, run)
    gate = torch.zeros((rows, 6144), dtype=torch.bfloat16)
    with guard:
        output = obj._try_numeric_pool_attention(*args, gate=gate)
    assert sig.variant == 1 and isinstance(output, torch.Tensor)
    assert seen[0].gate is None


@pytest.mark.parametrize("value", ["0", "false", " FALSE ", "no", "off", ""])
def test_source_explicit_off_never_declares_or_queries(value, monkeypatch):
    monkeypatch.setenv("SGLANG_QSA_POOL_DECODE", value)
    obj, args = state()
    assert not [
        s
        for s in attention().numeric_signatures((32,), 4)
        if s.operation == api.NumericOperation.DIRECT_POOL_ATTENTION
    ]
    with patch.object(numeric_binding, "numeric_operator") as get:
        assert obj._try_numeric_pool_attention(*args) is None
        get.assert_not_called()


@pytest.mark.parametrize("server_tokens", [1, 2, 3, 4, 8])
def test_server_tokens_are_not_a_model_role_or_runtime_phase(server_tokens):
    selected = [
        s
        for s in attention().numeric_signatures((1, 4, 7, 8, 32, 64, 65), server_tokens)
        if s.operation == api.NumericOperation.DIRECT_POOL_ATTENTION
    ]
    assert {(s.rows, s.variant) for s in selected} == {
        (1, 2),
        (4, 2),
        (4, 1),
        (7, 2),
        (8, 2),
        (8, 1),
        (32, 2),
        (32, 1),
        (64, 2),
        (64, 1),
    }


@pytest.mark.parametrize(
    "field,value", [("num_heads", 6), ("num_kv_heads", 1), ("head_dim", 128)]
)
def test_wrong_startup_geometry_not_published(field, value):
    obj = attention()
    setattr(obj, field, value)
    assert not [
        s
        for s in obj.numeric_signatures((4, 32), 4)
        if s.operation == api.NumericOperation.DIRECT_POOL_ATTENTION
    ]


@pytest.mark.parametrize(
    "width,rows", [(4, 4), (4, 8), (4, 32), (4, 64), (1, 1), (1, 8), (1, 32), (1, 64)]
)
def test_startup_registry_facade_caller_buffer_and_original_metadata(width, rows):
    obj, args = state(rows, width)
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(7)

    guard, sig = install(rows, width, run)
    with guard:
        output = obj._try_numeric_pool_attention(*args)
    assert len(seen) == 1 and torch.equal(
        output, torch.full((rows, 6144), 7, dtype=torch.bfloat16)
    )
    io = seen[0]
    assert io.gate is None and io.query_width == width and io.store_new_kv
    assert io.new_k is args[1] and io.new_v is args[2]
    assert io.new_k.stride() == io.new_v.stride() == (1024, 256, 1)
    assert io.selection.full_slot_map is obj.req_to_token_pool.req_to_token
    assert io.selection.row_requests is args[-1].row_req_pool_indices
    assert io.selection.sequence_lengths is args[-1].sequence_lengths
    assert (
        io.selection.query_positions
        is args[-1].indexer_metadata.decode_logical_positions
    )
    assert io.out_cache_loc is args[8] and io.scale == args[3].scaling
    assert (
        io.selection.selected_indices.dtype == torch.int32
        and io.selection.selected_indices.is_contiguous()
    )
    assert torch.equal(io.selection.selected_indices, args[5].int())
    assert io.output.data_ptr() == output.data_ptr()
    assert io.scratch_out.numel() == policy.pool_scratch_elements(
        rows,
        148,
        kv_heads=2,
        head_dim=256,
        query_heads=24,
        selected_count=512,
        compress_ratio=4,
    )
    assert io.scratch_out.dtype == torch.float32 and io.scratch_out.is_contiguous()
    assert io.kv.k is args[6] and io.kv.v is args[7] and io.kv.page_size == 1
    assert sig.variant == (1 if width == 4 else 2)


@pytest.mark.parametrize(
    "case",
    [
        "master",
        "tp",
        "arch",
        "fullpool",
        "positions",
        "rowsmeta",
        "slotmap",
        "float_indices",
        "kv_dtype",
        "k_stride",
        "q_stride",
        "ratio",
        "phase",
    ],
)
def test_ineligible_never_queries_or_writes(case):
    obj, args = state()
    with ExitStack() as stack:
        if case == "master":
            stack.enter_context(
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False)
            )
        elif case == "arch":
            stack.enter_context(
                patch.object(backend, "is_sm100_supported", return_value=False)
            )
        elif case == "fullpool":
            obj._fused_kv_pool_eligible = False
        elif case == "positions":
            args[-1].indexer_metadata.decode_logical_positions = None
        elif case == "rowsmeta":
            args[-1].row_req_pool_indices = torch.zeros(1, dtype=torch.int64)
        elif case == "slotmap":
            obj.req_to_token_pool.req_to_token = None
        elif case == "float_indices":
            args[5] = args[5].float()
        elif case == "kv_dtype":
            args[6] = args[6].float()
        elif case == "k_stride":
            args[1] = args[1].transpose(1, 2)
        elif case == "q_stride":
            args[0] = args[0].transpose(1, 2)
        elif case == "ratio":
            obj.compress_ratio = 8
        elif case == "phase":
            args[4].forward_mode = ForwardMode.EXTEND
        if case in ("tp", "ratio"):
            source = attention()
            if case == "ratio":
                source.indexer.compress_ratio = 8
            if case == "tp":
                stack.enter_context(
                    patch.object(model, "get_parallel", return_value=NS(tp_size=4))
                )
            assert source.numeric_signatures((32,), 4) == ()
            with patch.object(
                numeric_binding, "numeric_registry", api.NumericRegistry()
            ):
                assert obj._try_numeric_pool_attention(*args) is None
        else:
            with patch.object(numeric_binding, "numeric_operator") as lookup:
                assert obj._try_numeric_pool_attention(*args) is None
                lookup.assert_not_called()


def test_no_offer_and_unprepared_exact_signature_do_not_write():
    obj, args = state()
    calls = []
    guard, _ = install(4, 4, lambda io: calls.append(io))
    with (
        guard,
        patch.object(
            torch,
            "empty",
            side_effect=AssertionError("no caller allocation before an exact offer"),
        ),
    ):
        assert obj._try_numeric_pool_attention(*args) is None
    assert calls == []


def test_prestore_and_real_row_map_fallback_preserve_identity():
    # Source fallback is legal for decode, where requests already have one row.
    obj, args = state(8, 1)
    args[-1].row_req_pool_indices = None
    seen = []
    guard, _ = install(8, 1, lambda io: seen.append(io))
    with guard:
        obj._try_numeric_pool_attention(*args, kv_prestored=True)
    assert seen[0].selection.row_requests is args[4].req_pool_indices
    assert seen[0].store_new_kv is False


def test_source_m4_tcgen_flag_does_not_veto_the_pool_offer(monkeypatch):
    # The source TCGEN05 M4 core is a separate direct variant; its flag must not
    # leave the pool's M4 offer stock.
    monkeypatch.setenv("Q38_ATTN_CORE_V1", "1")
    obj, args = state(4, 4)
    for graph in (True, False):
        args[-1].is_cuda_graph = graph
        with patch.object(
            numeric_binding, "numeric_operator", return_value=None
        ) as lookup:
            assert obj._try_numeric_pool_attention(*args) is None
            lookup.assert_called_once()


def test_failure_after_provider_write_propagates_without_stock_retry():
    obj, args = state()
    stock = []
    obj._forward_trtllm_sparse = lambda *a, **kw: stock.append(True)

    def run(io):
        io.kv.k.fill_(1)
        raise RuntimeError("after first KV write")

    guard, _ = install(32, 4, run)
    with (
        guard,
        patch.object(backend, "_resolve_trtllm_sparse_decode", return_value=object()),
    ):
        with pytest.raises(RuntimeError, match="after first KV write"):
            obj._try_fused_kv_attention(*args[:6])
    assert stock == [] and torch.all(args[6] == 1)


def test_backend_complete_dispatch_preserves_external_gate_and_stock_fallback():
    obj, args = state()
    seen = []
    stock = []
    obj._forward_trtllm_sparse = lambda *a, **kw: stock.append((a, kw)) or "stock"
    gate = torch.zeros_like(args[0])
    guard, _ = install(32, 4, lambda io: seen.append(io))
    with (
        guard,
        patch.object(backend, "_resolve_trtllm_sparse_decode", return_value=object()),
    ):
        result, gate_applied = obj._try_fused_kv_attention(*args[:6], output_gate=gate)
    assert (
        result.shape == (32, 6144)
        and gate_applied is False
        and seen[0].gate is None
        and stock == []
    )
    with (
        patch.object(numeric_binding, "numeric_registry", api.NumericRegistry()),
        patch.object(backend, "_resolve_trtllm_sparse_decode", return_value=object()),
    ):
        assert obj._try_fused_kv_attention(*args[:6]) == "stock"
    assert len(stock) == 1


def test_tp4_specialized_success_precedes_pool_and_fallback():
    obj, args = state(4, 4)
    chosen = object()
    obj._try_numeric_direct_attention = lambda *a: chosen
    obj._try_numeric_pool_attention = lambda *a, **kw: pytest.fail(
        "pool must not precede specialized core"
    )
    with patch.object(backend, "_resolve_trtllm_sparse_decode", return_value=object()):
        result, applied = obj._try_fused_kv_attention(
            *args[:6], output_gate=torch.zeros_like(args[0])
        )
    assert result is chosen and applied is True


def direct_call(args, gate):
    q, k, v, layer, batch, indices, kp, vp, loc, width, metadata = args
    return (q, k, v, layer, batch, indices, gate, kp, vp, loc, width, metadata)


def tcgen05_declared(tp_size, rows=(1, 4, 8, 32, 64)):
    with patch.object(model, "get_parallel", return_value=NS(tp_size=tp_size)):
        declared = attention().numeric_signatures(rows, 4)
    return {
        (s.rows, s.query_heads, s.kv_heads)
        for s in declared
        if s.operation == api.NumericOperation.DIRECT_POOL_ATTENTION
        and s.variant == policy.DIRECT_TCGEN05_CORE
    }


@pytest.mark.parametrize("tp_size,expected", [(1, {(4, 24, 2)}), (4, set())])
def test_tcgen05_core_declares_only_the_tp1_width4_verify(
    monkeypatch, tp_size, expected
):
    assert tcgen05_declared(tp_size) == set()
    monkeypatch.setenv("Q38_ATTN_CORE_V1", "1")
    assert tcgen05_declared(tp_size) == expected


@pytest.mark.parametrize(
    "flags,looked_up",
    [
        ((), []),
        (("SGLANG_Q38_ATTN_CORE_TP4",), [0]),
        (("Q38_ATTN_CORE_V1",), [policy.DIRECT_TCGEN05_CORE]),
        (
            ("SGLANG_Q38_ATTN_CORE_TP4", "Q38_ATTN_CORE_V1"),
            [0, policy.DIRECT_TCGEN05_CORE],
        ),
    ],
)
def test_direct_core_asks_each_enabled_source_variant(monkeypatch, flags, looked_up):
    for name in flags:
        monkeypatch.setenv(name, "1")
    obj, args = state(4, 4)
    gate = torch.zeros((4, 6144), dtype=torch.bfloat16).as_subclass(DeviceMetadata)
    asked = []

    def lookup(signature, *, prepared):
        asked.append((signature.variant, prepared))
        return None

    with patch.object(numeric_binding, "numeric_operator", side_effect=lookup):
        assert obj._try_numeric_direct_attention(*direct_call(args, gate)) is None
    assert asked == [(variant, True) for variant in looked_up]


@pytest.mark.parametrize("tp4_flag", [False, True])
@pytest.mark.parametrize("gate_shape", [(4, 6144), (4, 24, 256)])
def test_tcgen05_core_runs_the_startup_offer_on_two_head_scratch(
    monkeypatch, tp4_flag, gate_shape
):
    if tp4_flag:
        # Its TP4-only declaration has no TP1 offer; the TCGEN05 core still binds.
        monkeypatch.setenv("SGLANG_Q38_ATTN_CORE_TP4", "1")
    monkeypatch.setenv("Q38_ATTN_CORE_V1", "1")
    obj, args = state(4, 4)
    gate = torch.zeros(gate_shape, dtype=torch.bfloat16).as_subclass(DeviceMetadata)
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(3)

    registry = api.NumericRegistry()
    for signature in numeric_binding.collect_numeric_signatures(
        (attention(),), (4,), 4
    ):
        if signature.operation == api.NumericOperation.DIRECT_POOL_ATTENTION:
            registry.register(api.NumericOffer(signature, run))
    with patch.object(numeric_binding, "numeric_registry", registry):
        output = obj._try_numeric_direct_attention(*direct_call(args, gate))
    assert output.shape == (4, 6144) and bool(torch.all(output == 3))
    (io,) = seen
    assert io.gate is gate and io.store_new_kv is True and io.query_width == 4
    assert io.q.shape == (4, 24, 256) and io.kv.k is args[6]
    assert io.scratch_out.dtype == torch.float32
    assert io.scratch_out.numel() == 4 * 2 * 33 * 16 * (256 + 2)
    assert io.selection.selected_indices.dtype == torch.int32


@pytest.mark.parametrize(
    "change", ["rows8", "no_graph", "decode", "topk2", "float_kv", "no_positions"]
)
def test_tcgen05_core_keeps_the_source_eligibility(monkeypatch, change):
    monkeypatch.setenv("Q38_ATTN_CORE_V1", "1")
    rows = 8 if change == "rows8" else 4
    obj, args = state(rows, 4)
    metadata, batch = args[-1], args[4]
    if change == "no_graph":
        metadata.is_cuda_graph = False
    elif change == "decode":
        batch.forward_mode = ForwardMode.DECODE
    elif change == "topk2":
        batch.spec_info.topk = 2
    elif change == "float_kv":
        args[6], args[7] = args[6].float(), args[7].float()
    elif change == "no_positions":
        metadata.indexer_metadata = NS(decode_logical_positions=None)
    gate = torch.zeros((rows, 6144), dtype=torch.bfloat16).as_subclass(DeviceMetadata)
    asked = []

    def lookup(signature, *, prepared):
        asked.append(prepared)
        return None

    with patch.object(numeric_binding, "numeric_operator", side_effect=lookup):
        assert obj._try_numeric_direct_attention(*direct_call(args, gate)) is None
    assert asked == [False]


@pytest.mark.parametrize("rows", range(1, 65))
def test_workspace_matches_original_source_choose_config(rows):
    for sms in (1, 80, 132, 148, 256):
        # Literal source arithmetic, replacing only triton.cdiv with integer cdiv.
        cdiv = lambda a, b: (a + b - 1) // b
        rh = rows * 2
        bn = 32 if rh >= 32 else 64
        waves = 3.0 if rh >= 32 else 1.0
        splits = max(1, round(waves * sms / rh))
        chunk = cdiv(cdiv(2051, splits), bn) * bn
        splits = cdiv(2051, chunk)
        assert policy.pool_scratch_elements(
            rows,
            sms,
            kv_heads=2,
            head_dim=256,
            query_heads=24,
            selected_count=512,
            compress_ratio=4,
        ) == rows * 2 * splits * 16 * (256 + 2)


def test_sms_only_eager_read_and_capture_cache_hit_or_cold_refusal():
    device = torch.device("cpu")
    with patch.object(
        torch.cuda, "get_device_properties", return_value=NS(multi_processor_count=148)
    ) as read:
        assert policy.pool_device_sms(device) == 148
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            assert policy.pool_device_sms(device) == 148
            with pytest.raises(RuntimeError, match="before capture"):
                policy.pool_device_sms(torch.device("cuda", 1))
        assert read.call_count == 1


@pytest.mark.parametrize("width", [2, 3])
def test_actual_verify_width_two_or_three_retains_stock(width):
    obj, args = state(12, 4)
    args[4].spec_info.draft_token_num = width
    args[4].req_pool_indices = torch.zeros(12 // width, dtype=torch.int64)
    args[-2] = width
    with patch.object(numeric_binding, "numeric_operator") as lookup:
        assert obj._try_numeric_pool_attention(*args) is None
        lookup.assert_not_called()


@pytest.mark.parametrize("length", [499, 512, 513, 4096, 4097, 8192, 8193])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_lengths_positions_maps_are_original_uncompressed_metadata(length, dtype):
    obj, args = state(8, 1)
    meta = args[-1]
    obj.req_to_token_pool.req_to_token = obj.req_to_token_pool.req_to_token.to(dtype)
    meta.row_req_pool_indices = meta.row_req_pool_indices.to(dtype)
    meta.sequence_lengths = torch.full((8,), length, dtype=dtype)
    meta.indexer_metadata.decode_logical_positions = torch.full(
        (8,), length - 1, dtype=dtype
    )
    seen = []
    guard, _ = install(8, 1, seen.append)
    with guard:
        obj._try_numeric_pool_attention(*args)
    assert seen[0].selection.sequence_lengths is meta.sequence_lengths
    assert (
        seen[0].selection.query_positions
        is meta.indexer_metadata.decode_logical_positions
    )
    assert torch.equal(
        seen[0].selection.sequence_lengths, torch.full((8,), length, dtype=dtype)
    )


def test_capture_storage_metadata_cache_miss_refuses_before_any_write():
    obj, args = state()
    guard, _ = install(32, 4, lambda io: pytest.fail("cold SM metadata must not run"))
    with (
        guard,
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
    ):
        with pytest.raises(RuntimeError, match="before capture"):
            obj._try_numeric_pool_attention(*args)


def test_runtime_loader_and_master_gate_do_not_execute_an_offer():
    obj, args = state()
    calls = []
    guard, _ = install(32, 4, calls.append)
    with guard, patch.object(binding, "megakernel_enabled", return_value=False):
        assert obj._try_numeric_pool_attention(*args) is None
    assert calls == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))


def fp8_state(rows=128):
    obj, args = state(rows, 4)
    fp8 = torch.float8_e4m3fn
    args[1] = torch.zeros((rows, 2, 256), dtype=fp8)
    args[2] = torch.zeros((rows, 2, 256), dtype=fp8)
    args[6] = torch.zeros((rows + 1, 2, 256), dtype=fp8)
    args[7] = torch.zeros((rows + 1, 2, 256), dtype=fp8)
    return obj, args


def union_signatures(widths):
    return [
        s
        for s in attention().numeric_signatures(widths, 4)
        if s.operation == api.NumericOperation.POOL_UNION_ATTENTION
    ]


def test_fp8_union_declared_only_when_enabled():
    widths = (4, 60, 64, 66, 128, 576, 580)
    assert not union_signatures(widths)
    with envs.SGLANG_QSA_FP8_POOL_VERIFY.override(True):
        assert [s.rows for s in union_signatures(widths)] == [64, 128, 576]
        assert all(s.variant == 0 for s in union_signatures(widths))
        with envs.SGLANG_QSA_FP8_POOL_VERIFY_UNION_WARPS.override(16):
            assert {s.variant for s in union_signatures(widths)} == {16}
            with envs.SGLANG_QSA_FP8_POOL_VERIFY_WS.override(True):
                assert {s.variant for s in union_signatures(widths)} == {16 | 64}
                with envs.SGLANG_QSA_FP8_POOL_VERIFY_WS_PDL.override(True):
                    assert {s.variant for s in union_signatures(widths)} == {
                        16 | 64 | 128
                    }
            # PDL rides only the warp-specialized kernel.
            with envs.SGLANG_QSA_FP8_POOL_VERIFY_WS_PDL.override(True):
                assert {s.variant for s in union_signatures(widths)} == {16}
            with envs.SGLANG_QSA_FUSED_FP8_KV_WRITE.override(True):
                assert {s.variant for s in union_signatures(widths)} == {16 | 256}
        with (
            envs.SGLANG_QSA_FP8_POOL_VERIFY_UNION_WARPS.override(12),
            pytest.raises(ValueError, match="UNION_WARPS"),
        ):
            union_signatures(widths)
        with patch.object(model, "get_parallel", return_value=NS(tp_size=4)):
            assert not union_signatures(widths)


@pytest.mark.parametrize("ws", [False, True])
@pytest.mark.parametrize("warps", [8, 16])
@pytest.mark.parametrize("rows", [64, 128, 576])
def test_fp8_union_caller_buffers_and_source_plan(rows, warps, ws):
    obj, args = fp8_state(rows)
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(3)

    with (
        envs.SGLANG_QSA_FP8_POOL_VERIFY.override(True),
        envs.SGLANG_QSA_FP8_POOL_VERIFY_UNION_WARPS.override(warps),
        envs.SGLANG_QSA_FP8_POOL_VERIFY_WS.override(ws),
    ):
        registry = api.NumericRegistry()
        (signature,) = union_signatures((rows,))
        # The offer is looked up with the declared variant.
        assert signature.variant == (0 if warps == 8 else warps) | (64 if ws else 0)
        registry.register(api.NumericOffer(signature, run))
        with patch.object(numeric_binding, "numeric_registry", registry):
            output = obj._try_numeric_pool_attention(*args)
    assert torch.equal(output, torch.full((rows, 6144), 3, dtype=torch.bfloat16))
    (io,) = seen
    assert type(io) is api.PoolUnionAttentionIO
    assert io.new_k is args[1] and io.kv.k is args[6] and io.kv.v is args[7]
    assert (io.kv.k_scale, io.kv.v_scale, io.query_width) == (1.0, 1.0, 4)
    assert io.selection.selected_indices.dtype == torch.int32
    splits, integers, floats = policy.union_workspace(
        rows,
        148,
        kv_heads=2,
        head_dim=256,
        query_heads=24,
        selected_count=512,
        compress_ratio=4,
        variant=signature.variant,
    )
    assert io.split_count == splits
    assert io.integer_scratch_out.dtype == torch.int32
    assert io.integer_scratch_out.numel() == integers
    assert io.scratch_out.dtype == torch.float32 and io.scratch_out.numel() == floats


def test_fp8_union_takes_bf16_new_kv_for_the_cast_write():
    rows = 64
    obj, args = fp8_state(rows)
    args[1] = torch.zeros((rows, 3, 2, 256), dtype=torch.bfloat16)[:, 0]
    args[2] = torch.zeros((rows, 3, 2, 256), dtype=torch.bfloat16)[:, 1]
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(3)

    with (
        envs.SGLANG_QSA_FP8_POOL_VERIFY.override(True),
        envs.SGLANG_QSA_FUSED_FP8_KV_WRITE.override(True),
    ):
        registry = api.NumericRegistry()
        (signature,) = union_signatures((rows,))
        assert signature.variant == policy.UNION_FP8_KV_WRITE
        registry.register(api.NumericOffer(signature, run))
        with patch.object(numeric_binding, "numeric_registry", registry):
            obj._try_numeric_pool_attention(*args)
    (io,) = seen
    assert io.new_k is args[1] and io.new_v is args[2]
    assert io.new_k.dtype == torch.bfloat16


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_target_verify_defers_the_fp8_cast_to_the_union_write(enabled, pooled):
    rows = 64
    obj, args = state(rows, 4)
    fp8 = torch.float8_e4m3fn
    kc = torch.zeros((rows + 1, 2, 256), dtype=fp8)
    vc = torch.zeros_like(kc)
    obj.token_to_kv_pool = NS(
        layer_router=NS(read=lambda layer_id: NS(key=lambda: kc, value=lambda: vc))
    )
    q, k, v, layer, batch, indices = args[:6]
    seen = {}

    def pool(*a, **kw):
        seen["pool"] = a[1].dtype
        return torch.zeros((rows, 6144), dtype=torch.bfloat16) if pooled else None

    def sparse(*a, **kw):
        seen["sparse"] = kw["new_kv"][0].dtype
        return torch.zeros((rows, 6144), dtype=torch.bfloat16)

    with (
        envs.SGLANG_QSA_FUSED_FP8_KV_WRITE.override(enabled),
        patch.object(backend, "_resolve_trtllm_sparse_decode", return_value=object()),
        patch.object(obj, "_try_numeric_pool_attention", pool, create=True),
        patch.object(obj, "_forward_trtllm_sparse", sparse, create=True),
    ):
        obj._try_fused_kv_attention(q, k, v, layer, batch, indices)
    assert seen["pool"] == (torch.bfloat16 if enabled else fp8)
    # Any route other than the union write gets the stock casts.
    assert seen.get("sparse", fp8) == fp8 and ("sparse" in seen) != pooled


def draft_extend_state(rows=64):
    obj, args = fp8_state(rows)
    kc, vc = args[6], args[7]
    obj.token_to_kv_pool = NS(
        layer_router=NS(read=lambda layer_id: NS(key=lambda: kc, value=lambda: vc))
    )
    obj.qsa_profile = NS(block_topk=512, budget=2048)
    args[4].forward_mode = ForwardMode.DRAFT_EXTEND_V2
    q, k, v, layer, batch = args[:5]
    k, v = (torch.zeros((rows, 2, 256), dtype=torch.bfloat16) for _ in range(2))
    # The indexer's expanded token indices: block c's first token at column 4 * c.
    blocks = torch.randint(0, 900, (rows, 512), dtype=torch.int32)
    blocks[:, -3:] = -1
    tokens = (blocks.repeat_interleave(4, 1) * 4 + torch.arange(4).repeat(512)).int()
    tokens[blocks.repeat_interleave(4, 1) < 0] = -1
    tokens = torch.cat((tokens, torch.full((rows, 3), 5, dtype=torch.int32)), 1)
    return obj, (q, k, v, layer, batch, tokens), blocks


@pytest.mark.parametrize("enabled", [False, True])
def test_draft_extend_attends_on_the_union_provider(enabled):
    rows = 64
    obj, args, blocks = draft_extend_state(rows)
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(5)

    with (
        envs.SGLANG_QSA_FP8_POOL_VERIFY.override(True),
        envs.SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION.override(enabled),
    ):
        registry = api.NumericRegistry()
        (signature,) = union_signatures((rows,))
        registry.register(api.NumericOffer(signature, run))
        with patch.object(numeric_binding, "numeric_registry", registry):
            output = obj._try_draft_extend_union(*args)
    if not enabled:
        assert output is None and not seen
        return
    assert torch.equal(output, torch.full((rows, 6144), 5, dtype=torch.bfloat16))
    (io,) = seen
    assert torch.equal(io.selection.selected_indices, blocks)
    assert io.store_new_kv and io.query_width == 4
    assert io.new_k.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("case", ["eager", "rows", "bf16_pool", "no_union", "verify"])
def test_draft_extend_union_keeps_stock_outside_its_scope(case):
    rows = 64
    obj, args, _ = draft_extend_state(rows)
    q, k, v, layer, batch, tokens = args
    metadata = obj._resolve_metadata(batch)
    if case == "eager":
        metadata.is_cuda_graph = False
    elif case == "rows":
        args = (q, k, v, layer, batch, tokens[:-4])
    elif case == "bf16_pool":
        kc = torch.zeros((rows + 1, 2, 256), dtype=torch.bfloat16)
        obj.token_to_kv_pool = NS(
            layer_router=NS(read=lambda layer_id: NS(key=lambda: kc, value=lambda: kc))
        )
    elif case == "verify":
        batch.forward_mode = ForwardMode.TARGET_VERIFY
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION.override(True))
        if case != "no_union":
            stack.enter_context(envs.SGLANG_QSA_FP8_POOL_VERIFY.override(True))
        lookup = stack.enter_context(patch.object(numeric_binding, "numeric_operator"))
        lookup.return_value = None
        assert obj._try_draft_extend_union(*args) is None


@pytest.mark.parametrize("case", ["off", "decode", "topk", "bf16_new_kv"])
def test_fp8_pool_outside_union_scope_keeps_stock(case):
    obj, args = fp8_state()
    if case == "decode":
        args[4].forward_mode = ForwardMode.DECODE
    elif case == "topk":
        args[4].spec_info.topk = 4
    elif case == "bf16_new_kv":
        args[1] = args[1].to(torch.bfloat16)
    with ExitStack() as stack:
        if case != "off":
            stack.enter_context(envs.SGLANG_QSA_FP8_POOL_VERIFY.override(True))
        lookup = stack.enter_context(patch.object(numeric_binding, "numeric_operator"))
        assert obj._try_numeric_pool_attention(*args) is None
        lookup.assert_not_called()


@pytest.mark.parametrize("sms", [80, 148, 256])
@pytest.mark.parametrize("requests", [16, 43, 64, 86, 128, 144])
def test_union_workspace_matches_original_source_plan(requests, sms):
    # Literal source choose_union_splits and buffers at W4, Q24/KV2, 512 blocks.
    pairs = requests * 2
    if pairs > sms:
        splits = 2
    else:
        cost = lambda s: -(-pairs * s // sms) / s * (1.0 + 0.1 * s)
        splits = min(range(1, 17), key=lambda s: (cost(s), s))
    nb, nt, qp = 2048, 2048 * 4 + 16, 64
    assert policy.union_workspace(
        requests * 4,
        sms,
        kv_heads=2,
        head_dim=256,
        query_heads=24,
        selected_count=512,
        compress_ratio=4,
    ) == (
        splits,
        requests * (2 * nt + nb + 1) + requests * 4,
        requests * 2 * splits * qp * (256 + 2),
    )


@pytest.mark.parametrize("sms", [80, 148, 212, 256])
@pytest.mark.parametrize("requests", [16, 43, 44, 64, 128, 144])
def test_union_workspace_follows_the_ws_kernel_split_plan(requests, sms):
    # Literal source choose_ws_splits: the most splits within one wave, at most 4.
    splits = max(1, min(4, sms // (requests * 2)))
    nb, nt, qp = 2048, 2048 * 4 + 16, 64
    assert policy.union_workspace(
        requests * 4,
        sms,
        kv_heads=2,
        head_dim=256,
        query_heads=24,
        selected_count=512,
        compress_ratio=4,
        variant=64 | 16,
    ) == (
        splits,
        requests * (2 * nt + nb + 1) + requests * 4,
        requests * 2 * splits * qp * (256 + 2),
    )


@pytest.mark.parametrize("value", [1, 2, 16])
def test_union_workspace_takes_the_source_ws_split_count(value):
    nb, nt, qp, requests = 2048, 2048 * 4 + 16, 64, 176
    kwargs = dict(
        kv_heads=2, head_dim=256, query_heads=24, selected_count=512, compress_ratio=4
    )
    with envs.SGLANG_QSA_FP8_POOL_VERIFY_WS_SPLITS.override(value):
        assert policy.union_workspace(requests * 4, 212, **kwargs, variant=64 | 16) == (
            value,
            requests * (2 * nt + nb + 1) + requests * 4,
            requests * 2 * value * qp * (256 + 2),
        )
        # Kernel 2 keeps its own plan.
        assert policy.union_workspace(requests * 4, 212, **kwargs, variant=16)[0] == 2
    for bad in (0, 17):
        with envs.SGLANG_QSA_FP8_POOL_VERIFY_WS_SPLITS.override(bad):
            with pytest.raises(ValueError):
                policy.union_workspace(requests * 4, 212, **kwargs, variant=64)


def packed_prefill_signatures(stages=None):
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_ENABLE_QSA_TMA_PREFILL.override(True))
        if stages is not None:
            stack.enter_context(envs.SGLANG_QSA_TMA_PREFILL_STAGES.override(stages))
        return [
            s
            for s in attention().numeric_signatures((32, 600), 4)
            if s.operation == api.NumericOperation.PACKED_PREFILL_ATTENTION
        ]


def test_packed_prefill_declares_one_row_independent_signature():
    assert not [
        s
        for s in attention().numeric_signatures((32, 600), 4)
        if s.operation == api.NumericOperation.PACKED_PREFILL_ATTENTION
    ]
    expected = api.NumericSignature(
        api.NumericOperation.PACKED_PREFILL_ATTENTION,
        0,
        24,
        2,
        256,
        2051,
        4,
        api.ScalarType.BF16,
        variant=3,
    )
    assert packed_prefill_signatures() == [expected]
    assert [s.variant for s in packed_prefill_signatures(stages=2)] == [2]
    assert packed_prefill_signatures(stages=9) == []
    # The source admits any TP; the layer's local heads carry the geometry.
    with patch.object(model, "get_parallel", return_value=NS(tp_size=4)):
        assert packed_prefill_signatures() == [expected]


def packed_prefill_call(rows=600, extend_lens=(300, 300), **change):
    obj = object.__new__(backend.QwenSparseAttnBackend)
    obj.compress_ratio = 4
    q = torch.zeros((rows, 24, 256), dtype=torch.bfloat16).as_subclass(DeviceMetadata)
    k = torch.zeros((1200, 2, 256), dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    args = dict(
        q=q,
        layer=NS(tp_q_head_num=24, head_dim=256, scaling=0.0625, layer_id=0),
        forward_batch=NS(extend_seq_lens=torch.tensor(extend_lens)),
        k=k,
        v=v,
        indices=torch.zeros((rows, 2051), dtype=torch.int32),
        cu_q=torch.tensor([0, 300, 600], dtype=torch.int32),
        cu_k=torch.tensor([0, 600, 1200], dtype=torch.int32),
        kv_lens=torch.tensor([600, 600], dtype=torch.int32),
        extend_lens=list(extend_lens),
    )
    args.update(change)
    return obj, args


def packed_prefill_registry(run):
    (signature,) = packed_prefill_signatures()
    registry = api.NumericRegistry()
    registry.register(api.NumericOffer(signature, run))
    return patch.object(numeric_binding, "numeric_registry", registry)


def test_packed_prefill_serves_the_stock_tile_with_host_max_query_rows():
    seen = []

    def run(io):
        seen.append(io)
        io.output.fill_(7)

    obj, args = packed_prefill_call(extend_lens=(250, 350))
    with (
        packed_prefill_registry(run),
        envs.SGLANG_ENABLE_QSA_TMA_PREFILL.override(True),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA GB300"),
    ):
        output = obj._numeric_packed_prefill(**args)
    (io,) = seen
    assert io.max_query_rows == 350 and io.scale == 0.0625
    assert io.query is args["q"] and io.packed_k is args["k"] and io.output is output
    assert io.selected_indices is args["indices"] and io.cu_key is args["cu_k"]
    assert output.shape == args["q"].shape and torch.all(output == 7)


@pytest.mark.parametrize(
    "case", ["off", "capturing", "stale_host_lens", "small_tile", "int64_indices"]
)
def test_packed_prefill_outside_source_admission_is_stock(case):
    change = {}
    if case == "stale_host_lens":
        change["forward_batch"] = NS(extend_seq_lens=torch.tensor([300, 300, 0]))
    if case == "small_tile":
        change["q"] = torch.zeros((512, 24, 256), dtype=torch.bfloat16).as_subclass(
            DeviceMetadata
        )
    if case == "int64_indices":
        change["indices"] = torch.zeros((600, 2051), dtype=torch.int64)
    obj, args = packed_prefill_call(**change)
    with (
        packed_prefill_registry(lambda io: pytest.fail("provider ran")),
        envs.SGLANG_ENABLE_QSA_TMA_PREFILL.override(case != "off"),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA GB300"),
        patch.object(
            policy.torch.cuda,
            "is_current_stream_capturing",
            return_value=case == "capturing",
        ),
        patch.object(numeric_binding, "numeric_operator") as lookup,
    ):
        assert obj._numeric_packed_prefill(**args) is None
    lookup.assert_not_called()
