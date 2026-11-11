"""Data-only contracts for optional rank-local numeric providers.

Inputs are borrowed for one call. Providers write only the explicitly named
output/scratch tensors and declared seed-publication rows; state commits and communication
remain in the framework. Writable pool ranges are limited to deferred KV prepare's declared current slots, direct attention's
new KV slots and the declared convolution/ring slots of the local recurrent ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor


class ScalarType(IntEnum):
    BF16 = 1
    FP16 = 2
    FP32 = 3
    FP8_E4M3 = 7
    INT64 = 4
    INT32 = 5
    PACKED_FP4 = 6


class NumericOperation(IntEnum):
    SPARSE_SCORE = 2
    SPARSE_SELECT = 3
    PACKED_ATTENTION = 4  # Reserved unsupported operation; no producer or IO surface.
    DIRECT_POOL_ATTENTION = 5
    POOL_UNION_ATTENTION = 6
    MTP_CAPTURE_SEED = 7
    MTP_DEFERRED_KV_PREPARE = 8
    PREFILL_SELECT = 9
    PACKED_PREFILL_ATTENTION = 17
    HC_GATE = 10
    HC_APPLY_NORM = 11
    HC_APPLY_NORM_GATE = 12
    HC_MIX = 14
    GDN_FRONT = 30
    GDN_CORE = 31
    GDN_VERIFY_GLUE = 32
    GDN_NATIVE_PDL = 33
    LOCAL_EXPERTS = 40
    FINALIZE_SHARED = 41
    FINALIZE_PERMUTED_SHARED = 42
    ROUTER_QUANT = 43


@dataclass(frozen=True, slots=True)
class NumericCapabilityQuery:
    """Host-resolved source facts; the selected provider owns supported domains."""

    operation: NumericOperation
    attributes: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if not isinstance(self.operation, NumericOperation):
            raise TypeError("capability operation must be NumericOperation")
        if not isinstance(self.attributes, tuple):
            raise TypeError("capability attributes must be an immutable tuple")
        names = set()
        for pair in self.attributes:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("capability attributes must contain name/value tuples")
            name, value = pair
            if not isinstance(name, str) or not name:
                raise TypeError("capability attribute names must be nonempty strings")
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    "capability attribute values must be integers, not bool"
                )
            if name in names:
                raise ValueError("capability attribute names must be unique")
            names.add(name)


@dataclass(frozen=True, slots=True)
class NumericSignature:
    """An exact, host-resolved capability; zero is a value, never a wildcard."""

    operation: NumericOperation
    rows: int
    query_heads: int
    kv_heads: int
    head_dim: int
    selected_count: int
    compress_ratio: int
    scalar_type: int
    rotary_dim: int = 0
    context_bound: int = 0
    # Sparse score: 0=MK, 3=MK short32, 4=grouped V7/W4 defaults,
    # 5=per-row defaults, 6=grouped4 chain, 7=grouped4 chain with a two-stage
    # key pipeline. Sparse select: 0=MK, 1=split defaults. Prefill select and
    # packed prefill attention: rows is 0 (packed prefill rows vary per call);
    # context_bound bounds a select row's keys, and attention's variant is its
    # key-load pipeline stage count. Prefill select's variant: bits 0-3 = the
    # top-k (0 sampled, 1 one-pass), bit 4 = the tensor-core scorer.
    variant: int = 0
    # Grid CTAs per four-row request group of sparse score 6/7; 0 elsewhere.
    ctas_per_group: int = 0


@dataclass(frozen=True, slots=True)
class HcSignature:
    operation: NumericOperation
    rows: int
    stream_count: int
    hidden_size: int
    scalar_type: int
    # Combine CTA partition count (1/2/4), not the fixed 8 gate-reduction splits.
    parts: int = 2
    lowrank: int = 0
    variant: int = 0


class GdnFrontVariant(IntEnum):
    # Exact projection family and source-resolved pipeline stage count.
    MK_STAGES_6 = 0
    MK_STAGES_12 = 1
    TGV_STAGES_8 = 2
    TGV_STAGES_12 = 3


class GdnActivation(IntEnum):
    NONE = 0
    SILU = 1
    SIGMOID = 2


class GdnDotPrecision(IntEnum):
    TF32 = 0
    TF32X3 = 1
    IEEE = 2


@dataclass(frozen=True, slots=True)
class GdnSignature:
    operation: NumericOperation
    rows: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    tokens_per_request: int
    scalar_type: int
    # Operation-scoped: core uses raw/fused-norm variants; front uses GdnFrontVariant.
    # An enum value declares a layout, not that any provider supports it.
    variant: int = 0
    gate_scalar_type: int = 3


@dataclass(frozen=True, slots=True)
class ExpertSignature:
    operation: NumericOperation
    rows: int
    hidden_size: int
    intermediate_size: int
    experts: int
    topk: int
    scalar_type: int
    variant: int = 0


@dataclass(frozen=True, slots=True)
class FinalizePermutedSignature:
    """Native permuted GEMM2 finalize plus gated, initially ungated shared rows.

    Rows 0 is one declaration for every row count above the graph widths.
    """

    operation: NumericOperation
    rows: int
    hidden_size: int
    topk: int
    scalar_type: int
    weight_scalar_type: int
    num_warps: int = 16
    use_pdl: int = 1
    variant: int = 0


@dataclass(frozen=True, slots=True)
class NativeGdnPdlSignature:
    """Original native verify, distinct from MK core and conv/verify glue.

    Every source-resolved launch flag is an integer; providers reject bool
    before a signature cache lookup. Shapes are finite framework declarations.
    """

    operation: NumericOperation
    rows: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    tokens_per_request: int
    scalar_type: int
    gate_scalar_type: int = 1
    state_scalar_type: int = 1
    dt_bias_scalar_type: int = 3
    cache_len: int = 16
    storage_mode: int = 1
    wait_for_gate_producer: int = 0
    state_prefetch: int = 0
    dot_precision: int = 0  # GdnDotPrecision.TF32 or IEEE, resolved by framework.


@dataclass(frozen=True, slots=True)
class RouterQuantSignature:
    """Router logits and NVFP4 quantization with operation-scoped variants.

    Variants: 0=narrow K2, 1=narrow K4, 2=wide K2. A provider validates
    every integer field, rejecting bool before a signature-key cache lookup.
    Source eligibility and bundle preparation belong to the framework.
    """

    operation: NumericOperation
    rows: int
    hidden_size: int
    expert_count: int
    scalar_type: int
    variant: int = 0


@dataclass(frozen=True, slots=True)
class MtpCaptureSeedSignature:
    """Exact BS1 seed publication metadata; no tensor contents in the key."""

    operation: NumericOperation
    rows: int
    columns: int
    topk_row_stride: int
    seed_rows: int
    seed_columns: int
    seed_row_stride: int
    front_tokens: int
    trash_row: int
    extend_scalar_type: int
    accept_scalar_type: int
    request_scalar_type: int
    length_scalar_type: int
    pdl: int


@dataclass(frozen=True, slots=True)
class MtpDeferredKvSignature:
    """Exact TP4/H1 deferred lookup folded into the original KV packer.

    metadata_dtype_mask bits 0..5 describe locations, row_requests,
    sequence_lengths, req_to_token, lookup_requests and lookup_positions;
    Each bit value 0 means INT32 and value 1 means INT64. No tensor contents
    enter this key.
    """

    operation: NumericOperation
    rows: int
    kv_heads: int
    head_dim: int
    scalar_type: int
    index_columns: int
    packed_columns: int
    tail_width: int
    index_pool_rows: int
    cache_slots: int
    request_pool_rows: int
    map_columns: int
    map_row_stride: int
    new_k_row_stride: int
    new_v_row_stride: int
    metadata_dtype_mask: int


OperatorSignature = (
    NumericSignature
    | HcSignature
    | GdnSignature
    | ExpertSignature
    | FinalizePermutedSignature
    | RouterQuantSignature
    | NativeGdnPdlSignature
    | MtpCaptureSeedSignature
    | MtpDeferredKvSignature
)


@runtime_checkable
class NumericMetadataProducer(Protocol):
    """Framework-internal producer; the provider receives only its result."""

    def numeric_signatures(
        self, widths: tuple[int, ...], tokens_per_request: int
    ) -> tuple[OperatorSignature, ...]: ...


@dataclass(frozen=True, slots=True)
class KvTensors:
    k: Tensor
    v: Tensor
    k_scale: float
    v_scale: float
    page_size: int


@dataclass(frozen=True, slots=True)
class SparseScoreIO:
    query: Tensor
    compressed_keys: Tensor
    compressed_slot_map: Tensor
    compressed_lengths: Tensor
    # Required for MK variants; unused native grouped/per-row variants use None.
    row_requests: Tensor | None
    query_weights: Tensor | None
    page_size: int
    compress_ratio: int
    scale: float
    scores_out: Tensor
    scratch_out: Tensor


@dataclass(frozen=True, slots=True)
class PackedPrefillAttentionIO:
    """Packed BF16 prefix-chunk K/V with live request boundaries; caller output.

    ``max_query_rows`` is the host maximum of the call's per-request query rows.
    """

    query: Tensor
    packed_k: Tensor
    packed_v: Tensor
    selected_indices: Tensor
    cu_query: Tensor
    cu_key: Tensor
    sequence_lengths: Tensor
    scale: float
    max_query_rows: int
    output: Tensor


@dataclass(frozen=True, slots=True)
class PrefillSelectIO:
    """Packed BF16 queries/keys, request-relative ranges and caller token IDs.

    Ranges, positions and lengths are the call's live per-row metadata. All
    transient workspace is caller-owned: ``chunk_rows`` rows per logits chunk
    (padded to 32 in the scratch views), ``logits_out`` as wide as the call's
    longest row, and ``split_tag`` whose length bounds the scorer's key split.
    """

    query: Tensor
    compressed_keys: Tensor
    row_starts: Tensor
    row_ends: Tensor
    query_positions: Tensor
    sequence_lengths: Tensor
    selected_indices_out: Tensor
    query_scratch_out: Tensor
    starts_scratch_out: Tensor
    ends_scratch_out: Tensor
    row_lengths: Tensor
    zero_offsets: Tensor
    block_indices_out: Tensor
    logits_out: Tensor
    split_tag: Tensor
    chunk_rows: int
    scorer_sms: int


@dataclass(frozen=True, slots=True)
class SparseSelectIO:
    """The selected indices must match the stock ordered indices, including ties."""

    scores: Tensor
    lengths: Tensor
    selected_count: int
    selected_indices_out: Tensor
    # None means counts were not requested and must not be produced or read.
    # When supplied, the framework prepares min(lengths, selected_count) and
    # providers preserve that tensor. Selected-index ordering is unchanged.
    counts_out: Tensor | None
    scratch_out: Tensor


@dataclass(frozen=True, slots=True)
class DirectPoolSelection:
    """Read-only direct-pool metadata; lengths are uncompressed token lengths."""

    full_slot_map: Tensor
    row_requests: Tensor
    selected_indices: Tensor
    sequence_lengths: Tensor
    query_positions: Tensor


@dataclass(frozen=True, slots=True)
class DirectPoolAttentionIO:
    """Only new K/V at out_cache_loc and output/scratch may be written.

    KV writes require store_new_kv=True. Selected indices, maps, positions and
    lengths are read-only. A
    provider cannot advance allocation, history, or speculative state here.
    """

    q: Tensor
    gate: Tensor | None
    new_k: Tensor
    new_v: Tensor
    out_cache_loc: Tensor
    kv: KvTensors
    selection: DirectPoolSelection
    scale: float
    query_width: int
    store_new_kv: bool
    output: Tensor
    scratch_out: Tensor


@dataclass(frozen=True, slots=True)
class PoolUnionAttentionIO(DirectPoolAttentionIO):
    """FP8-pool target verify; scratch_out is FP32, integer_scratch_out INT32.

    split_count is the host-resolved split plan, fixed before capture.
    """

    integer_scratch_out: Tensor
    split_count: int


@dataclass(frozen=True, slots=True)
class HcGateIO:
    normed: Tensor
    inject_weight: Tensor
    stream_count: int
    hidden_size: int
    partials_out: Tensor


@dataclass(frozen=True, slots=True)
class HcApplyNormIO:
    block: Tensor
    residual: Tensor
    gate_partials: Tensor
    norm_weight: Tensor
    epsilon: float
    parts: int
    combined_out: Tensor
    normalized_out: Tensor


@dataclass(frozen=True, slots=True)
class HcApplyNormGateIO:
    block: Tensor
    residual: Tensor
    gate_partials: Tensor
    norm_weight: Tensor
    next_inject_weight: Tensor
    epsilon: float
    parts: int
    combined_out: Tensor
    normalized_out: Tensor
    next_partials_out: Tensor


@dataclass(frozen=True, slots=True)
class HcMixIO:
    normed: Tensor
    down_weight: Tensor
    permuted_up_weight: Tensor
    stream_count: int
    hidden_size: int
    output: Tensor
    scratch_out: Tensor


@dataclass(frozen=True, slots=True)
class GdnRingIO:
    """Only indexed ring entries are writable; all cursors are read-only."""

    d: Tensor
    k: Tensor
    g: Tensor
    raw_v: Tensor
    raw_k: Tensor
    beta: Tensor
    d_low: Tensor | None
    k_low: Tensor | None
    state_indices: Tensor
    replay_indices: Tensor
    write_pos: Tensor
    cache_base: Tensor
    is_flush: Tensor


@dataclass(frozen=True, slots=True)
class GdnLaunchConfig:
    # Verify glue derives its tile from the ring; block_n=0, hoist=False.
    block_n: int
    num_warps: int
    num_stages: int
    hoist: bool
    dot_precision: int  # GdnDotPrecision; not a backend-specific string.
    prefetch: bool
    historical_branch: int  # 0/1, matching the original HIST_BRANCH boolean.
    use_pdl: bool
    trigger_early: bool


@dataclass(frozen=True, slots=True)
class GdnFrontIO:
    x: Tensor
    # Present only for the explicitly offered input-projection/front variant.
    packed_weight: Tensor | None
    conv_state: Tensor
    conv_weight: Tensor
    conv_bias: Tensor | None
    state_indices: Tensor
    intermediate_indices: Tensor
    activation: int  # GdnActivation; front variants require SILU.
    token_major_out: bool
    window_mode: int  # 0 = native indexed [slot, T, channel, history] window.
    launch: GdnLaunchConfig
    convolved_out: Tensor
    intermediate_conv_window_out: Tensor
    # Input-projection/front variants require this caller-owned full projection.
    # Appended defaults preserve existing constructors, not offer admission.
    packed_out: Tensor | None = None
    # Native convolution sentinel; independent of the window layout policy.
    pad_slot: int = -1


@dataclass(frozen=True, slots=True)
class GdnCoreIO:
    qkv: Tensor
    a: Tensor
    b: Tensor
    a_row_stride: int
    b_row_stride: int
    a_log: Tensor
    dt_bias: Tensor
    # Checkpoint is read-only: commit remains a later framework operation.
    checkpoint: Tensor
    ring: GdnRingIO
    z: Tensor | None
    norm_weight: Tensor | None
    epsilon: float
    value_rows: int
    raw_out: Tensor
    normalized_out: Tensor | None


@dataclass(frozen=True, slots=True)
class GdnVerifyGlueIO:
    mixed_qkvz: Tensor
    mixed_ba: Tensor
    conv_state: Tensor
    conv_weight: Tensor
    conv_bias: Tensor | None
    intermediate_indices: Tensor
    a_log: Tensor
    dt_bias: Tensor
    norm_weight: Tensor
    norm_epsilon: float
    checkpoint: Tensor
    ring: GdnRingIO
    key_heads: int
    key_dim: int
    tokens_per_request: int
    null_slot: int
    conv_activation: int
    norm_activation: int
    launch: GdnLaunchConfig
    output: Tensor
    intermediate_conv_window_out: Tensor


@dataclass(frozen=True, slots=True)
class LocalExpertsIO:
    """Rank-local math only; descriptors are provider-internal bind-time resources."""

    packed_w13: Tensor
    packed_w2: Tensor
    scale13: Tensor
    scale2: Tensor
    global_scale1: Tensor
    global_scale1_c: Tensor
    global_scale2: Tensor
    packed_x: Tensor
    x_scale: Tensor
    routing_logits: Tensor
    experts: int
    intermediate_size: int
    hidden_size: int
    topk: int
    activation: int
    local_expert_start: int
    local_expert_count: int
    routed_scale: float
    route_weights_out: Tensor
    route_ids_out: Tensor
    expert_rows_out: Tensor
    final_out: Tensor | None
    scratch_out: Tensor


@dataclass(frozen=True, slots=True)
class FinalizeSharedIO:
    """Local ordered finalize and shared gate/add; no TP sum is performed."""

    hidden: Tensor
    gate_weight: Tensor
    shared: Tensor
    expert_rows: Tensor
    route_weights: Tensor
    cols: int
    use_pdl: bool
    output: Tensor


@dataclass(frozen=True, slots=True)
class FinalizePermutedSharedIO:
    """Borrowed native FI tensors; output only is written, without TP reduction.

    Shared rows have not yet been gated. The permutation is expanded-to-permuted
    INT32, with negative entries skipped. GEMM2's actual row stride is retained.
    """

    hidden: Tensor
    gate_weight: Tensor
    shared: Tensor
    gemm2_out: Tensor
    expanded_idx_to_permuted_idx: Tensor
    expert_weights: Tensor
    output: Tensor


@dataclass(frozen=True, slots=True)
class NativeGdnPdlIO:
    """Borrowed native pointers; checkpoint/cursors are read-only in verify.

    Only caller out and indexed ring entries may be written. The framework
    supplies real strided q/k/v, query offsets and already-int8 flush flags;
    commit/fold/cursor advance and producer stream ordering stay outside.
    """

    q: Tensor
    k: Tensor
    v: Tensor
    a: Tensor
    b: Tensor
    a_log: Tensor
    dt_bias: Tensor
    out: Tensor
    checkpoint_state: Tensor
    d_cache: Tensor
    k_cache: Tensor
    g_cache: Tensor
    rawv_cache: Tensor | None
    rawk_cache: Tensor | None
    beta_cache: Tensor | None
    query_start_loc: Tensor
    ssm_state_indices: Tensor
    replay_indices: Tensor
    write_pos: Tensor
    cache_base: Tensor
    is_flush: Tensor


@dataclass(frozen=True, slots=True)
class RouterQuantIO:
    """Borrowed router inputs and three aligned caller-owned output buffers.

    global_scale is the original w13_input_scale_quant-domain FP32 scalar,
    not an inverse FI scale. scale_out is linear E4M3, not interleaved scales.
    Providers preserve the original launch/current-stream/PDL semantics and
    write only logits_out, packed_out and scale_out; no allocation or cast.
    """

    hidden_states: Tensor
    router_weight: Tensor
    global_scale: Tensor
    logits_out: Tensor
    packed_out: Tensor
    scale_out: Tensor


@dataclass(frozen=True, slots=True)
class MtpCaptureSeedIO:
    """Five reads and two borrowed persistent outputs; no allocation/commit.

    Only seed_out[resolved request-or-trash, :columns] and the corresponding
    captured_lengths_out element may be written. Remaining rows/tail stay intact.
    """

    topk_indices: Tensor
    extend_lengths: Tensor
    accept_counts: Tensor
    requests: Tensor
    expanded_lengths: Tensor
    seed_out: Tensor
    captured_lengths_out: Tensor


@dataclass(frozen=True, slots=True)
class MtpDeferredKvIO:
    """Fifteen borrowed tensors; only declared current KV slots and outputs write.

    The framework owns requests/lookup metadata and the KV/packed capacities.
    Counts is a required raw output. Ordinary QK/RoPE and attention stay outside.
    """

    k_cache: Tensor
    v_cache: Tensor
    new_k: Tensor
    new_v: Tensor
    locations: Tensor
    req_to_token: Tensor
    row_requests: Tensor
    sequence_lengths: Tensor
    indices_pool: Tensor
    lookup_requests: Tensor
    captured_lengths: Tensor
    lookup_positions: Tensor
    counts_out: Tensor
    packed_k_out: Tensor
    packed_v_out: Tensor


NumericIO = (
    PackedPrefillAttentionIO
    | PrefillSelectIO
    | SparseScoreIO
    | SparseSelectIO
    | DirectPoolAttentionIO
    | PoolUnionAttentionIO
    | HcGateIO
    | HcApplyNormIO
    | HcApplyNormGateIO
    | HcMixIO
    | GdnFrontIO
    | GdnCoreIO
    | GdnVerifyGlueIO
    | LocalExpertsIO
    | FinalizeSharedIO
    | FinalizePermutedSharedIO
    | RouterQuantIO
    | NativeGdnPdlIO
    | MtpCaptureSeedIO
    | MtpDeferredKvIO
)
NumericKernel = Callable[[NumericIO], None]


@dataclass(frozen=True, slots=True)
class NumericOffer:
    """Provider declarations are fixed at binding time, before capture."""

    signature: OperatorSignature
    run: NumericKernel


class NumericRegistry:
    """Startup-only declarations; re-registering the same offer is idempotent."""

    def __init__(self) -> None:
        self._offers: dict[OperatorSignature, NumericOffer] = {}

    def register(self, offer: NumericOffer) -> None:
        previous = self._offers.get(offer.signature)
        if previous is not None and previous != offer:
            raise ValueError("conflicting numeric providers for the same signature")
        self._offers[offer.signature] = offer

    def get(self, signature: OperatorSignature) -> NumericOffer | None:
        return self._offers.get(signature)


numeric_registry = NumericRegistry()
