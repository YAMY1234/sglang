# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Opt-in dispatch options for FlashInfer's compute capability 10.7 inference MegaMOE
kernel (FlashInfer 0.7.4 / 0.7.5).

Its router kernel sends each rank's expert counts to every peer, then a helper
CTA builds a push table from the full count matrix, and the push CTAs store
each route's metadata into its destination pool, fence, and bump a ready counter
on every peer. The main kernel's token-in warps wait for all of
those counters before they pull any row. Two independent options shorten that:

- Per-CTA push layout: every push warp computes its route's destination offset
  from the count matrix itself (a warp sum over at most one rank's experts and
  over the world size), so the pushes no longer wait for the helper's table.
- Tagged dispatch: every route's metadata (and its score) travels as one 64-bit
  word that carries a per-launch tag, stored with a relaxed system-scope store
  after one system fence per push warp. A token-in warp polls the words of its
  own rows with acquire loads and pulls each row as soon as its word lands, so
  the push needs no trailing fence and no ready counter. The tag is a launch
  epoch plus one; every rank advances its epoch once per launch after the
  kernel's last NVLink barrier, so ranks stay in step and a word from an earlier
  launch never matches. Without the ready counter nothing else holds the main
  kernel's tail reset behind this rank's own push CTAs, so each push CTA counts
  itself done and CTA 0 waits for all of them before the tail.

Both options keep the routes, the pool layout and the arithmetic, so outputs
are bit-identical to the stock kernel. They require the token-in warps to own
no token-back work (BF16 combine with ``token_back_mode="epi_warps"``) and a
router with one data CTA (at most 16384 routes per rank). Every EP rank must make
the same choice.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import bootstrap_paths

# Puts FlashInfer's vendored kernel tree (package ``sources``) on sys.path.
bootstrap_paths()

from sources.communication.nvlink_domain.token_comm import (  # noqa: E402
    TokenCommNonDeterministic,
    _copy_atom,
    _MetadataPushRouter,
)
from sources.communication.token_protocol import TokenSrcMetadata  # noqa: E402
from sources.helpers.dsl_helpers import mark_alignment  # noqa: E402
from sources.helpers.flag_batch import make_flag_batch_tracker  # noqa: E402
from sources.helpers.iket_compat import iket  # noqa: E402
from sources.helpers.ptx_helpers import (  # noqa: E402
    cp_async_bulk_s2g,
    nanosleep,
    red_add_relaxed_sys_s32,
    red_add_release_gpu_s32,
    stg_b64,
    stg_f32,
    tma_load_1d,
)
from sources.helpers.utils import ceil_div  # noqa: E402

from sglang.srt.layers.moe import flashinfer_megamoe_sm107_opts  # noqa: E402

_stock_router_helper_single_cta = _MetadataPushRouter._router_helper_single_cta
_stock_register_router_workspace = _MetadataPushRouter._register_router_workspace

_tagged_dispatch = False
_per_cta_push_layout = False

# Local: the epoch this launch tags its words with; advanced once per launch.
_DISPATCH_EPOCH_REGION = "nvlink.token_comm.dispatch_epoch"
# Shared, one tagged word per pool row: the route's score bits under the tag.
_TAGGED_SCORES_REGION = "nvlink.token_comm.tagged_scores"
# Local, tail-reset: push CTAs of this launch done with the router's counters.
_PUSHES_DONE_REGION = "nvlink.token_comm.dispatch_pushes_done"
# Tagged metadata word: token bits 0-15, top-k slot 16-23, source rank 24-31,
# tag 32-63 (never zero, so a zeroed workspace holds no valid word).
_TOKEN_LIMIT = 1 << 16
_FIELD8_LIMIT = 1 << 8
_POLL_SLEEP_CYCLES = 32


@dsl_user_op
def _st_relaxed_sys_b64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "st.relaxed.sys.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_sys_b64(address: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [address.ir_value()],
            "ld.acquire.sys.global.b64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_sys_b64(address: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [address.ir_value()],
            "ld.relaxed.sys.global.b64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


def _check_router_eligible(component) -> None:
    if component.router_data_cta_count != 1:
        raise ValueError(
            "The FlashInfer MegaMOE dispatch options need a router with one data "
            f"CTA (at most 16384 routes per rank), got {component.router_data_cta_count}."
        )


def _check_eligible(token_comm: TokenCommNonDeterministic) -> None:
    _check_router_eligible(token_comm)
    if token_comm.token_back_enabled:
        raise ValueError(
            "The FlashInfer MegaMOE dispatch options need BF16 combine with "
            f"token_back_mode='epi_warps', got {token_comm.token_back_mode!r} with "
            f"{token_comm.combine_format}."
        )
    if _tagged_dispatch:
        if (
            token_comm.max_tokens_per_rank > _TOKEN_LIMIT
            or token_comm.topk > _FIELD8_LIMIT
            or token_comm.world_size > _FIELD8_LIMIT
        ):
            raise ValueError(
                "The tagged dispatch packs a route into 32 bits: it needs at most "
                f"{_TOKEN_LIMIT} tokens per rank and at most {_FIELD8_LIMIT} ranks and "
                f"top-k slots, got {token_comm.max_tokens_per_rank}, "
                f"{token_comm.world_size} and {token_comm.topk}."
            )
        if TokenCommNonDeterministic.reset_tail is not _reset_tail_advancing_epoch:
            raise RuntimeError(
                "The tagged dispatch must be installed after every other MegaMOE "
                "kernel option: its tail no longer advances the dispatch epoch."
            )


# ----- Tagged words -----


@cute.jit
def _dispatch_tag(device_workspace) -> Int32:
    # The epoch is in [0, 2**32 - 2], so the tag is never zero.
    epoch = cute.arch.load(
        device_workspace.ptr(_DISPATCH_EPOCH_REGION), Int32, sem="relaxed", scope="gpu"
    )
    return epoch + Int32(1)


@cute.jit
def _tagged_word(tag: Int32, low: Int32) -> Int64:
    return (Int64(tag) << Int64(32)) | (Int64(low) & Int64(0xFFFFFFFF))


@cute.jit
def _word_tag(word: Int64) -> Int32:
    return Int32(word >> Int64(32))


@cute.jit
def _poll_tagged_word(address: Int64, tag: Int32) -> Int64:
    word = _ld_acquire_sys_b64(address)
    while _word_tag(word) != tag:
        nanosleep(_POLL_SLEEP_CYCLES)
        word = _ld_acquire_sys_b64(address)
    return word


@cute.jit
def _tagged_metadata_low(stock_metadata: Int64) -> Int32:
    # Stock record: rank bits 48-63, top-k slot 32-47, token 0-31.
    token = Int32(stock_metadata & Int64(0xFFFF))
    topk = Int32((stock_metadata >> Int64(32)) & Int64(0xFF))
    rank = Int32((stock_metadata >> Int64(48)) & Int64(0xFF))
    return token | (topk << Int32(16)) | (rank << Int32(24))


def _decode_tagged_metadata(word: Int64) -> TokenSrcMetadata:
    return TokenSrcMetadata(
        src_rank=Int32((word >> Int64(24)) & Int64(0xFF)),
        src_token=Int32(word & Int64(0xFFFF)),
        src_topk=Int32((word >> Int64(16)) & Int64(0xFF)),
    )


@classmethod
def _load_tagged_metadata(cls, address) -> TokenSrcMetadata:
    # Every pool record of the process is a tagged word once tagged dispatch is
    # installed; a system-scope load sees the word token-in already acquired.
    raw_address = address if isinstance(address, Int64) else address.toint()
    return _decode_tagged_metadata(_ld_relaxed_sys_b64(raw_address))


@cute.jit
def _advance_dispatch_epoch(token_comm) -> None:
    # After the tail's last NVLink barrier every peer has finished token-in, so
    # every push warp that wrote a word for this launch has read this epoch.
    transfer_thread_idx = (
        token_comm._transfer_warp_idx * Int32(32) + token_comm._lane_idx
    )
    if token_comm._linear_cta_idx == Int32(0):
        if transfer_thread_idx == Int32(0):
            epoch_pointer = token_comm._device_workspace.ptr(_DISPATCH_EPOCH_REGION)
            next_epoch = cute.arch.load(
                epoch_pointer, Int32, sem="relaxed", scope="gpu"
            ) + Int32(1)
            if next_epoch == Int32(-1):
                next_epoch = Int32(0)
            cute.arch.store(epoch_pointer, next_epoch, sem="relaxed", scope="gpu")


@cute.jit
def _wait_for_own_pushes(token_comm) -> None:
    # The tail's grid sync holds every CTA behind CTA 0, so the tail reset follows this wait.
    transfer_thread_idx = (
        token_comm._transfer_warp_idx * Int32(32) + token_comm._lane_idx
    )
    if token_comm._linear_cta_idx == Int32(0):
        if transfer_thread_idx == Int32(0):
            pushes_done = token_comm._device_workspace.ptr(_PUSHES_DONE_REGION)
            while cute.arch.load(
                pushes_done, Int32, sem="acquire", scope="gpu"
            ) != Int32(token_comm.router_push_cta_count):
                nanosleep(_POLL_SLEEP_CYCLES)


@cute.jit
def _reset_tail_advancing_epoch(self) -> None:
    iket.range_push("tail.wait_own_pushes")
    _wait_for_own_pushes(self)
    iket.range_pop()
    if cutlass.const_expr(flashinfer_megamoe_sm107_opts._installed):
        flashinfer_megamoe_sm107_opts._reset_tail_single_barrier(self)
    else:
        flashinfer_megamoe_sm107_opts._stock_reset_tail(self)
    iket.range_push("tail.advance_dispatch_epoch")
    _advance_dispatch_epoch(self)
    iket.range_pop()


def _register_router_workspace(self, workspace) -> None:
    _stock_register_router_workspace(self, workspace)
    if _tagged_dispatch:
        workspace.register(
            _DISPATCH_EPOCH_REGION,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )
        workspace.register(
            _PUSHES_DONE_REGION,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            reset="tail_reset",
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                _TAGGED_SCORES_REGION,
                cutlass.Int64,
                (self.worst_case_token_count,),
                buffer_space="shared",
                byte_alignment=16,
                reset="zero_on_first_allocate",
            )


# ----- Router: push CTAs -----


@cute.jit
def _router_helper_single_cta(self, smem_base: cute.Pointer) -> None:
    # With the per-CTA layout nothing reads the helper's tables.
    if cutlass.const_expr(not _per_cta_push_layout):
        _stock_router_helper_single_cta(self, smem_base)


@cute.jit
def _per_cta_destination_begin(self, global_expert: Int32) -> Int32:
    """The helper's push_destination_base entry, summed by the expert's own warp."""
    lane_idx = self._router_lane_idx
    sizes = self._device_workspace.tensor(self.sizes_region)
    sizes_by_rank = self._device_workspace.tensor(self.sizes_by_rank_region)
    destination_rank = global_expert // Int32(self.experts_per_rank)
    destination_expert_begin = destination_rank * Int32(self.experts_per_rank)
    padding = Int32(self.token_padding_block)
    pool_base = Int32(0)
    for expert_round in cutlass.range_constexpr(ceil_div(self.experts_per_rank, 32)):
        expert = destination_expert_begin + Int32(expert_round * 32) + lane_idx
        padded = Int32(0)
        if expert < global_expert:
            padded = (sizes[expert] + padding - Int32(1)) // padding * padding
        pool_base = pool_base + cute.arch.warp_reduction_sum(padded)
    local_ring_position = (
        Int32(self._router_local_rank) - destination_rank + Int32(self.world_size)
    ) % Int32(self.world_size)
    ring_offset = Int32(0)
    for ring_round in cutlass.range_constexpr(ceil_div(self.world_size, 32)):
        ring_position = Int32(ring_round * 32) + lane_idx
        count = Int32(0)
        if ring_position < local_ring_position:
            source_rank = (destination_rank + ring_position) % Int32(self.world_size)
            count = sizes_by_rank[source_rank, global_expert]
        ring_offset = ring_offset + cute.arch.warp_reduction_sum(count)
    return pool_base + ring_offset


@cute.jit
def _router_push_metadata(self) -> None:
    _check_router_eligible(self)
    if cutlass.const_expr(_tagged_dispatch):
        # Publishes this rank's staged rows at system scope before any tagged
        # word: each word below is then a release pattern of its own lane.
        cute.arch.fence_acq_rel_sys()
    sorted_metadata_ready = self._device_workspace.ptr(
        self.sorted_metadata_ready_region
    )
    if self._router_thread_idx == Int32(0):
        while cute.arch.load(
            sorted_metadata_ready, Int32, sem="acquire", scope="gpu"
        ) != Int32(self.router_data_cta_count):
            nanosleep(150)
        if cutlass.const_expr(_per_cta_push_layout):
            sizes_ready = self._device_workspace.ptr(self.sizes_ready_region)
            while cute.arch.load(
                sizes_ready, Int32, sem="acquire", scope="sys"
            ) != Int32(self.world_size):
                nanosleep(150)
        else:
            push_table_ready = self._device_workspace.ptr(self.push_table_ready_region)
            while cute.arch.load(
                push_table_ready, Int32, sem="acquire", scope="gpu"
            ) != Int32(1):
                nanosleep(150)
    cute.arch.sync_threads()

    global_expert = (
        self._router_linear_cta_idx * Int32(self.router_warps_per_cta)
        + self._router_warp_idx
    )
    if global_expert < Int32(self.expert_count):
        if cutlass.const_expr(_per_cta_push_layout):
            destination_begin = _per_cta_destination_begin(self, global_expert)
        else:
            destination_begin = self._device_workspace.tensor(
                self.push_destination_base_region
            )[global_expert]
        _push_expert_routes(self, global_expert, destination_begin)

    if cutlass.const_expr(_tagged_dispatch):
        # Every thread of this CTA is past its last read of a tail-reset region.
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            red_add_release_gpu_s32(
                self._device_workspace.ptr(_PUSHES_DONE_REGION), Int32(1)
            )
    else:
        _publish_metadata_ready(self)


@cute.jit
def _push_expert_routes(self, global_expert: Int32, destination_begin: Int32) -> None:
    sizes_by_rank = self._device_workspace.tensor(self.sizes_by_rank_region)
    source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
    route_count = sizes_by_rank[self._router_local_rank, global_expert]
    source_begin = source_expert_base[global_expert]
    destination_rank = global_expert // Int32(self.experts_per_rank)
    peer_offset = self._peer_rank_ptr_mapper.map(Int64(0), destination_rank, Int64(0))
    source_metadata = self._device_workspace.ptr(self.sorted_metadata_region)
    destination_metadata_address = (
        self._device_workspace.ptr(self.token_src_metadata_region).toint() + peer_offset
    )
    if cutlass.const_expr(self.apply_topk_at_fc1):
        source_scores = self._device_workspace.ptr(self.sorted_scores_region)
        if cutlass.const_expr(_tagged_dispatch):
            destination_scores_address = (
                self._device_workspace.ptr(_TAGGED_SCORES_REGION).toint() + peer_offset
            )
        else:
            destination_scores_address = (
                self._device_workspace.ptr(self.fc1_topk_scores_region).toint()
                + peer_offset
            )
    if cutlass.const_expr(_tagged_dispatch):
        tag = _dispatch_tag(self._device_workspace)
    route_round_count = (route_count + Int32(31)) // Int32(32)
    for route_round in cutlass.range(route_round_count, unroll=1):
        route = Int32(route_round) * Int32(32) + self._router_lane_idx
        if route < route_count:
            source_position = source_begin + route
            destination_position = destination_begin + route
            metadata = cute.arch.load(source_metadata + source_position, cutlass.Int64)
            if cutlass.const_expr(_tagged_dispatch):
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    score_bits = cute.arch.load(
                        cute.make_ptr(
                            cutlass.Int32,
                            source_scores.toint() + Int64(source_position) * Int64(4),
                            AddressSpace.gmem,
                            assumed_align=4,
                        ),
                        cutlass.Int32,
                    )
                    _st_relaxed_sys_b64(
                        destination_scores_address
                        + Int64(destination_position) * Int64(8),
                        _tagged_word(tag, score_bits),
                    )
                _st_relaxed_sys_b64(
                    destination_metadata_address
                    + Int64(destination_position) * Int64(TokenSrcMetadata.nbytes),
                    _tagged_word(tag, _tagged_metadata_low(metadata)),
                )
            else:
                stg_b64(
                    destination_metadata_address
                    + Int64(destination_position) * Int64(TokenSrcMetadata.nbytes),
                    metadata,
                )
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    score = cute.arch.load(
                        source_scores + source_position, cutlass.Float32
                    )
                    stg_f32(
                        destination_scores_address
                        + Int64(destination_position) * Int64(4),
                        score,
                    )


@cute.jit
def _publish_metadata_ready(self) -> None:
    """The stock push epilogue: one system fence, then one bump per peer."""
    block_thread_count = self.router_warps_per_cta * 32
    cute.arch.sync_threads()
    if self._router_thread_idx == Int32(0):
        cute.arch.fence_acq_rel_sys()
    cute.arch.sync_threads()
    metadata_ready_address = self._device_workspace.ptr(
        self.metadata_ready_region
    ).toint()
    rank_round_count = ceil_div(self.world_size, block_thread_count)
    for rank_round in cutlass.range_constexpr(rank_round_count):
        destination_rank = (
            Int32(rank_round * block_thread_count) + self._router_thread_idx
        )
        if destination_rank < Int32(self.world_size):
            red_add_relaxed_sys_s32(
                self._peer_rank_ptr_mapper.map(
                    metadata_ready_address, destination_rank, Int64(0)
                ),
                Int32(1),
            )


# ----- Main kernel: token-in -----


@cute.jit
def _token_in(self, smem_workspace, smem_base: cute.Pointer) -> None:
    """Stock token-in, with pool bases from the staged sizes and, when tagged, per-row metadata waits."""
    _check_eligible(self)
    transfer_warp_idx = self._transfer_warp_idx
    lane_idx = self._lane_idx
    transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx

    iket.range_push("token_in.wait_sizes_ready")
    self._router.wait_for_sizes_ready(self._device_workspace)
    iket.range_pop()
    iket.range_push("token_in.stage_sizes")
    owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
    _stage_owned_sizes(self, owned_sizes, transfer_thread_idx)
    iket.range_pop()
    iket.range_push("token_in.size_barrier")
    pipeline.NamedBarrier(
        barrier_id=self.token_in_size_barrier_id, num_threads=self.transfer_thread_count
    ).arrive_and_wait()
    iket.range_pop()

    iket.range_push("token_in.pull_payload")
    _pull_rows(self, smem_workspace, smem_base, owned_sizes)
    iket.range_pop()


@cute.jit
def _stage_owned_sizes(
    self, owned_sizes: cute.Tensor, transfer_thread_idx: Int32
) -> None:
    sizes = self._router.sizes_tensor(self._device_workspace)
    owner_expert_begin = self._local_rank * Int32(self.experts_per_rank)
    source_sizes = cute.make_tensor(
        sizes.iterator + owner_expert_begin, cute.make_layout((self.experts_per_rank,))
    )
    copy_elements = 4 if self.experts_per_rank % 4 == 0 else 1
    source_size_vectors = cute.zipped_divide(
        (
            mark_alignment(source_sizes, 16)
            if cutlass.const_expr(copy_elements == 4)
            else source_sizes
        ),
        (copy_elements,),
    )
    destination_size_vectors = cute.zipped_divide(owned_sizes, (copy_elements,))
    size_vector_count = cute.size(destination_size_vectors, mode=[1])
    size_copy_atom = _copy_atom(cutlass.Int32, copy_elements * 32)
    size_copy_rounds = ceil_div(size_vector_count, self.transfer_thread_count)
    size_copy_registers = cute.make_rmem_tensor(
        (copy_elements, size_copy_rounds), cutlass.Int32
    )
    for size_copy_round in cutlass.range_constexpr(size_copy_rounds):
        vector_idx = (
            Int32(size_copy_round * self.transfer_thread_count) + transfer_thread_idx
        )
        if vector_idx < Int32(size_vector_count):
            cute.copy(
                size_copy_atom,
                source_size_vectors[None, vector_idx],
                size_copy_registers[None, size_copy_round],
            )
    if cutlass.const_expr(not _tagged_dispatch):
        iket.range_push("token_in.wait_metadata_ready")
        self._router.wait_for_metadata_ready(self._device_workspace)
        iket.range_pop()
    for size_copy_round in cutlass.range_constexpr(size_copy_rounds):
        vector_idx = (
            Int32(size_copy_round * self.transfer_thread_count) + transfer_thread_idx
        )
        if vector_idx < Int32(size_vector_count):
            cute.copy(
                size_copy_atom,
                size_copy_registers[None, size_copy_round],
                destination_size_vectors[None, vector_idx],
            )


@cute.jit
def _pull_rows(
    self, smem_workspace, smem_base: cute.Pointer, owned_sizes: cute.Tensor
) -> None:
    transfer_warp_idx = self._transfer_warp_idx
    lane_idx = self._lane_idx
    global_warp_idx = (
        self._linear_cta_idx * Int32(self.transfer_warp_count) + transfer_warp_idx
    )
    global_warp_count = Int32(
        self.promised_launchable_sm_count * self.transfer_warp_count
    )
    token_metadata_address = self._router.token_src_metadata_pointer(
        self._device_workspace
    ).toint()
    if cutlass.const_expr(_tagged_dispatch):
        tag = _dispatch_tag(self._device_workspace)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            tagged_scores_address = self._device_workspace.ptr(
                _TAGGED_SCORES_REGION
            ).toint()
            pool_scores_address = self._router.fc1_topk_scores_tensor(
                self._device_workspace
            ).iterator.toint()

    token_in_mbarriers = smem_workspace.ptr(self.token_in_mbarrier_region, smem_base)
    token_in_activation = smem_workspace.tensor(
        self.token_in_activation_smem_region, smem_base
    )
    token_in_sf = smem_workspace.tensor(self.token_in_sf_smem_region, smem_base)
    warp_mbarrier = token_in_mbarriers + transfer_warp_idx
    warp_activation_stage = token_in_activation[transfer_warp_idx, None]
    warp_sf_stage = token_in_sf[transfer_warp_idx, (None, None)]
    if lane_idx == Int32(0):
        cute.arch.mbarrier_init(warp_mbarrier, 1)
    cute.arch.sync_warp()

    fc1_activation_pointer = self._device_workspace.ptr(self.fc1_activation_region)
    fc1_activation_sf = self.fc1_activation_sf_tensor(self._device_workspace)
    fc1_ready_counter = self._device_workspace.ptr(self.fc1_ready_region)
    activation_bytes = (
        cute.cosize(warp_activation_stage) * int(self.activation_dtype.width) // 8
    )
    activation_sf_bytes = (
        cute.cosize(warp_sf_stage) * int(self.activation_sf_dtype.width) // 8
    )
    sf_copy_elements = 4
    source_sf_vectors = cute.zipped_divide(
        cute.slice_(warp_sf_stage, (0, None)), (sf_copy_elements,)
    )
    sf_copy_atom = _copy_atom(
        self.activation_sf_dtype, sf_copy_elements * int(self.activation_sf_dtype.width)
    )
    padding = Int32(self._router.token_padding_block)

    next_dense_token = global_warp_idx
    expert_valid_begin = Int32(0)
    expert_pool_begin = Int32(0)
    expert_sf_begin = Int32(0)
    expert_ready_slot_begin = Int32(0)
    pull_phase = Int32(0)
    flag_tracker = make_flag_batch_tracker(
        use_async=self.token_in_flag_batch == 1,
        flag_address=Int64(0),
        accumulated_flags=Int32(0),
        phase=Int32(0),
        thread_idx=lane_idx,
    )

    local_expert = Int32(0)
    while local_expert < Int32(self.experts_per_rank):
        expert_token_count = owned_sizes[local_expert]
        expert_valid_end = expert_valid_begin + expert_token_count
        pull_count = Int32(0)
        if next_dense_token < expert_valid_end:
            pull_count = (
                expert_valid_end - next_dense_token + global_warp_count - Int32(1)
            ) // global_warp_count

        for pull_round in cutlass.range(pull_count, unroll=1):
            dense_token_idx = next_dense_token + Int32(pull_round) * global_warp_count
            token_in_expert = dense_token_idx - expert_valid_begin
            pool_token_idx = expert_pool_begin + token_in_expert
            sf_token_idx = expert_sf_begin + token_in_expert
            ready_slot_idx = expert_ready_slot_begin + token_in_expert // Int32(
                self.tokens_per_fc1_ready_slot
            )

            metadata_address = token_metadata_address + Int64(pool_token_idx) * Int64(
                TokenSrcMetadata.nbytes
            )
            if cutlass.const_expr(_tagged_dispatch):
                metadata = _decode_tagged_metadata(
                    _poll_tagged_word(metadata_address, tag)
                )
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    score_word = _poll_tagged_word(
                        tagged_scores_address + Int64(pool_token_idx) * Int64(8), tag
                    )
                    with cute.arch.elect_one():
                        cute.arch.store(
                            cute.make_ptr(
                                cutlass.Int32,
                                pool_scores_address + Int64(pool_token_idx) * Int64(4),
                                AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            Int32(score_word & Int64(0xFFFFFFFF)),
                        )
            else:
                metadata = TokenSrcMetadata.load(metadata_address)
            peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(
                Int64(0), metadata.src_rank, Int64(0)
            )
            remote_activation_address = (
                self._token_comm_args.activation.iterator.toint()
                + peer_offset
                + Int64(metadata.src_token) * Int64(self.bytes_per_token)
            )
            remote_sf = cute.make_tensor(
                cute.make_ptr(
                    self._token_comm_args.activation_sf.dtype,
                    self._token_comm_args.activation_sf.iterator.toint() + peer_offset,
                    AddressSpace.gmem,
                    assumed_align=(
                        self._token_comm_args.activation_sf.iterator.max_alignment
                    ),
                ),
                self._token_comm_args.activation_sf.layout,
            )
            remote_sf_row = remote_sf[Int64(metadata.src_token), None]

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    warp_mbarrier, Int32(activation_bytes + activation_sf_bytes)
                )
                tma_load_1d(
                    warp_activation_stage.iterator,
                    Int64(remote_activation_address),
                    warp_mbarrier,
                    Int32(activation_bytes),
                )
                tma_load_1d(
                    warp_sf_stage.iterator,
                    remote_sf_row.iterator,
                    warp_mbarrier,
                    Int32(activation_sf_bytes),
                )
            cute.arch.sync_warp()
            cute.arch.mbarrier_wait(warp_mbarrier, pull_phase)

            destination_activation = cute.make_ptr(
                self.activation_dtype,
                fc1_activation_pointer.toint()
                + Int64(pool_token_idx) * Int64(self.bytes_per_token),
                AddressSpace.gmem,
                assumed_align=16,
            )
            with cute.arch.elect_one():
                cp_async_bulk_s2g(
                    destination_activation,
                    warp_activation_stage.iterator,
                    Int32(activation_bytes),
                )
            cute.arch.sync_warp()
            cute.arch.cp_async_bulk_commit_group()

            destination_sf_row = fc1_activation_sf[
                Int64(sf_token_idx), ((None, None), None)
            ]
            destination_sf_values = cute.group_modes(
                cute.slice_(destination_sf_row, (0, None, None)), 0, 2
            )
            destination_sf_vectors = cute.zipped_divide(
                destination_sf_values, (sf_copy_elements,)
            )
            sf_vector_count = cute.size(destination_sf_vectors, mode=[1])
            for sf_round in cutlass.range_constexpr(ceil_div(sf_vector_count, 32)):
                sf_vector_idx = Int32(sf_round * 32) + lane_idx
                if sf_vector_idx < Int32(sf_vector_count):
                    cute.copy(
                        sf_copy_atom,
                        source_sf_vectors[None, sf_vector_idx],
                        destination_sf_vectors[None, sf_vector_idx],
                    )

            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            ready_address = (fc1_ready_counter + ready_slot_idx).toint()
            flag_tracker = flag_tracker.accumulate(
                Int32(0), self.token_in_flag_batch, ready_address
            )
            cute.arch.sync_warp()
            pull_phase = pull_phase ^ Int32(1)

        next_dense_token = next_dense_token + pull_count * global_warp_count
        expert_valid_begin = expert_valid_end
        # The helper's pool_expert_base: the padded runs of the earlier local experts.
        expert_pool_begin = (
            expert_pool_begin
            + (expert_token_count + padding - Int32(1)) // padding * padding
        )
        expert_sf_begin = expert_sf_begin + (
            (expert_token_count + Int32(self.sf_padding_block - 1))
            // Int32(self.sf_padding_block)
        ) * Int32(self.sf_padding_block)
        expert_ready_slot_begin = expert_ready_slot_begin + (
            (expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1))
            // Int32(self.tokens_per_fc1_ready_slot)
        )
        local_expert = local_expert + Int32(1)

    flag_tracker.fire()
    cute.arch.sync_warp()


# ----- Install -----


def _apply() -> None:
    _MetadataPushRouter._register_router_workspace = _register_router_workspace
    _MetadataPushRouter._router_push_metadata = _router_push_metadata
    _MetadataPushRouter._router_helper_single_cta = _router_helper_single_cta
    TokenCommNonDeterministic.token_in = _token_in
    if _tagged_dispatch:
        TokenSrcMetadata.load = _load_tagged_metadata
        TokenCommNonDeterministic.reset_tail = _reset_tail_advancing_epoch


def install_sm107_megamoe_tagged_dispatch() -> None:
    """Send the inference kernel's route metadata as tagged words, once per process.

    Install after the single tail barrier option: this one wraps whichever tail is installed.
    """
    global _tagged_dispatch
    _tagged_dispatch = True
    _apply()


def install_sm107_megamoe_per_cta_push_layout() -> None:
    """Let each router push warp lay out its own routes, once per process."""
    global _per_cta_push_layout
    _per_cta_push_layout = True
    _apply()
