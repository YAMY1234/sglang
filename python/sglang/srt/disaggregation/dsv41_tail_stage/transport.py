"""The tail server's Mooncake side: PP rank 1 toward decode, write target for stage 0."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.dsv41_tail_stage import protocol as P
from sglang.srt.disaggregation.dsv41_tail_stage.forward import (
    TailRun,
    TailStageForward,
)
from sglang.srt.disaggregation.dsv41_tail_stage.layout import (
    c1_buf_infos,
    item_lens_error,
    tail_swa_buf_infos,
)
from sglang.srt.disaggregation.dsv41_tail_stage.registration import (
    TAIL_STAGE_PP_RANK,
    stage1_state_indices,
    tail_entries,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.utils import (
    KVClassType,
    TransferBackend,
    append_state_component,
    get_kv_class,
    get_pd_kv_layout,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_device, get_disagg

logger = logging.getLogger(__name__)

_NO_PAGES = np.zeros(0, dtype=np.int32)


class TailStageKVManager(MooncakeKVManager):
    """PP rank 1 of every encoder-only prefill's bootstrap table. The service
    registers it with each stage 0; it sends no KV pages and no metadata row, and
    pairs only the state entries it sends."""

    owned_entries: Sequence[int] = ()

    def register_to_bootstrap(self):
        pass

    def validate_aux_buffers(self, dst_aux_ptrs, dst_aux_item_lens) -> None:
        pass

    def _validate_peer_state_layout(
        self, registration_info
    ) -> Tuple[bool, Optional[str]]:
        for i in self.owned_entries:
            dst = registration_info.dst_state_item_lens
            err = item_lens_error(
                f"state entry {i} ({self.kv_args.state_types[i].value})",
                self.kv_args.state_item_lens[i],
                dst[i] if i < len(dst) else [],
            )
            if err is not None:
                return False, err
        return True, None

    def send_kvcache(self, mooncake_session_id, prefill_kv_indices, *args, **kwargs):
        assert len(prefill_kv_indices) == 0, "the tail stage sends no KV pages"
        return 0


def tail_kv_args(
    fwd: TailStageForward, *, stage0_wire: List[list], start_layer: int, end_layer: int
) -> KVArgs:
    """W0 as empty entries (stage 0 sends them), then the entries stage 1 sends."""
    pool = fwd.model_runner.token_to_kv_pool
    kv_args = get_kv_class(TransferBackend.MOONCAKE, KVClassType.KVARGS)()
    kv_args.engine_rank = 0
    kv_args.pp_rank = TAIL_STAGE_PP_RANK
    kv_args.system_dp_rank = 0
    kv_args.rust_http_port = None
    kv_args.kv_cache_dtype_str = fwd.model_runner.kv_cache_dtype_str
    kv_args.kv_layout = get_pd_kv_layout(pool)
    kv_args.prefill_start_layer = start_layer
    kv_args.prefill_end_layer = end_layer
    kv_args.kv_data_ptrs, kv_args.kv_data_lens, kv_args.kv_item_lens = [], [], []
    kv_args.kv_layer_ids = []
    kv_args.num_draft_entries = 0
    kv_args.page_size = pool.page_size
    kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = [], [], []
    kv_args.ib_device = get_disagg().disaggregation_ib_device
    kv_args.gpu_id = get_device().gpu_id
    kv_args.state_types = []
    kv_args.state_sub_pools = []
    kv_args.state_wire_keys = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []
    kv_args.state_slice_outer_counts = []
    kv_args.state_layer_ids = []
    kv_args.state_conv_shard_groups = []
    kv_args.dsa_index_k_token_bytes = None
    kv_args.swa_kv_cache_layout = None
    kv_args.mla_compression_ratios = None
    kv_args.kv_packable = False
    kv_args.is_hybrid_mla_backend = False
    for entry in stage0_wire:
        append_state_component(
            kv_args, StateType(entry[0]), [], [], [], wire_key=entry[1]
        )
    swa = tail_swa_buf_infos(pool, range(start_layer, end_layer))
    append_state_component(
        kv_args, StateType.DSV41_TAIL_SWA, swa.ptrs, swa.lens, swa.item_lens
    )
    if fwd.aux is not None:
        aux = fwd.aux
        append_state_component(
            kv_args,
            StateType.DSV41_TAIL_DRAFT_AUX,
            [aux.data_ptr()],
            [aux.nbytes],
            [aux[0].nbytes],
        )
    kv_args.state_window_tails = [
        entry[2] if len(entry) > 2 else None for entry in stage0_wire
    ] + [None] * len(tail_entries(fwd.aux is not None))
    return kv_args


class NullDecodeSender:
    """SGLANG_DSV41_TAIL_STAGE_NULL_DECODE (rate benchmark only): a decode that is
    always ready and takes every send at once."""

    def __init__(self) -> None:
        self.status = KVPoll.WaitingForInput

    def poll(self) -> KVPoll:
        return self.status

    def pop_decode_prefix_len(self) -> int:
        return 0

    def init(self, *args, **kwargs) -> None:
        pass

    def send(self, *args, **kwargs) -> None:
        self.status = KVPoll.Success


class MooncakeTailBackend:
    def __init__(self, fwd: TailStageForward, server_args) -> None:
        self.fwd = fwd
        self.memory = fwd.memory
        self.server_args = server_args
        self.mgr: Optional[TailStageKVManager] = None
        self.stage0_wire_len = 0
        model = fwd.model_runner.model.model
        self.start_layer = model.tail_stage_start_layer
        self.end_layer = model.end_layer
        # Forwards run here, so the loop's reads on the default stream (a grant's c1
        # page ids, prepare) never wait behind them.
        self._fwd_stream = torch.cuda.Stream()
        self._busy_ms = 0.0
        self._busy_batches = 0
        self._busy_since = time.monotonic()

    def start(self, stage0_wire: List[list]) -> Tuple[str, int, P.Welcome]:
        kv_args = tail_kv_args(
            self.fwd,
            stage0_wire=stage0_wire,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )
        self.stage0_wire_len = len(stage0_wire)
        self.mgr = TailStageKVManager(
            kv_args, DisaggregationMode.PREFILL, self.server_args, True
        )
        self.mgr.owned_entries = range(self.stage0_wire_len, len(kv_args.state_types))
        kv, index = c1_buf_infos(self.fwd.model_runner.token_to_kv_pool)
        residual = self.fwd.residual
        # Stage 0 writes the c1 pages and the residual slots.
        self.mgr.engine.batch_register(
            kv.ptrs + index.ptrs + [residual.data_ptr()],
            kv.lens + index.lens + [residual.nbytes],
        )
        welcome = P.Welcome(
            tail_session=self.mgr.get_session_id(),
            c1_kv_ptrs=kv.ptrs,
            c1_kv_item_lens=kv.item_lens,
            c1_index_ptrs=index.ptrs,
            c1_index_item_lens=index.item_lens,
            residual_ptr=residual.data_ptr(),
            residual_item_len=residual[0].nbytes,
            page_size=self.fwd.page_size,
            num_slots=residual.shape[0],
            tail_state_item_lens=[
                list(kv_args.state_item_lens[i]) for i in self.mgr.owned_entries
            ],
        )
        host = envs.SGLANG_DSV41_TAIL_STAGE_HOST.get() or self.mgr.local_ip
        logger.info(
            "DSV4.1 tail stage: PP rank 1 at %s:%d, %d state entries (%d stage 0's)",
            host,
            self.mgr.rank_port,
            len(kv_args.state_types),
            self.stage0_wire_len,
        )
        return host, self.mgr.rank_port, welcome

    def make_sender(self, bootstrap_addr: str, room: int) -> MooncakeKVSender:
        return MooncakeKVSender(
            mgr=self.mgr, bootstrap_addr=bootstrap_addr, bootstrap_room=room
        )

    def send(self, sender: MooncakeKVSender, tail) -> None:
        sender.pop_decode_prefix_len()
        sender.init(0, aux_index=0)
        swa_rows = None
        if tail.slot >= 0:
            swa_rows = self.memory.held[tail.room].swa_payload
        sender.send(
            _NO_PAGES,
            state_indices=stage1_state_indices(
                stage0_wire_len=self.stage0_wire_len,
                draft_aux=self.fwd.aux is not None,
                swa_rows=swa_rows,
                aux_slot=tail.slot if swa_rows is not None else None,
            ),
        )

    def run(self, tails):
        """Launch the forward without waiting for it; the check turns True once it has
        run, so the transfer threads may read the windows' SWA rows and the aux."""
        held = self.memory.held
        runs = [
            TailRun(slot=t.slot, s=t.s, e=t.e, window_full=held[t.room].window_full)
            for t in tails
        ]
        prepared = self.fwd.prepare(runs)
        self._fwd_stream.wait_stream(torch.cuda.current_stream())
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        with torch.cuda.stream(self._fwd_stream):
            start.record()
            self.fwd.execute(runs, prepared)
            end.record()

        # The default argument keeps `prepared` alive until the forward that reads it ran.
        def done(prepared=prepared) -> bool:
            if not end.query():
                return False
            self._account(start.elapsed_time(end))
            return True

        return done

    def _account(self, forward_ms: float) -> None:
        self._busy_ms += forward_ms
        self._busy_batches += 1
        now = time.monotonic()
        if now - self._busy_since >= 5.0:
            logger.info(
                "DSV4.1 tail stage GPU forward busy %.0f%% (%d forwards, %.1f ms mean)",
                100 * self._busy_ms / ((now - self._busy_since) * 1e3),
                self._busy_batches,
                self._busy_ms / self._busy_batches,
            )
            self._busy_ms, self._busy_batches, self._busy_since = 0.0, 0, now
