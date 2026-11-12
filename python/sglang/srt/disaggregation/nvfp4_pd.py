"""Narrow H28 NVFP4 MLA -> native FP8 PD bridge; existing kernels/transport.

Only target/draft KV rows are converted. DSA indexer state and request metadata
retain the original byte-copy paths. No new quantizer or decode kernel is used.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
CAPABILITY = "dsa_nvfp4_to_fp8_unit_v1"
# Poisoned transfer handles can outlive their manager during a role switch.
# Keep allocation/pool ownership until process exit; never recycle these buffers.
_QUARANTINED_SENDERS = []
PAGE_SIZE = 64
PACKED_ROW = 324
FP8_ROW = 576


def effective_unit_scales(attention_layers):
    """Match DSA's None -> 1 scale rule; verify loaded layers, not config guesses."""
    manifest = []
    for layer in attention_layers:
        scales = tuple(
            1.0 if getattr(layer, name, None) is None else float(getattr(layer, name))
            for name in ("k_scale_float", "v_scale_float")
        )
        if scales != (1.0, 1.0):
            raise ValueError(
                f"H28 mixed PD requires effective unit FP8 scales, "
                f"layer={layer.layer_id}, k/v={scales}"
            )
        manifest.append((int(layer.layer_id), *scales))
    if not manifest:
        raise ValueError("H28 mixed PD could not identify loaded attention layers")
    return manifest


def _pool_entries(target, draft, total_layers, pointers, *, nvfp4):
    """Bind actual registered pointers to owning pool and local/global layer IDs."""
    entries = {}
    for is_draft, pool in ((False, target), (True, draft)):
        if pool is None:
            continue
        if (
            pool.page_size != PAGE_SIZE
            or pool.kv_lora_rank + pool.qk_rope_head_dim != FP8_ROW
        ):
            raise ValueError("H28 mixed PD supports 64-token, 576-wide MLA only")
        ptrs, lengths, strides = pool.get_contiguous_buf_infos()
        if len(ptrs) != pool.layer_num or len(pool.kv_buffer) != len(ptrs):
            raise ValueError("H28 mixed PD requires one KV buffer per layer")
        for i, (ptr, length, stride, tensor) in enumerate(
            zip(ptrs, lengths, strides, pool.kv_buffer)
        ):
            expected = PAGE_SIZE * (PACKED_ROW if nvfp4 else FP8_ROW)
            if ptr != tensor.data_ptr() or stride != expected or length % expected:
                raise ValueError("H28 mixed PD KV pointer/stride layout mismatch")
            layer_id = int(pool.start_layer + i)
            wire_id = total_layers + i if is_draft else layer_id
            if ptr in entries:
                raise ValueError("H28 mixed PD encountered aliased KV entries")
            entries[ptr] = (pool, layer_id, wire_id, length // expected)
    if len(entries) != len(pointers) or set(entries) != set(pointers):
        raise ValueError(
            "H28 mixed PD descriptors do not cover registered target/draft KV"
        )
    return [entries[ptr] for ptr in pointers]


def bind_source(args, target, draft, total_layers):
    from sglang.srt.mem_cache.memory_pool import DSANVFP4TokenToKVPool

    if not isinstance(target, DSANVFP4TokenToKVPool):
        return
    if draft is not None and not isinstance(draft, DSANVFP4TokenToKVPool):
        raise ValueError("H28 mixed PD requires target/draft NVFP4 KV on prefill")
    args.nvfp4_pd_sources = _pool_entries(
        target, draft, total_layers, args.kv_data_ptrs, nvfp4=True
    )
    logger.info(
        "H28 mixed PD prefill source: target/draft KV entries=%s; packed row=324; indexer/state unchanged",
        [(entry[2], entry[3]) for entry in args.nvfp4_pd_sources],
    )


def bind_destination(args, target, draft, total_layers, models):
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    if args.kv_cache_dtype_str not in ("fp8_e4m3", "fp8_e4m3fn") or not isinstance(
        target, DSATokenToKVPool
    ):
        return
    if draft is not None and not isinstance(draft, DSATokenToKVPool):
        return
    layers = [
        module
        for model in models
        for module in model.modules()
        if isinstance(module, RadixAttention)
    ]
    try:
        scales = effective_unit_scales(layers)
        entries = _pool_entries(
            target, draft, total_layers, args.kv_data_ptrs, nvfp4=False
        )
    except ValueError as error:
        # Ordinary FP8 peers remain valid; mixed-dtype bootstrap stays rejected.
        args.nvfp4_pd_unsupported = str(error)
        logger.info("H28 mixed PD destination unsupported: %s", error)
        return
    args.nvfp4_pd_destination = {
        "capability": CAPABILITY,
        "dtype": args.kv_cache_dtype_str,
        "ids": [entry[2] for entry in entries],
        "page_capacities": [entry[3] for entry in entries],
        "page_size": PAGE_SIZE,
        "row_bytes": FP8_ROW,
        "effective_unit_scales": True,
    }
    logger.info(
        "H28 mixed PD native FP8 destination loaded scales verified: %s; entries=%s",
        scales,
        args.nvfp4_pd_destination["ids"],
    )


def validate_peer(
    sources, source_strides, destination, destination_ptrs, destination_strides
):
    """No general dtype/layout bypass: allow this exact conversion domain only."""
    if (
        not destination
        or destination.get("capability") != CAPABILITY
        or destination.get("dtype") not in ("fp8_e4m3", "fp8_e4m3fn")
        or destination.get("page_size") != PAGE_SIZE
        or destination.get("row_bytes") != FP8_ROW
        or destination.get("effective_unit_scales") is not True
    ):
        raise ValueError("H28 mixed PD peer lacks native unit-scale FP8 capability")
    ids = destination.get("ids", [])
    capacities = destination.get("page_capacities", [])
    if (
        len(ids) != len(destination_ptrs)
        or len(capacities) != len(ids)
        or len(destination_strides) != len(ids)
        or len(set(ids)) != len(ids)
        or len(sources) != len(source_strides)
        or any(type(n) is not int or n <= 0 for n in capacities)
    ):
        raise ValueError("H28 mixed PD peer entry lists are inconsistent")
    by_id = {layer_id: i for i, layer_id in enumerate(ids)}
    pairs = []
    for i, (_, _, wire_id, _) in enumerate(sources):
        j = by_id.get(wire_id)
        if (
            j is None
            or source_strides[i] != PACKED_ROW * PAGE_SIZE
            or destination_strides[j] != FP8_ROW * PAGE_SIZE
        ):
            raise ValueError("H28 mixed PD target/draft layer or page stride mismatch")
        pairs.append((i, j))
    return pairs


def plan_page_blocks(
    source_pages, destination_pages, source_capacity, destination_capacity
):
    """Dense scratch page ordinal -> real destination page; numpy integers allowed."""
    import operator

    try:
        src = [operator.index(p) for p in source_pages]
        dst = [operator.index(p) for p in destination_pages]
    except TypeError as error:
        raise ValueError("H28 mixed PD page IDs must be integers") from error
    if (
        len(src) != len(dst)
        or any(p < 0 or p >= source_capacity for p in src)
        or any(p < 0 or p >= destination_capacity for p in dst)
    ):
        raise ValueError("H28 mixed PD page vector exceeds registered KV buffer")
    blocks = []
    stride = PAGE_SIZE * FP8_ROW
    for ordinal, page in enumerate(dst):
        start = ordinal * stride
        end = page * stride
        if (
            blocks
            and blocks[-1][0] + blocks[-1][2] == start
            and blocks[-1][1] + blocks[-1][2] == end
        ):
            old = blocks[-1]
            blocks[-1] = (old[0], old[1], old[2] + stride)
        else:
            blocks.append((start, end, stride))
    return src, blocks


class NVFP4PDSender:
    """One transfer worker owns this scratch until transport has settled.

    Failed/unsettled sends quarantine this object in its manager until process
    teardown. They cannot reuse/deregister scratch or claim a successful ACK.
    """

    def __init__(self, manager):
        import torch
        from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
        from sglang.srt.disaggregation.common.staging_handler import (
            _get_custom_mem_pool,
        )
        from sglang.srt.runtime_context import get_schedule

        self.manager = manager
        self.sources = manager.kv_args.nvfp4_pd_sources
        tokens = get_schedule().chunked_prefill_size
        if tokens is None or tokens <= 0 or tokens % PAGE_SIZE:
            raise ValueError(
                "H28 mixed PD requires positive page-aligned chunked prefill size"
            )
        self.max_pages = tokens // PAGE_SIZE
        self.rows = tokens
        self.bytes_per_layer = tokens * FP8_ROW
        device = f"cuda:{manager.kv_args.gpu_id}"
        pool, pool_type = _get_custom_mem_pool(device)
        if pool is None or pool_type != "NVLINK":
            raise ValueError("H28 mixed PD requires the configured NVLINK custom pool")
        self.buffer = StagingBuffer(
            self.bytes_per_layer * len(self.sources),
            device,
            manager.kv_args.gpu_id,
            custom_mem_pool=pool,
        )
        self.output = self.buffer.buffer.view(torch.float8_e4m3fn).view(
            len(self.sources), tokens, 1, FP8_ROW
        )
        self.indices = torch.empty((1, tokens), dtype=torch.int32, device=device)
        self.remap = torch.empty_like(self.indices)
        self.offsets = torch.arange(PAGE_SIZE, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        self.poisoned = False
        self._logged_send = False
        if (
            manager.engine.batch_register(
                [self.buffer.get_ptr()], [self.buffer.get_size()]
            )
            != 0
        ):
            raise RuntimeError("H28 mixed PD scratch registration failed")
        logger.info(
            "H28 mixed PD registered worker scratch: GPU=%s bytes=%s entries=%s; MNNVL/NVLINK",
            manager.kv_args.gpu_id,
            self.buffer.get_size(),
            len(self.sources),
        )

    def send(
        self,
        session,
        source_pages,
        destination_ptrs,
        destination_pages,
        destination_strides,
        destination,
        executor,
        bootstrap_room,
    ):
        import torch
        from sglang.kernels.ops.attention.dsa.nvfp4_mla_cache import (
            gather_dequant_nvfp4_mla_cache_generation,
        )

        if self.poisoned:
            raise RuntimeError(
                "H28 mixed PD worker scratch is quarantined after an unsettled send"
            )
        pairs = validate_peer(
            self.sources,
            self.manager.kv_args.kv_item_lens,
            destination,
            destination_ptrs,
            destination_strides,
        )
        if len(source_pages) != len(destination_pages):
            raise ValueError("H28 mixed PD source/destination vector length mismatch")
        try:
            for begin in range(0, len(source_pages), self.max_pages):
                src_pages = source_pages[begin : begin + self.max_pages]
                dst_pages = destination_pages[begin : begin + self.max_pages]
                page_count = len(src_pages)
                plans = []
                for i, j in pairs:
                    _, blocks = plan_page_blocks(
                        src_pages,
                        dst_pages,
                        self.sources[i][3],
                        destination["page_capacities"][j],
                    )
                    plans.append((i, j, blocks))
                rows = page_count * PAGE_SIZE
                with torch.cuda.stream(self.stream):
                    pages = torch.as_tensor(
                        src_pages, dtype=torch.int32, device=self.output.device
                    )
                    slots = (pages[:, None] * PAGE_SIZE + self.offsets).reshape(1, rows)
                    self.indices[:, :rows].copy_(slots)
                    for i, _, _ in plans:
                        pool, layer_id, _, _ = self.sources[i]
                        packed, scale = pool.get_nvfp4_mla_buffers(layer_id)
                        gather_dequant_nvfp4_mla_cache_generation(
                            packed,
                            self.indices[:, :rows],
                            scale,
                            head_dim=FP8_ROW,
                            page_size=PAGE_SIZE,
                            output=self.output[i, :rows],
                            compact_indices=self.remap[:, :rows],
                        )
                self.stream.synchronize()
                layer_blocks = []
                for i, j, blocks in plans:
                    base = self.buffer.get_ptr() + i * self.bytes_per_layer
                    layer_blocks.append(
                        [
                            (base + src, destination_ptrs[j] + dst, size)
                            for src, dst, size in blocks
                        ]
                    )
                # Match this GB043 manager's native NVLINK path: one transfer
                # future per local layer (send_kvcache), including the draft.
                futures = [
                    executor.submit(self.manager._transfer_data, session, group)
                    for group in layer_blocks
                ]
                status = self.manager._await_transfer_futures(futures)
                if status != 0:
                    self._quarantine(bootstrap_room)
                    return status
        except BaseException:
            self._quarantine(bootstrap_room)
            raise
        if not self._logged_send and len(source_pages):
            logger.info(
                "H28 mixed PD sent target/draft KV: pages=%s entries=%s; 324->576; state/aux unchanged",
                len(source_pages),
                len(pairs),
            )
            self._logged_send = True
        return 0

    def _quarantine(self, bootstrap_room):
        if not self.poisoned:
            self.poisoned = True
            _QUARANTINED_SENDERS.append(self)
        self.manager._poison_unsettled_send(bootstrap_room)

    def close(self):
        if self.poisoned:
            return
        self.stream.synchronize()
        self.manager.engine.batch_deregister([self.buffer.get_ptr()])
        self.output = None
        self.buffer = None
