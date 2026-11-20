"""FlashInfer 0.7 CuTe DSL KDA prefill with radix-cache checkpoints."""

from __future__ import annotations

import functools
import importlib.util
import logging
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.managers import pp_bubble_diag

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# SGLANG_K31_KDA_PREFILL_FI_SM107 runs FlashInfer v0.7.4's CuTe DSL KDA kernel module:
# 0.7.2's decomp-chain kernel can deadlock (flashinfer#8684) and pass non-finite V
# padding (#8599). On this path only that module differs, and it is imported lazily.
_BT16 = "flashinfer.kda_kernels.kda_chunked_bt16"
_BT16_V074 = (
    Path(__file__).with_name("flashinfer_kda_v074") / "kda_chunked_bt16.py",
    "65837616b001b3fdd5856bc936526930f4ae100e",
)
_BT16_BLOBS = {
    "38903d429cbcfa05c3363a760f3c32902532e7cf": "0.7.2",
    "65837616b001b3fdd5856bc936526930f4ae100e": "v0.7.4",
}
# kda_prefill_cute.py at 0.7.2 and v0.7.4: the vendored kernel's callers
_PREFILL_CUTE_BLOBS = (
    "150c7edc008ec68d6cf9dd96ac80b62ccace0921",
    "a46ae62155472a982d0b61e4fcce866518882c88",
)


@functools.cache
def _use_flashinfer_kda_bt16_v074() -> str:
    """Load flashinfer.kda_kernels.kda_chunked_bt16 from FlashInfer v0.7.4; returns
    the image's version of that module."""
    from flashinfer import kda_prefill_cute

    from sglang.srt.layers.attention.flashinfer_vendored import (
        git_blob_sha1,
        load_vendored,
    )

    spec = importlib.util.find_spec(_BT16)
    image = spec and _BT16_BLOBS.get(git_blob_sha1(Path(spec.origin).read_bytes()))
    caller = git_blob_sha1(Path(kda_prefill_cute.__file__).read_bytes())
    if image is None or caller not in _PREFILL_CUTE_BLOBS:
        raise RuntimeError(
            "SGLANG_K31_KDA_PREFILL_FI_SM107 needs FlashInfer 0.7.2 or v0.7.4 "
            f"{_BT16} and kda_prefill_cute; the image has other versions"
        )
    if image == "0.7.2":
        load_vendored({_BT16: _BT16_V074})
        logger.info(f"FlashInfer KDA prefill kernel: v0.7.4 from {_BT16_V074[0]}")
    return image


def build_flashinfer_kda_checkpoint_plan(
    forward_batch: ForwardBatch,
    metadata: ForwardMetadata,
    device: torch.device,
    chunk_size: int,
) -> None:
    """Plan scheduler-generated tracked boundaries, asserting their validity.

    Host track metadata follows build_prefill_track_plan's unaligned-row order.
    """
    extend_lens = forward_batch.extend_seq_lens_cpu
    track_lens = forward_batch.mamba_track_seqlens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    track_mask = forward_batch.mamba_prefill_track_mask_cpu

    checkpoint_counts = [length // chunk_size for length in extend_lens]
    checkpoint_starts = list(accumulate(checkpoint_counts, initial=0))
    # NOTE(kpham-sgl): FlashInfer GDN writes all periodic checkpoints for SGLang
    # to select from; FlashInfer KDA writes only mapped destinations (-1 skips).
    checkpoint_destinations = [-1] * checkpoint_starts[-1]
    num_tracked_checkpoints = 0
    for row, tracked in enumerate(track_mask):
        if not tracked:
            continue
        relative_track_len = track_lens[row] - prefix_lens[row]
        if relative_track_len % chunk_size == 0:
            continue
        completed_chunks = relative_track_len // chunk_size
        assert 0 < completed_chunks <= checkpoint_counts[row], (
            f"Invalid KDA checkpoint metadata: row={row}, "
            f"relative_track_len={relative_track_len}, extend_len={extend_lens[row]}, "
            f"chunk_size={chunk_size}"
        )
        checkpoint_destinations[checkpoint_starts[row] + completed_chunks - 1] = (
            num_tracked_checkpoints
        )
        num_tracked_checkpoints += 1

    assert num_tracked_checkpoints == metadata.track_ssm_h_batch_src.numel(), (
        "KDA checkpoint rows disagree with the prefill tracking metadata"
    )
    metadata.state_checkpoint_cu_starts = torch.tensor(
        checkpoint_starts, dtype=torch.int64, pin_memory=True
    ).to(device, non_blocking=True)
    metadata.num_state_checkpoints = num_tracked_checkpoints
    metadata.state_checkpoint_every_n_tokens = chunk_size
    metadata.state_checkpoint_indices = torch.tensor(
        checkpoint_destinations, dtype=torch.int32, pin_memory=True
    ).to(device, non_blocking=True)


class FlashInferKDAPrefillKernel(LinearAttnKernelBase):
    """Execute KDA prefill; FlashInfer validates the tensor contract."""

    uses_state_checkpoints = True
    # FlashInfer memoizes cu_seqlens host contents on tensor identity, but casts an
    # int32 one to a new int64 tensor per call; one int64 tensor per forward hits it.
    wants_int64_query_start_loc = True
    supports_track_state_snapshot = True
    supports_safe_gate = True

    def __init__(self):
        from sglang.srt.environ import envs

        capability = torch.cuda.get_device_capability()
        sm107 = capability == (10, 7) and envs.SGLANG_K31_KDA_PREFILL_FI_SM107.get()
        if capability not in ((10, 0), (10, 3)) and not sm107:
            raise ValueError(
                "FlashInfer KDA prefill requires SM100 or SM103 "
                "(SM107 with SGLANG_K31_KDA_PREFILL_FI_SM107=1)"
            )
        from flashinfer import kda

        if envs.SGLANG_K31_KDA_PREFILL_FI_SM107.get():
            _use_flashinfer_kda_bt16_v074()
        if sm107:
            # flashinfer#6201 (in 0.7.3) only adds (10, 7) to this set. A no-op on
            # FlashInfer >= 0.7.3.
            from flashinfer import kda_prefill_cute

            kda_prefill_cute._SUPPORTED_COMPUTE_CAPABILITIES.add((10, 7))

        self._recurrent_kda = kda.recurrent_kda
        # Older FlashInfer installations retain the eager BCG break.
        self.prefill_graph_wrapper_cls = getattr(
            kda, "RecurrentKDAPrefillWrapper", None
        )

    def decode(self, *args, **kwargs):
        raise NotImplementedError("FlashInferKDAPrefillKernel is prefill-only")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        query_start_loc_i64: Optional[torch.Tensor] = None,
        return_intermediate_states: bool = False,
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        state_checkpoint_indices: Optional[torch.Tensor] = None,
        track_ssm_h_batch_src: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        needs_checkpoint = (
            return_intermediate_states and kwargs.get("track_state") is not None
        )
        checkpoints = (
            ssm_states.new_empty((num_state_checkpoints, *ssm_states.shape[1:]))
            if needs_checkpoint
            else None
        )
        with pp_bubble_diag.kda_wait():
            result = self._recurrent_kda(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                g=g[:, : q.shape[1]].contiguous(),
                beta=beta[:, : q.shape[1]].contiguous(),
                A_log=A_log.reshape(-1).float().contiguous(),
                # FlashInfer's grouped-CTA recurrent kernel (extends with a cached prefix)
                # indexes dt_bias as a flat [H*K] tensor and asserts a unit leading stride.
                dt_bias=dt_bias.reshape(-1).float().contiguous(),
                initial_state=ssm_states,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                lower_bound=lower_bound,
                cu_seqlens=(
                    query_start_loc if query_start_loc_i64 is None else query_start_loc_i64
                ),
                ssm_state_indices=cache_indices.to(torch.int32),
                beta_is_logit=True,
                state_checkpoints=checkpoints,
                checkpoint_cu_starts=(
                    state_checkpoint_cu_starts if needs_checkpoint else None
                ),
                checkpoint_state_indices=(
                    state_checkpoint_indices if needs_checkpoint else None
                ),
                checkpoint_every_n_tokens=(
                    state_checkpoint_every_n_tokens if needs_checkpoint else 0
                ),
                backend="cute-dsl",
            )
        if needs_checkpoint:
            kwargs["track_state"][track_ssm_h_batch_src] = checkpoints.float()
        output = result[0]
        return (output, None) if return_intermediate_states else output
