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
"""Opt-in single tail barrier for FlashInfer 0.7.3's SM107 inference MegaMOE kernel.

FlashInfer compiles the SM107 kernel with CuTe DSL when a workspace first
launches, so the option replaces a method of the kernel's token-communication
component before then; CuTe DSL keys its compile cache on the traced IR, so the
changed kernel never reuses a stock binary. Data movement and arithmetic are
unchanged: outputs are bit-identical to the stock kernel.

The stock tail joins the grid, runs a drain NVLink barrier, lets CTA 0 zero the
per-launch counters, then runs a second barrier to publish the zeros. Peers
write the shared counters (expert sizes and the two ready flags) only from
their router kernel, before this rank's token-in passes its readiness waits, so
CTA 0 can zero them as soon as every local CTA has joined. One barrier then
both drains the launch and publishes the zeros: a peer signals only after its
own zeros, and starts its next launch only after every signal. When one CTA
cannot clear the counters (or under FlashInfer's ncu replay mode) the stock
tail runs. Every EP rank must make the same choice.

The per-K-tile FC1 readiness option lets FC2 start a K tile as soon as the FC1
tiles that write it have published. Stock FC2 waits once, before its K loop,
until all 32 FC1 tiles of the token block have. FC2 K tile k (256 intermediate
columns) reads only what FC1 tiles 4k..4k+3 (64 columns and their scale
factors each) wrote, so each token-block slot gets 8 counters in place of one.
The FC1 epilogue bumps tile m's counter m // 4 at the stock publication point.
The scheduler's peek and FC2's B producer read all 8 counters, so every bump is
observed before the launch's tail zeroes them. Loads, math and stores are
unchanged: outputs are bit-identical. Every EP rank must make the same choice.
"""

import hashlib
import importlib
import logging
import os
import re
import tempfile

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, T, dsl_user_op
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import bootstrap_paths

# Puts FlashInfer's vendored SM107 kernel tree (package ``sources``) on sys.path.
bootstrap_paths()

from sources.communication.nvlink_domain.token_comm import (  # noqa: E402
    TokenCommNonDeterministic,
)
from sources.helpers.iket_compat import iket  # noqa: E402

logger = logging.getLogger(__name__)

_stock_reset_tail = TokenCommNonDeterministic.reset_tail
_installed = False
_logged_tails = set()


def _log_compiled_tail(single_barrier: bool, reset_bytes: int) -> None:
    # Runs while CuTe DSL traces the kernel, once per compiled variant.
    if single_barrier in _logged_tails:
        return
    _logged_tails.add(single_barrier)
    if single_barrier:
        logger.info(
            "FlashInfer MegaMOE SM107 kernel tail: one NVLink barrier "
            f"(CTA 0 resets {reset_bytes} bytes)."
        )
    else:
        logger.info(
            "FlashInfer MegaMOE SM107 kernel tail: stock two NVLink barriers "
            f"(the {reset_bytes}-byte reset is not single-CTA)."
        )


@cute.jit
def _reset_tail_single_barrier(self) -> None:
    # self: the kernel's TokenCommNonDeterministic (stock: its reset_tail).
    _log_compiled_tail(
        self.tail_reset_single_cta,
        sum(
            self._device_workspace.tail_reset_bytes(space)
            for space in ("local", "shared")
        ),
    )
    if cutlass.const_expr(not self.tail_reset_single_cta):
        _stock_reset_tail(self)
    else:
        transfer_thread_idx = self._transfer_warp_idx * Int32(32) + self._lane_idx
        iket.range_push("tail.grid_join")
        # After this join no CTA of this rank reads the per-launch state again.
        self._nvlink_barrier.sync(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
        )
        iket.range_pop()
        iket.range_push("tail.reset_workspace")
        if self._linear_cta_idx == Int32(0):
            self._device_workspace.reset_tail_space(
                "shared", transfer_thread_idx, self.transfer_thread_count
            )
            self._device_workspace.reset_tail_space(
                "local", transfer_thread_idx, self.transfer_thread_count
            )
        iket.range_pop()
        iket.range_push("tail.nvlink_drain_publish")
        # CTA 0 joins its threads and fences before it signals the peers.
        self._nvlink_barrier.arrive_and_wait(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
            prologue_grid_sync=False,
            epilogue_grid_sync=False,
        )
        iket.range_pop()


def install_sm107_megamoe_single_tail_barrier() -> None:
    """Patch the SM107 inference kernel's tail once per process."""
    global _installed
    if not _installed:
        TokenCommNonDeterministic.reset_tail = _reset_tail_single_barrier
        _installed = True


# Per-K-tile FC1 readiness: FlashInfer 0.7.3 inference kernel at intermediate 2048.
_FC1KB_K_TILES = 8  # FC2 K tiles of 256 intermediate columns
_FC1KB_TILES_PER_K = 4  # FC1 CTA tiles (64 intermediate columns each) per FC2 K tile
_FC1KB_INTERMEDIATE = 2048
_FC1KB_ALL = (1 << _FC1KB_K_TILES) - 1
_fc1_kblock_wait_installed = None  # None, "fence" or "acquire_v8"


@cute.jit
def _fc1kb_ready_mask(base: cute.Pointer) -> Int32:
    # Bit k: all FC1 tiles of K tile k have published. Relaxed loads; callers fence before relying on a set bit.
    mask = Int32(0)
    for k in cutlass.range_constexpr(_FC1KB_K_TILES):
        if cute.arch.load(base + k, Int32, sem="relaxed", scope="gpu") >= Int32(_FC1KB_TILES_PER_K):
            mask = mask | Int32(1 << k)
    return mask


@dsl_user_op
def _fc1kb_ld_acquire_v8(base, *, loc=None, ip=None):
    # One slot's 8 counters in one 256-bit acquire load (32-byte aligned; PTX ld .v8.b32, global, sm_100+).
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * _FC1KB_K_TILES),
        [base.toint(loc=loc, ip=ip).ir_value()],
        "ld.acquire.gpu.global.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [k])) for k in range(_FC1KB_K_TILES))


@cute.jit
def _fc1kb_ready_mask_acq(base: cute.Pointer) -> Int32:
    # As _fc1kb_ready_mask, from one acquire load: a set bit needs no fence (each element read is an acquire).
    values = _fc1kb_ld_acquire_v8(base)
    mask = Int32(0)
    for k in cutlass.range_constexpr(_FC1KB_K_TILES):
        if values[k] >= Int32(_FC1KB_TILES_PER_K):
            mask = mask | Int32(1 << k)
    return mask


_FC1KB_UNSUPPORTED = (
    "per-K-tile FC1 readiness supports intermediate 2048 with 64-column FC1 tiles and 256-column FC2 K tiles"
)

# (module, class, method, [(stock text, replacement), ...]); each stock text occurs exactly once in the method.
_FC1KB_EDITS = (
    (
        "block_scaled_swap_ab_mega_moe_kernel",
        "BlockScaledSwapAbMegaMoeKernel",
        "_build_device_workspace",
        (("            (counter_slot_count,),\n", "            (counter_slot_count * _FC1KB_K_TILES,),\n"),),
    ),
    (
        "block_scaled_swap_ab_fc12_extension",
        "BlockScaledSwapAbFc12Extension",
        "prepare_work_tile",
        (
            (
                "                counter_pointer = self.fc1_done_counter_pointer + counter_slot\n"
                "                peek_flag = Int32(0)\n"
                "                if spin_peek(counter_pointer, lambda value: value >= self.fc2_spin_threshold):\n"
                "                    peek_flag = Int32(peek_ready_bit)\n",
                "                counter_pointer = self.fc1_done_counter_pointer + counter_slot * Int32(_FC1KB_K_TILES)\n"
                "                peek_flag = Int32(0)\n"
                "                if _fc1kb_ready_mask(counter_pointer) == Int32(_FC1KB_ALL):\n"
                "                    cute.arch.fence_acq_rel_gpu()\n"
                "                    peek_flag = Int32(peek_ready_bit)\n",
            ),
        ),
    ),
    (
        "block_scaled_swap_ab_fc12_epilogue",
        "SwapABFc1Epilogue",
        "signal_fc1_done",
        (
            (
                "        slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx\n"
                "        flag_address = Int64(0)\n"
                "        if in_bound:\n"
                "            flag_address = (self.fc1_done_counter.iterator + slot).toint()\n"
                "        return flag_tracker.accumulate(next_work_tile_info.phase, self.fc1_epi_flag_batch, flag_address)\n",
                "        if cutlass.const_expr(\n"
                "            needs_intermediate_guard\n"
                "            or self.intermediate_downproj != _FC1KB_INTERMEDIATE\n"
                "            or self._EpilogueFc1IntermediateDownTileSize * _FC1KB_TILES_PER_K * _FC1KB_K_TILES\n"
                "            != _FC1KB_INTERMEDIATE\n"
                "        ):\n"
                "            raise NotImplementedError(_FC1KB_UNSUPPORTED)\n"
                "        slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx\n"
                "        k_tile_counter = (\n"
                "            self.fc1_done_counter.iterator\n"
                "            + slot * Int32(_FC1KB_K_TILES)\n"
                "            + work_tile_info.tile_m_idx // Int32(_FC1KB_TILES_PER_K)\n"
                "        )\n"
                "        if cute.arch.make_warp_uniform(self.warp_idx) == Int32(0):\n"
                "            with cute.arch.elect_one():\n"
                "                red_async_add_release_gpu_s32(\n"
                "                    cute.make_ptr(cutlass.Int32, k_tile_counter.toint(), AddressSpace.gmem, assumed_align=4),\n"
                "                    Int32(1),\n"
                "                )\n"
                "        # No all-tiles counter: the tracker keeps its batching state and fires nothing for FC1.\n"
                "        return flag_tracker.accumulate(next_work_tile_info.phase, self.fc1_epi_flag_batch, Int64(0))\n",
            ),
        ),
    ),
    (
        "block_scaled_swap_ab_fc12_mainloop",
        "BlockScaledSwapAbFc12Mainloop",
        "run_tma_b",
        (
            (
                "                counter_pointer = fc1_done_counter_pointer + counter_slot\n"
                "                iket.range_push(\"tma_token_fc2_wait\")\n"
                "                spin_wait(\n"
                "                    counter_pointer,\n"
                "                    lambda value: value >= fc2_spin_threshold,\n"
                "                    sleep_cycles=500,\n"
                "                    peek_status=work_tile.peek_ready,\n"
                "                )\n"
                "                iket.range_pop()\n",
                "                fc1kb_base = fc1_done_counter_pointer + counter_slot * Int32(_FC1KB_K_TILES)\n"
                "                iket.range_push(\"tma_token_fc2_wait\")\n"
                "                # Read every K tile's counter once, unless the scheduler's peek already saw all ready.\n"
                "                fc1kb_ready = Int32(_FC1KB_ALL)\n"
                "                if not work_tile.peek_ready:\n"
                "                    fc1kb_ready = _fc1kb_ready_mask(fc1kb_base)\n"
                "                    cute.arch.fence_acq_rel_gpu()\n"
                "                iket.range_pop()\n",
            ),
            (  # at the method's top level: CuTe DSL refuses a raise inside a staged if / loop
                '        """Run the TMA-B warp that loads token data and scale factors."""\n',
                '        """Run the TMA-B warp that loads token data and scale factors."""\n'
                "        if cutlass.const_expr(\n"
                "            self.fc2_k_tile_count != _FC1KB_K_TILES or self.mma_tile_k != 256 or self.cta_tile_m != 128\n"
                "        ):\n"
                "            raise NotImplementedError(_FC1KB_UNSUPPORTED)\n",
            ),
            (
                "                    b_issue_atom = bind_executable_tma_load_fields(\n"
                "                        fc2_tma_b_exec_atom,\n",
                "                    if ((fc1kb_ready >> k_tile_idx) & Int32(1)) == Int32(0):\n"
                "                        spin_wait(\n"
                "                            fc1kb_base + k_tile_idx,\n"
                "                            lambda value: value >= Int32(_FC1KB_TILES_PER_K),\n"
                "                            sleep_cycles=500,\n"
                "                        )\n"
                "                    b_issue_atom = bind_executable_tma_load_fields(\n"
                "                        fc2_tma_b_exec_atom,\n",
            ),
        ),
    ),
)


_FC1KB_FENCE_LINE = "                    cute.arch.fence_acq_rel_gpu()\n"


def _fc1kb_edits(acquire_v8: bool):
    """The D3 edits, or with acquire_v8 the same edits reading the counters with one acquire v8 load and no fence
    (the counter region 32-byte aligned for it)."""
    if not acquire_v8:
        return _FC1KB_EDITS
    out, fences = [], 0
    for module_name, class_name, method, edits in _FC1KB_EDITS:
        new_edits = []
        for stock, replacement in edits:
            fences += replacement.count(_FC1KB_FENCE_LINE)
            replacement = replacement.replace(_FC1KB_FENCE_LINE, "").replace("_fc1kb_ready_mask(", "_fc1kb_ready_mask_acq(")
            new_edits.append((stock, replacement))
        if method == "_build_device_workspace":
            new_edits.append(("            byte_alignment=16,\n", "            byte_alignment=32,\n"))
        out.append((module_name, class_name, method, tuple(new_edits)))
    assert fences == 2, fences
    return tuple(out)


def _fc1kb_method_source(source: str, method: str) -> str:
    """One method of a module's class (decorators included), at class indentation."""
    lines = source.splitlines(keepends=True)
    head = next(i for i, line in enumerate(lines) if line.startswith(f"    def {method}("))
    start = head
    while lines[start - 1].startswith("    @"):
        start -= 1
    end = head + 1
    while end < len(lines) and not re.match(r"(    @|    def |\S)", lines[end]):
        end += 1
    return "".join(lines[start:end])


def _fc1kb_rebuild(module_name: str, class_name: str, method: str, edits) -> None:
    from cutlass.cute.typing import AddressSpace
    from cutlass.cutlass_dsl import Int64
    from sources.helpers.ptx_helpers import red_async_add_release_gpu_s32

    module = importlib.import_module("sources.kernel_src.rubin.inference.mega." + module_name)
    with open(module.__file__) as f:
        text = _fc1kb_method_source(f.read(), method)
    for stock, replacement in edits:
        if text.count(stock) != 1:
            raise RuntimeError(f"{module_name}.{method}: expected one {stock.strip()[:80]!r}, found {text.count(stock)}")
        text = text.replace(stock, replacement)
    text = "".join(line[4:] if line.startswith("    ") else line for line in text.splitlines(keepends=True))
    # CuTe DSL reads a jit function's source when it traces it: the edited method lives in a real file.
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"sglang_sm107_fc1kb_{method}_{digest}.py")
    if not os.path.exists(path):
        staging = f"{path}.{os.getpid()}"
        with open(staging, "w") as f:
            f.write(text)
        os.replace(staging, path)
    namespace = dict(vars(module))
    namespace.update(
        _FC1KB_K_TILES=_FC1KB_K_TILES,
        _FC1KB_TILES_PER_K=_FC1KB_TILES_PER_K,
        _FC1KB_INTERMEDIATE=_FC1KB_INTERMEDIATE,
        _FC1KB_ALL=_FC1KB_ALL,
        _FC1KB_UNSUPPORTED=_FC1KB_UNSUPPORTED,
        _fc1kb_ready_mask=_fc1kb_ready_mask,
        _fc1kb_ready_mask_acq=_fc1kb_ready_mask_acq,
        red_async_add_release_gpu_s32=red_async_add_release_gpu_s32,
        AddressSpace=AddressSpace,
        Int32=Int32,
        Int64=Int64,
        cute=cute,
        cutlass=cutlass,
    )
    exec(compile(text, path, "exec"), namespace)
    setattr(getattr(module, class_name), method, namespace[method])


def install_sm107_megamoe_fc1_kblock_wait(acquire_v8: bool = False) -> None:
    """Every SM107 inference kernel traced afterwards gates FC2 per K tile on FC1 readiness.

    acquire_v8 reads a slot's counters with one acquire load instead of relaxed loads and a GPU fence.
    The edited classes are shared with nothing else the inference variant builds; a process runs one SM107
    variant and one mode.
    """
    global _fc1_kblock_wait_installed
    mode = "acquire_v8" if acquire_v8 else "fence"
    if _fc1_kblock_wait_installed == mode:
        return
    if _fc1_kblock_wait_installed is not None:
        raise RuntimeError(f"per-K-tile FC1 readiness is installed as {_fc1_kblock_wait_installed}, not {mode}")
    for module_name, class_name, method, edits in _fc1kb_edits(acquire_v8):
        _fc1kb_rebuild(module_name, class_name, method, edits)
    _fc1_kblock_wait_installed = mode
