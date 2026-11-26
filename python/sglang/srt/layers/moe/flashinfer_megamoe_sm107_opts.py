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
"""Opt-in single tail barrier for FlashInfer's compute capability 10.7 inference
MegaMOE kernel (FlashInfer 0.7.4 / 0.7.5).

FlashInfer compiles the kernel with CuTe DSL when a workspace first launches, so
the option replaces a method of the kernel's token-communication component before
then; CuTe DSL keys its compile cache on the traced IR, so the changed kernel never
reuses a stock binary. Data movement and arithmetic are unchanged: outputs are
bit-identical to the stock kernel.

The stock tail joins the grid, runs a drain NVLink barrier, lets CTA 0 zero the
per-launch counters, then runs a second barrier to publish the zeros. Peers
write the shared counters (expert sizes and the two ready flags) only from
their router kernel, before this rank's token-in passes its readiness waits, so
CTA 0 can zero them as soon as every local CTA has joined. One barrier then
both drains the launch and publishes the zeros: a peer signals only after its
own zeros, and starts its next launch only after every signal. When one CTA
cannot clear the counters (or under FlashInfer's ncu replay mode) the stock
tail runs. Every EP rank must make the same choice.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import bootstrap_paths

# Puts FlashInfer's vendored kernel tree (package ``sources``) on sys.path.
bootstrap_paths()

from sources.communication.nvlink_domain.token_comm import (  # noqa: E402
    TokenCommNonDeterministic,
)
from sources.helpers.iket_compat import iket  # noqa: E402

_stock_reset_tail = TokenCommNonDeterministic.reset_tail
_installed = False


@cute.jit
def _reset_tail_single_barrier(self) -> None:
    # self: the kernel's TokenCommNonDeterministic (stock: its reset_tail).
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
    """Patch the inference kernel's tail once per process."""
    global _installed
    if not _installed:
        TokenCommNonDeterministic.reset_tail = _reset_tail_single_barrier
        _installed = True
