"""The request window under the breakable prefill CUDA graph.

The graph captures the window gather and the SWA K store (they run inside its
segments) and runs the commit in the eager attention break. A replay refreshes
the captured layout in place, with bucket-padding rows that must never reach a
request's window.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.dsv41_request_window import (
    RequestWindow,
    copy_packed_tokens,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

WINDOW = 8
CAPACITY = 16
NUM_SLOTS = 4
LAYERS = 2
PAGE = 16
GROUPS = NUM_SLOTS + 1  # every request slot plus the bucket-padding group
BUCKET = 8


def _pool_factory(layout):
    def make(size, layers):
        pages = -(-size // PAGE)
        g = torch.Generator(device="cuda").manual_seed(size * 31 + layers)
        bufs = [
            torch.randint(
                0,
                256,
                (pages, layout.page_bytes(PAGE)),
                dtype=torch.uint8,
                device="cuda",
                generator=g,
            )
            for _ in range(layers)
        ]
        return SimpleNamespace(kv_buffer=bufs, kv_layout=layout, size=pages * PAGE)

    return make


def _layout(window, req, pos, floor, live):
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _request_window_layout,
    )

    def cuda(values):
        return torch.tensor(values, device="cuda")

    return _request_window_layout(
        cuda(req),
        cuda(pos),
        capacity=window.capacity,
        floor=cuda(floor),
        num_groups=GROUPS,
        live_rows=live,
        window=WINDOW,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBreakablePrefillGraphWindow(CustomTestCase):
    def test_captured_gather_and_store_follow_a_padded_fold(self):
        """Captured with one dummy request, replayed with a folded hit, a cold
        request and one padding row: state, workspace and tags match gather, store
        and commit by indexing, and the padding row never reaches slot 3."""
        layout = KVLayout.V4
        window = RequestWindow(
            _pool_factory(layout),
            num_slots=NUM_SLOTS,
            layers=LAYERS,
            page_size=PAGE,
            capacity=CAPACITY,
            workspace_rows=GROUPS * WINDOW + BUCKET,
        )
        k_rows = _pool_factory(layout)(BUCKET, 1).kv_buffer[0]
        rows = torch.arange(BUCKET, device="cuda")
        captured = _layout(
            window, [0] * BUCKET, list(range(BUCKET)), [0] * BUCKET, BUCKET
        )
        window.activate(captured)
        window.initialize_dummy_history()

        def store():
            # The K store writes through the gathered workspace at write_loc.
            workspace = window.buffer(0)
            copy_packed_tokens(
                k_rows,
                workspace,
                rows,
                captured.write_loc,
                page_size=PAGE,
                layout=layout,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            store()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        window.prepared = None
        with torch.cuda.graph(graph):
            store()

        # Slot 2 replays positions 10-12 (floor 10) before its new 13-14; slot 3
        # is cold at 0-1; the last row pads the bucket with slot 3's id.
        req = [2] * 5 + [3] * 3
        pos = [10, 11, 12, 13, 14, 0, 1, 0]
        floor = [10] * 5 + [0] * 3
        captured.copy_(_layout(window, req, pos, floor, live=7))
        self.assertFalse(bool(captured.commit_mask[7]))
        window.tags.fill_(-1)
        state = window.state.kv_buffer[0]
        state.copy_(torch.randint_like(state, 0, 256))
        window.workspace.kv_buffer[0].fill_(0xA5)
        k_rows.copy_(torch.randint_like(k_rows, 0, 256))

        cap, lw = window.capacity, captured
        exp_state = state.clone()
        exp_ws = window.workspace.kv_buffer[0].clone()
        exp_tags = window.tags[0].clone()
        src = torch.where(
            lw.history_valid,
            lw.history_req * cap + lw.history_pos % cap,
            window.zero_row,
        )
        copy_packed_tokens(
            exp_state, exp_ws, src, lw.history_loc, page_size=PAGE, layout=layout
        )
        copy_packed_tokens(
            k_rows, exp_ws, rows, lw.write_loc, page_size=PAGE, layout=layout
        )
        dst = torch.where(lw.commit_mask, lw.req * cap + lw.pos % cap, window.sink_row)
        copy_packed_tokens(
            exp_ws, exp_state, lw.write_loc, dst, page_size=PAGE, layout=layout
        )
        exp_tags[dst] = lw.pos

        graph.replay()
        window.mark_gathered(0)  # the eager break: the segment already gathered
        window.commit(0)
        torch.cuda.synchronize()

        keep = torch.ones_like(exp_tags, dtype=torch.bool)
        keep[window.sink_row] = False
        self.assertTrue(torch.equal(window.tags[0][keep], exp_tags[keep]))
        self.assertTrue(torch.equal(window.workspace.kv_buffer[0], exp_ws))
        sink_page = window.sink_row // PAGE
        rows_kept = [p for p in range(state.shape[0]) if p != sink_page]
        self.assertTrue(torch.equal(state[rows_kept], exp_state[rows_kept]))
        # Slot 3's position 0 holds its live row (5), not the padding row (7).
        live = torch.zeros_like(state)
        copy_packed_tokens(
            k_rows,
            live,
            rows[5:6],
            torch.tensor([3 * cap], device="cuda"),
            page_size=PAGE,
            layout=layout,
        )
        slot = torch.tensor([3 * cap], device="cuda")
        got = torch.zeros_like(state)
        copy_packed_tokens(state, got, slot, slot, page_size=PAGE, layout=layout)
        self.assertTrue(torch.equal(got, live))


if __name__ == "__main__":
    unittest.main()
