import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.dsv41_request_window import RequestWindow, window_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

WINDOW = 128
LAYERS = 2
SLOTS = 4


def _pool_factory(layout, page_size, device):
    def make(size, layers):
        pages = -(-size // page_size)
        return SimpleNamespace(
            size=pages * page_size,
            kv_layout=layout,
            dtype=torch.uint8,
            kv_buffer=[
                torch.zeros(
                    pages,
                    layout.page_bytes(page_size),
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(layers)
            ],
        )

    return make


def _row_cols(loc, width, base, page_size):
    loc = loc.long()[:, None]
    cols = torch.arange(width, device=loc.device)
    return loc // page_size, base + (loc % page_size) * width + cols


def _parts(layout, page_size):
    return (
        (layout.data_bytes, 0),
        (layout.scale_bytes, page_size * layout.data_bytes),
    )


def _store(buf, loc, rows, layout, page_size):
    # Stands in for the fused store kernel: one packed (data, scale) row per token.
    start = 0
    for width, base in _parts(layout, page_size):
        buf[_row_cols(loc, width, base, page_size)] = rows[:, start : start + width]
        start += width


def _gather(buf, indices, layout, page_size):
    # What attention reads: the packed row behind every valid index, in order.
    flat = indices[indices >= 0]
    return torch.cat(
        [buf[_row_cols(flat, w, b, page_size)] for w, b in _parts(layout, page_size)],
        dim=1,
    )


class _Pair:
    """The workspace path and the direct path, fed the same steps."""

    def __init__(self, layout, page_size, draft_tokens, device):
        factory = _pool_factory(layout, page_size, device)
        kwargs = {
            "num_slots": SLOTS,
            "layers": LAYERS,
            "page_size": page_size,
            "capacity": WINDOW + draft_tokens,
            "workspace_rows": (SLOTS + 1) * WINDOW + 512,
        }
        self.ref = RequestWindow(factory, **kwargs)
        self.new = RequestWindow(factory, **kwargs)
        self.layout, self.page_size, self.device = layout, page_size, device
        self.row_bytes = layout.data_bytes + layout.scale_bytes
        self.gen = torch.Generator(device="cpu").manual_seed(0)

    def _rows(self, n):
        return torch.randint(
            0, 256, (LAYERS, n, self.row_bytes), generator=self.gen, dtype=torch.uint8
        ).to(self.device)

    def _layout(self, window, req, pos, floor, num_groups, direct):
        return window_layout(
            torch.tensor(req, device=self.device),
            torch.tensor(pos, device=self.device),
            capacity=window.capacity,
            floor=None if floor is None else torch.tensor(floor, device=self.device),
            num_groups=num_groups,
            sink_row=window.sink_row if direct else None,
        )

    def _forward(self, window, layout, rows):
        window.activate(layout)
        seen = []
        for layer in range(LAYERS):
            buf = window.buffer(layer)
            _store(buf, layout.write_loc, rows[layer], self.layout, self.page_size)
            seen.append(_gather(buf, layout.indices, self.layout, self.page_size))
            window.commit(layer)
        return seen

    def extend(self, req, pos, floor=None):
        rows = self._rows(len(pos))
        for window in (self.ref, self.new):
            layout = self._layout(window, req, pos, floor, len(set(req)), False)
            self._forward(window, layout, rows)

    def decode(self, req, pos, floor=None, num_groups=None, pad=0):
        """Run one step on both paths; the direct path also gets ``pad`` padding
        rows shaped like graph/ragged padding (last request, position 0)."""
        rows = self._rows(len(pos) + pad)
        ref_layout = self._layout(self.ref, req, pos, floor, num_groups, False)
        ref_seen = self._forward(self.ref, ref_layout, rows[:, : len(pos)])
        padded_floor = None if floor is None else floor + [0] * pad
        new_layout = self._layout(
            self.new,
            req + req[-1:] * pad,
            pos + [0] * pad,
            padded_floor,
            num_groups,
            True,
        )
        new_seen = self._forward(self.new, new_layout, rows)
        real = new_layout.lengths[: len(pos)].sum().item()
        for a, b in zip(ref_seen, new_seen):
            torch.testing.assert_close(a, b[:real], rtol=0, atol=0)
        torch.testing.assert_close(
            ref_layout.lengths, new_layout.lengths[: len(pos)], rtol=0, atol=0
        )
        return new_layout

    def assert_same_state(self, slots=None):
        if slots is None:
            keep = torch.ones(self.ref.rows + self.page_size, dtype=torch.bool)
            keep[self.ref.sink_row] = False
            live = torch.arange(keep.numel())[keep]
        else:
            cap = self.ref.capacity
            live = torch.cat([torch.arange(s * cap, (s + 1) * cap) for s in slots])
        live = live.to(self.device)
        for layer in range(LAYERS):
            for width, base in _parts(self.layout, self.page_size):
                idx = _row_cols(live, width, base, self.page_size)
                torch.testing.assert_close(
                    self.ref.state.kv_buffer[layer][idx],
                    self.new.state.kv_buffer[layer][idx],
                    rtol=0,
                    atol=0,
                )
        torch.testing.assert_close(
            self.ref.tags[:, live], self.new.tags[:, live], rtol=0, atol=0
        )


class TestRequestWindowDirectDecode(CustomTestCase):
    """Decode/verify on the ring must see and leave the same bytes as the workspace."""

    def _run(self, device):
        layouts = (KVLayout.V4, KVLayout.V41, KVLayout.V41_FP4)
        for layout, page_size, draft in product(layouts, (16, 64, 256), (0, 6)):
            with self.subTest(layout=layout, page_size=page_size, draft=draft):
                self._sequence(_Pair(layout, page_size, draft, device), draft)

    def _sequence(self, pair, draft):
        gen = torch.Generator().manual_seed(1)
        q = max(draft, 1)
        # Slot 1 starts short (history below 0 is masked), slot 2 long enough to
        # wrap its ring several times; slot 3 starts after a replay floor.
        prompt = {1: 5, 2: 700, 3: 300}
        floor = {1: 0, 2: 0, 3: 200}
        for slot, n in prompt.items():
            pair.extend([slot] * n, list(range(n)), [floor[slot]] * n)
        pair.assert_same_state()
        nxt = dict(prompt)
        for step in range(40):
            slots = [s for s in prompt if (step + s) % 4]
            req = [s for s in slots for _ in range(q)]
            pos = [nxt[s] + i for s in slots for i in range(q)]
            floors = [floor[s] for s in slots for _ in range(q)]
            pad = int(torch.randint(0, 3, (1,), generator=gen)) if q > 1 else 0
            groups = len(slots) if q > 1 else None
            pair.decode(req, pos, floors, num_groups=groups, pad=pad)
            pair.assert_same_state()
            for s in slots:
                nxt[s] += int(torch.randint(1, q + 1, (1,), generator=gen))

    def test_direct_matches_workspace_cpu(self):
        self._run("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_direct_matches_workspace_cuda(self):
        self._run("cuda")

    def test_direct_layout_skips_the_workspace(self):
        pair = _Pair(KVLayout.V4, 64, 6, "cpu")
        pair.extend([1] * 200, list(range(200)))
        layout = pair.decode([1] * 6, list(range(200, 206)), num_groups=1)
        self.assertTrue(layout.direct)
        self.assertIs(pair.new.buffer(0), pair.new.state.kv_buffer[0])
        cap = pair.new.capacity
        ring = layout.indices[layout.indices >= 0]
        for loc in (layout.write_loc, ring):
            self.assertTrue(bool(((loc >= cap) & (loc < 2 * cap)).all()))

    def test_padding_rows_go_to_the_sink(self):
        pair = _Pair(KVLayout.V4, 64, 6, "cpu")
        pair.extend([2] * 400, list(range(400)))
        layout = pair.decode([2] * 6, list(range(400, 406)), num_groups=1, pad=3)
        self.assertEqual(layout.write_loc[6:].tolist(), [pair.new.sink_row] * 3)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_direct_step_replays_in_cuda_graph(self):
        layout_kind, page_size = KVLayout.V4, 256
        pair = _Pair(layout_kind, page_size, 6, "cuda")
        pair.extend([1] * 300, list(range(300)))
        pair.extend([2] * 50, list(range(50)))
        window = pair.new
        # Capture on idle slots 0 and 3, then replay a live layout for 1 and 2.
        static = pair._layout(
            window, [0] * 6 + [3] * 6, list(range(6)) * 2, None, 2, True
        )
        rows = torch.zeros(LAYERS, 12, pair.row_bytes, dtype=torch.uint8, device="cuda")

        def step():
            window.activate(static)
            for layer in range(LAYERS):
                buf = window.buffer(layer)
                _store(buf, static.write_loc, rows[layer], layout_kind, page_size)
                window.commit(layer)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()

        req, pos = [1] * 6 + [2] * 6, list(range(300, 306)) + list(range(50, 56))
        static.copy_(pair._layout(window, req, pos, None, 2, True))
        live_rows = pair._rows(12)
        rows.copy_(live_rows)
        graph.replay()
        torch.cuda.synchronize()
        pair._forward(
            pair.ref, pair._layout(pair.ref, req, pos, None, 2, False), live_rows
        )
        pair.assert_same_state(slots=(1, 2))


if __name__ == "__main__":
    unittest.main()
