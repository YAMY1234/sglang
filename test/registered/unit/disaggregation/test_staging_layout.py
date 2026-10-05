"""Derived byte-layout contracts for v2; no pool, engine or device is required."""

import dataclasses
import unittest

from sglang.srt.disaggregation.common.staging_layout import (
    MAX_BYTES,
    StagingEntry,
    WriterLayout,
    align_bytes,
    decode_manifest,
    encode_manifest,
    negotiate_version,
    plan_chunk,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def entry(
    index=0,
    layer=47,
    width=20,
    kind="mla_latent",
    heads=0,
    component="target",
    page_size=4,
):
    return StagingEntry(
        component, layer, kind, index, "float16", page_size, width, width, heads
    )


def layout(tp=1, rank=0, pp=0, entries=None):
    return WriterLayout(
        f"p{pp}t{rank}",
        pp,
        rank,
        tp,
        tuple(entries if entries is not None else [entry()]),
    )


class TestStagingLayout(CustomTestCase):
    def test_manifest_round_trip_and_confirmation(self):
        writers, dst = (layout(),), dataclasses.replace(layout(), session="decode")
        encoded = encode_manifest(writers, dst)
        self.assertEqual(decode_manifest(encoded), (writers, dst))
        with self.assertRaisesRegex(ValueError, "confirmation"):
            decode_manifest(
                encoded.replace(b'"global_layer_id":47', b'"global_layer_id":48')
            )
        for local, remote in ((2, 0), (0, 2), (2, 1), (1, 2), (2, 3)):
            with (
                self.subTest(local=local, remote=remote),
                self.assertRaises(ValueError),
            ):
                negotiate_version(local, remote)
        for pair in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 2)):
            negotiate_version(*pair)

    def test_mixed_writers_cover_each_destination_byte_once(self):
        for src_tp, dst_tp, heads in (
            (8, 8, 8),
            (2, 8, 8),
            (8, 2, 8),
            (8, 2, 4),
            (2, 8, 4),
        ):
            for dst_rank in range(dst_tp):

                def entries(tp):
                    return [
                        entry(),
                        entry(1, 93, max(1, heads // tp) * 4, "mha_k", heads, "draft"),
                        entry(2, 93, max(1, heads // tp) * 6, "mha_v", heads, "draft"),
                    ]

                ranks = (
                    range(
                        dst_rank * src_tp // dst_tp, (dst_rank + 1) * src_tp // dst_tp
                    )
                    if src_tp >= dst_tp
                    else [dst_rank * src_tp // dst_tp]
                )
                writers = tuple(
                    layout(src_tp, rank, entries=entries(src_tp)) for rank in ranks
                )
                dst = dataclasses.replace(
                    layout(dst_tp, dst_rank, entries=entries(dst_tp)), session="decode"
                )
                plan = plan_chunk(writers, dst, 7)
                for target in dst.entries:
                    counts = [0] * target.copy_width_bytes
                    for region in plan.regions:
                        for copy in region.entries:
                            if copy.destination == target:
                                for i in range(
                                    copy.dst_offset, copy.dst_offset + copy.width
                                ):
                                    counts[i] += 1
                    self.assertEqual(counts, [1] * len(counts))
                latent = [
                    copy
                    for region in plan.regions
                    for copy in region.entries
                    if copy.source.kind == "mla_latent"
                ]
                self.assertEqual(len(latent), 1)
                if src_tp > dst_tp and heads >= src_tp:
                    self.assertEqual(len(plan.expected_writers), src_tp // dst_tp)
                self.assertEqual(
                    plan.payload_bytes, 7 * sum(e.copy_width_bytes for e in dst.entries)
                )

    def test_non_contiguous_pp_layers_and_variable_widths(self):
        counts = [6, 5, 6, 7]
        writers, full = [], []
        next_layer = 2
        for pp, count in enumerate(counts):
            entries = []
            for i in range(count):
                entries.append(entry(i, next_layer, width=8 + 2 * i))
                full.append(dataclasses.replace(entries[-1], index=len(full)))
                next_layer += 3
            if pp == 3:
                for kind in ("mha_k", "mha_v"):
                    entries.append(entry(len(entries), 93, 8, kind, 2, "draft"))
                    full.append(dataclasses.replace(entries[-1], index=len(full)))
            writers.append(layout(pp=pp, entries=entries))
        writers.append(layout(pp=4, entries=[]))
        dst = layout(entries=full)
        plan = plan_chunk(tuple(writers), dst, 65)
        self.assertEqual(len(plan.regions), 4)
        self.assertEqual(sum(len(r.entries) for r in plan.regions), 26)
        self.assertEqual(plan.payload_bytes, 65 * sum(e.copy_width_bytes for e in full))
        last = 0
        for region in plan.regions:
            self.assertEqual(region.offset, last)
            self.assertEqual(region.length % 256, 0)
            last = region.offset + region.length
        self.assertEqual(last, plan.total_bytes)

    def test_missing_duplicate_and_incompatible_entries_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            layout(entries=[entry(), entry(1)])
        with self.assertRaisesRegex(ValueError, "Missing"):
            plan_chunk((layout(entries=[]),), layout(), 1)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            plan_chunk((layout(pp=0), layout(pp=1)), layout(), 1)
        with self.assertRaisesRegex(ValueError, "widths"):
            plan_chunk((layout(),), layout(entries=[entry(width=22)]), 1)
        with self.assertRaisesRegex(ValueError, "geometry"):
            plan_chunk(
                (layout(),),
                layout(entries=[dataclasses.replace(entry(), dtype="bfloat16")]),
                1,
            )
        draft = entry(kind="mha_k", width=4, heads=8, component="draft")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            plan_chunk(
                (layout(8, 0, entries=[draft]),),
                layout(
                    2,
                    0,
                    entries=[
                        dataclasses.replace(
                            draft, row_stride_bytes=16, copy_width_bytes=16
                        )
                    ],
                ),
                1,
            )

    def test_64_bit_offsets_and_checked_alignment(self):
        plan = plan_chunk((layout(),), layout(), (1 << 32) + 1)
        self.assertGreater(plan.total_bytes, 1 << 32)
        self.assertEqual(align_bytes(256), 256)
        self.assertEqual(align_bytes(257), 512)
        for value in (-1, True, 1.5, MAX_BYTES, MAX_BYTES + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                align_bytes(value)
        with self.assertRaises(ValueError):
            plan_chunk((layout(),), layout(), MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
