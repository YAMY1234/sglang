"""AGG transient LinearCode must keep coordinates on device, including graphs.

CPU arithmetic and real serving-method routing are covered here. CUDA graph
execution and throughput are qualified by the paired C48 service gate.
"""

import ast
import copy
import logging
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
MODELS = ROOT / "python/sglang/srt/models"
sys.path.insert(0, str(MODELS))
from lightning_duet._common import load

common = load("latent_codec")
from flash_next_duet import latent, prefill_graph

SPEC = dict(latent_rank=16, latent_spikes=4, latent_id_side=True)


def codec(width=64, **overrides):
    torch.manual_seed(39)
    spec = dict(SPEC, **overrides)
    value = latent.FlashNextLatentCodec(device="cpu", spec=spec, width=width)
    for name, shape in (
        ("E", (spec["latent_rank"], width)),
        ("D", (width, spec["latent_rank"])),
        ("mean", (width,)),
    ):
        value.load(name, torch.randn(shape) / 8)
    return value


def real_function(path, name, namespace, cls=None):
    tree = ast.parse(path.read_text())
    body = (
        next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
        if cls
        else tree.body
    )
    node = copy.deepcopy(
        next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


class TransientCodecTests(unittest.TestCase):
    def test_matches_stored_roundtrip_without_mutating_input(self):
        for dtype in (torch.bfloat16, torch.float32):
            for id_side in (True, False):
                for spikes in (0, 4, 16):
                    with self.subTest(dtype=dtype, id_side=id_side, spikes=spikes):
                        c = codec(latent_id_side=id_side, latent_spikes=spikes)
                        h = torch.randn(17, 64).to(dtype)
                        base = torch.randn_like(h)
                        saved = h.clone()
                        positions = torch.arange(17) + 8192
                        record, expected = c.encode_and_decode(h, positions, base)
                        actual = c.reconstruct(h, positions, base)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(h, saved, rtol=0, atol=0)
                        torch.testing.assert_close(
                            c.decode(record, base), expected, rtol=0, atol=0
                        )

    def test_32k_transient_has_no_host_serialization_and_codes_first_token(self):
        c = codec()
        h = torch.randn(32768, 64).bfloat16()
        base = torch.randn_like(h)
        # The reference is intentionally small: a 32K Python gap8 loop is the
        # regression, not a useful CPU timing assertion.
        _, expected = c.encode_and_decode(h[:17], torch.arange(17), base[:17])
        with (
            patch.object(
                torch.Tensor,
                "tolist",
                side_effect=AssertionError("device-to-host list"),
            ),
            patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("device-to-host tensor")
            ),
            patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("device-to-host scalar"),
            ),
            patch.object(
                common, "pack_gap8", side_effect=AssertionError("unneeded storage pack")
            ),
            patch.object(
                common,
                "unpack_gap8",
                side_effect=AssertionError("unneeded storage unpack"),
            ),
        ):
            actual = c.reconstruct(h, torch.arange(32768), base)
            small = c.reconstruct(h[:17], torch.arange(17), base[:17])
        self.assertEqual(actual.shape, h.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(small, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(small[0], h[0]))

    def test_rejects_wrong_logical_rows_or_embedding_width(self):
        c = codec()
        h = torch.zeros(3, 64, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            c.reconstruct(h, torch.arange(4), h)
        with self.assertRaises(ValueError):
            c.reconstruct(h, torch.arange(3), h[:, :16])

    def test_real_31_of_48_bf16_prefill_routes_coded_rows_to_graph_and_eager_emitters(
        self,
    ):
        """Run the actual AGG serving method with CPU trunk/runner stand-ins.

        The graph bodies are real; capture/transport/attention are substitutes.
        This catches bypassing codec, using storage encode on the hot path,
        losing the 31/48 split, and passing bucket padding as live rows.
        """
        serving = ModuleType("flash_next_duet.serving")
        serving.embedding_streams = real_function(
            MODELS / "flash_next_duet/serving.py", "embedding_streams", {}
        )
        backend = SimpleNamespace(init_forward_metadata=lambda fb: None)
        namespace = dict(
            __name__="flash_next_duet.model",
            __package__="flash_next_duet",
            torch=torch,
            ForwardBatch=object,
            os=os,
            get_attn_backend=lambda: backend,
            get_global_expert_distribution_recorder=lambda: None,
            get_attn_tp_context=lambda: SimpleNamespace(
                maybe_input_scattered=lambda fb: nullcontext()
            ),
            optional=lambda name: None,
            LogitsProcessorOutput=SimpleNamespace,
            logger=logging.getLogger(__name__),
        )
        forward = real_function(
            MODELS / "flash_next_duet/model.py",
            "_twinstar_prefill",
            namespace,
            cls="Qwen4ExpForConditionalGeneration",
        )
        for rows in (17, 65):
            c = codec()
            h = torch.randn(rows, 64).bfloat16()
            embedding = torch.randn(rows, 16).bfloat16()
            positions = torch.arange(rows)
            _, expected = c.encode_and_decode(h, positions, embedding.repeat(1, 4))
            for graph in (False, True):
                with self.subTest(rows=rows, graph=graph):
                    visited, emitted = [], []
                    fb = SimpleNamespace(
                        extend_seq_lens_cpu=[rows],
                        extend_prefix_lens_cpu=[0],
                        input_ids=torch.zeros(rows, dtype=torch.long),
                        positions=positions,
                        batch_size=1,
                        spec_info=None,
                        return_logprob=False,
                        capture_hidden_mode=SimpleNamespace(is_full=lambda: False),
                    )
                    owner = SimpleNamespace(
                        fullstack=dict(latent="on", prefill_layer_trim=True),
                        fullstack_code=True,
                        fullstack_v3_latent=False,
                        fullstack_final=True,
                        profile=False,
                        boundary_mode="none",
                        state_audit_dir=None,
                        n_layers=48,
                        p_layer_ids=list(range(31)),
                        emitter_ids=list(range(31, 48)),
                        config=SimpleNamespace(
                            hc_count=4, hidden_size=16, vocab_size=2
                        ),
                        model=SimpleNamespace(model=object()),
                        bridges=[],
                        latent_codec=c,
                        n_graph_trunk=0,
                        n_graph_emitters=0,
                        n_twinstar=0,
                        n_prefix=0,
                        n_fallback=0,
                        n_graph_fallback=0,
                        _boundary_lens=lambda fb: [0],
                        _sub_batch=lambda fb, ids, pos, part: (fb, None),
                        _emit_ids=lambda: list(range(31, 48)),
                    )

                    def trunk(batch):
                        visited.extend(owner.p_layer_ids)
                        return h, embedding

                    owner._p_trunk = trunk
                    owner.emitters = {
                        str(layer): SimpleNamespace(
                            emit=lambda streams, batch, layer=layer: emitted.append(
                                (layer, streams.clone())
                            )
                        )
                        for layer in owner.emitter_ids
                    }
                    if graph:
                        trunk_body = prefill_graph._Body(owner, True, [])
                        emitter_body = prefill_graph._Body(
                            owner, False, owner.emitter_ids
                        )
                        padded = ((rows + 7) // 8) * 8
                        emitter_body.allocate(padded, "cpu")

                        def emit_run(batch, streams):
                            emitter_body.streams_in[:rows].copy_(streams)
                            emitter_body.streams_in[rows:].zero_()
                            emitter_body.forward(torch.empty(padded), positions, batch)

                        owner._prefill_runners = dict(
                            trunk=SimpleNamespace(
                                can_run=lambda fb: True,
                                run=lambda fb: trunk_body.forward(
                                    fb.input_ids, fb.positions, fb
                                ),
                            ),
                            emitters=SimpleNamespace(
                                can_run=lambda fb: True, run=emit_run
                            ),
                        )
                    with (
                        patch.dict(sys.modules, {"flash_next_duet.serving": serving}),
                        patch.object(
                            c,
                            "encode_and_decode",
                            side_effect=AssertionError("transient serialized"),
                        ),
                        patch.object(
                            torch.Tensor,
                            "tolist",
                            side_effect=AssertionError("host roundtrip"),
                        ),
                    ):
                        forward(owner, fb.input_ids, positions, fb)
                    self.assertEqual(visited, list(range(31)))
                    self.assertEqual(
                        [layer for layer, _ in emitted], list(range(31, 48))
                    )
                    self.assertEqual(owner.n_graph_trunk, int(graph))
                    self.assertEqual(owner.n_graph_emitters, int(graph))
                    for _, value in emitted:
                        self.assertEqual(value.dtype, torch.bfloat16)
                        torch.testing.assert_close(
                            value[:rows], expected, rtol=0, atol=0
                        )
                        if graph:
                            self.assertEqual(value.shape, (padded, 64))
                            self.assertEqual(
                                torch.count_nonzero(value[rows:]).item(), 0
                            )


if __name__ == "__main__":
    unittest.main()
