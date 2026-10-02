"""CPU routing replay of the actual model methods; CUDA/NLL are separate gates."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/sglang/srt/models/flash_next_duet/model.py"


def load_methods(scope):
    tree = ast.parse(SOURCE.read_text())
    helper = next(
        n
        for n in tree.body
        if getattr(n, "name", "") == "_alloff_prefill_graph_enabled"
    )
    model = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForConditionalGeneration"
    )
    names = {"forward", "_alloff_prefill_graph_runner", "_is_twinstar_prefill"}
    methods = [copy.deepcopy(n) for n in model.body if getattr(n, "name", "") in names]
    assert len(methods) == len(names)
    for method in methods:
        method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            helper,
            ast.ClassDef(
                name="Replay", bases=[], keywords=[], body=methods, decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["Replay"], scope["_alloff_prefill_graph_enabled"]


class Mode:
    def __init__(self, kind="extend"):
        self.kind = kind

    def is_extend(self):
        return self.kind in ("extend", "mixed", "verify", "draft")

    def is_mixed(self):
        return self.kind == "mixed"

    def is_target_verify(self):
        return self.kind == "verify"

    def is_draft_extend_v2(self):
        return self.kind == "draft"


class TestAlloffPrefillGraph(unittest.TestCase):
    def setUp(self):
        self.flag = False
        self.capturing = False
        self.calls = []
        self.scope = dict(
            envs=SimpleNamespace(
                SGLANG_FLASHNEXT_ALLOFF_PREFILL_GRAPH=SimpleNamespace(
                    get=lambda: self.flag
                )
            ),
            get_is_capture_mode=lambda: self.capturing,
            logger=SimpleNamespace(info=lambda *args: self.calls.append(("log", args))),
            LogitsProcessorOutput=SimpleNamespace,
        )
        self.cls, self.enabled = load_methods(self.scope)
        self.args = SimpleNamespace(
            disaggregation_mode="null", is_embedding=False, pp_size=1
        )
        self.fs = dict(prefill_layer_trim=False, gdn_rank=0)
        self.model = self.cls()
        self.model.fullstack = self.fs
        self.model.twinstar = {"fullstack": self.fs}
        self.model.alloff_prefill_graph = True
        self.model.n_alloff_graph = 0
        self.model.dump_dir = None
        self.model.state_audit_dir = None
        self.streams = object()
        self.mixed = object()
        self.output = SimpleNamespace(next_token_logits=object())
        self.fallback = object()
        self.ids = SimpleNamespace(shape=(8192,))
        self.positions = object()
        self.fb = SimpleNamespace(forward_mode=Mode(), spec_info=None)
        self.allowed = True
        self.tuple_output = True

        def can_run(fb):
            self.calls.append(("can_run", fb))
            return self.allowed

        def run(fb):
            self.calls.append(("graph", fb))
            return (self.streams, object()) if self.tuple_output else self.streams

        def mix(streams):
            self.calls.append(("mix", streams))
            return self.mixed, None

        def logits(ids, hidden, head, fb):
            self.calls.append(("logits", ids, hidden, head, fb))
            return self.output

        def stock(*args, **kwargs):
            self.calls.append(("stock", args, kwargs))
            return self.fallback

        self.model._prefill_runners = {
            "trunk": SimpleNamespace(can_run=can_run, run=run)
        }
        self.model.model = SimpleNamespace(
            model=SimpleNamespace(hyper_connection_mixer=SimpleNamespace(mix=mix)),
            logits_processor=logits,
            lm_head=object(),
            forward=stock,
        )

    def call(self, **kwargs):
        return self.model.forward(self.ids, self.positions, self.fb, **kwargs)

    def test_default_and_role_arm_gate(self):
        self.assertFalse(self.enabled(self.fs, self.args))
        self.flag = True
        self.assertTrue(self.enabled(self.fs, self.args))
        for fs in (
            None,
            {},
            dict(prefill_layer_trim=True, gdn_rank=0),
            dict(prefill_layer_trim=False, gdn_rank=8),
        ):
            with self.subTest(fs=fs):
                self.assertFalse(self.enabled(fs, self.args))
        for key, value in (
            ("pp_size", 2),
            ("is_embedding", True),
            ("disaggregation_mode", "prefill"),
            ("disaggregation_mode", "decode"),
        ):
            args = SimpleNamespace(**vars(self.args))
            setattr(args, key, value)
            with self.subTest(key=key, value=value):
                self.assertFalse(self.enabled(self.fs, args))

    def test_alloff_accuracy_guard_is_preserved(self):
        self.assertFalse(self.model._is_twinstar_prefill(self.fb))

    def test_whole_batch_tensor_and_tuple_outputs(self):
        for tuple_output in (False, True):
            self.tuple_output = tuple_output
            self.calls.clear()
            self.model.model.model.last_hc_hidden_states = object()
            self.assertIs(self.call(), self.output)
            self.assertIs(self.model.model.model.last_hc_hidden_states, self.streams)
            self.assertIs(self.output.hidden_states, self.streams)
            self.assertEqual(
                [r[0] for r in self.calls[:4]], ["can_run", "graph", "mix", "logits"]
            )
            self.assertIs(self.calls[1][1], self.fb)
            self.assertIs(self.calls[2][1], self.streams)
            self.assertIs(self.calls[3][1], self.ids)
            self.assertIs(self.calls[3][2], self.mixed)
            self.assertIs(self.calls[3][4], self.fb)
            self.assertFalse(any(r[0] == "stock" for r in self.calls))
        self.assertEqual(self.model.n_alloff_graph, 2)

    def test_runner_rejection_and_missing_runner_fall_back(self):
        self.allowed = False
        self.assertIs(self.call(), self.fallback)
        self.assertEqual([r[0] for r in self.calls], ["can_run", "stock"])
        self.calls.clear()
        self.model._prefill_runners = {}
        self.assertIs(self.call(), self.fallback)
        self.assertEqual([r[0] for r in self.calls], ["stock"])
        self.assertEqual(self.model.n_alloff_graph, 0)

    def test_ineligible_forward_never_calls_runner(self):
        cases = [dict(kind=k) for k in ("decode", "mixed", "verify", "draft")]
        cases += [
            dict(capture=True),
            dict(spec=True),
            dict(disabled=True),
            dict(embedding=True),
            dict(pp=True),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.calls.clear()
                self.capturing = case.get("capture", False)
                self.model.alloff_prefill_graph = not case.get("disabled", False)
                self.fb.forward_mode = Mode(case.get("kind", "extend"))
                self.fb.spec_info = object() if case.get("spec") else None
                self.assertIs(
                    self.call(
                        get_embedding=case.get("embedding", False),
                        pp_proxy_tensors=object() if case.get("pp") else None,
                    ),
                    self.fallback,
                )
                self.assertEqual([r[0] for r in self.calls], ["stock"])
        self.assertEqual(self.model.n_alloff_graph, 0)


if __name__ == "__main__":
    unittest.main()
