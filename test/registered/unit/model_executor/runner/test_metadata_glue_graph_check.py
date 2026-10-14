"""MetadataGlueGraph under SGLANG_OPT_DSV41_ATTN_METADATA_GRAPH: per-key warm-up,
capture and replay, the capture mode, and the replay check against an eager build."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.runner import metadata_glue_graph as glue_mod
from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.fixture
def fake_cudagraphs(monkeypatch):
    """Records captures and replays in place of CUDA graphs: a captured prep runs
    once, eagerly, and its replays run nothing."""
    graphs: list = []

    class FakeGraph:
        def __init__(self):
            self.replays = 0
            self.mode = None
            graphs.append(self)

        def replay(self):
            self.replays += 1

    @contextmanager
    def capture(graph, pool=None, stream=None, capture_error_mode="global"):
        graph.mode = capture_error_mode
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: None)
    return graphs


@dataclass
class Metadata:
    lens: torch.Tensor
    nested: dict


class Backend:
    """A leaf backend whose prep writes ``value`` into fresh metadata."""

    attn_backend_list = None

    def __init__(self):
        self.calls = 0
        self.value = 1
        self.forward_metadata = None

    def init_forward_metadata_out_graph(self, fb_view):
        self.calls += 1
        self.forward_metadata = Metadata(
            torch.tensor([self.value, 2]), {"idx": [torch.tensor([3])]}
        )


def test_warm_up_capture_then_replay_per_key(fake_cudagraphs):
    glue = MetadataGlueGraph("cpu", capture_error_mode="thread_local")
    backend = Backend()
    for _ in range(MetadataGlueGraph.NUM_WARMUP):
        glue.run(backend, None, "a")
    assert backend.calls == 2 and not fake_cudagraphs
    glue.run(backend, None, "a")
    captured = backend.forward_metadata
    assert backend.calls == 3 and len(fake_cudagraphs) == 1
    assert fake_cudagraphs[0].mode == "thread_local"
    assert fake_cudagraphs[0].replays == 1  # the capture step replays too
    backend.forward_metadata = None
    for _ in range(3):
        glue.run(backend, None, "a")
        assert backend.forward_metadata is captured
    assert backend.calls == 3 and fake_cudagraphs[0].replays == 4
    # Another key starts over; the first keeps its graph.
    glue.run(backend, None, "b")
    assert backend.calls == 4 and len(fake_cudagraphs) == 1


def test_replay_check_reports_differing_tensors(fake_cudagraphs, monkeypatch):
    warnings: list = []
    monkeypatch.setattr(glue_mod.logger, "warning", lambda *a: warnings.append(a))
    monkeypatch.setattr(glue_mod.logger, "info", lambda *a: None)
    glue = MetadataGlueGraph("cpu", check_replays=True)
    backend = Backend()
    for _ in range(MetadataGlueGraph.NUM_WARMUP + 2):
        glue.run(backend, None, "a")
    assert not warnings  # the eager reference build wrote the same values

    backend.value = 7  # a host-fed value the captured prep froze
    glue.run(backend, None, "a")
    assert len(warnings) == 1
    assert warnings[0][-1] == ["[0].lens"]


def test_default_mode_is_global_without_check(fake_cudagraphs):
    glue = MetadataGlueGraph("cpu")
    backend = Backend()
    for _ in range(MetadataGlueGraph.NUM_WARMUP + 2):
        glue.run(backend, None, "a")
    assert fake_cudagraphs[0].mode == "global"
    assert backend.calls == MetadataGlueGraph.NUM_WARMUP + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_replay_rebuilds_from_persistent_inputs():
    """A real graph: replays recompute the metadata from the current contents of
    the persistent inputs, and the replay check passes."""
    device = torch.device("cuda")
    seq_lens = torch.zeros(4, dtype=torch.int32, device=device)
    out = torch.zeros(4, dtype=torch.int32, device=device)

    class CudaBackend:
        attn_backend_list = None
        forward_metadata = None

        def init_forward_metadata_out_graph(self, fb_view):
            torch.add(seq_lens, 7, out=out)
            self.forward_metadata = Metadata(out, {"tmp": seq_lens * 2})

    glue = MetadataGlueGraph(
        device, capture_error_mode="thread_local", check_replays=True
    )
    backend = CudaBackend()
    for step in range(6):
        seq_lens.fill_(step)
        glue.run(backend, SimpleNamespace(), "a")
        torch.cuda.synchronize()
        meta = backend.forward_metadata
        assert meta.lens.tolist() == [step + 7] * 4
        assert meta.nested["tmp"].tolist() == [step * 2] * 4
    assert not glue.disabled and glue._num_checks == 3
