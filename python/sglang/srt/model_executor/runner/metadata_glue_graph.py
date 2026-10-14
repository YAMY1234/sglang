"""Glue-graph capture of the per-replay attention-metadata prep.

``decode_cuda_graph_runner.load_batch`` runs
``attn_backend.init_forward_metadata_out_graph(fb_view)`` eagerly on every
replay. At bs=1 spec decode this is an "op soup": dozens of tiny tensor ops
whose HOST dispatch cost dominates the inter-phase seam, while every device
input/output lives at a stable address — the replay fb view hands backends the
runner's static buffers, and pool tensors are persistent. Capturing the op
sequence once per replay key collapses the per-step host cost to a single
graph launch.

Correctness contract:

- The caller only routes here when the replay is padding-free
  (raw_bs == padded bs) and TBO / pdmux / LoRA are off, so every
  Python-visible branch inside the backends is constant per key.
- Python side effects (each backend's ``forward_metadata`` object) are
  snapshotted at capture time and re-installed on every replay; the graph
  replays only the device ops that refresh the tensors those objects point to.
- ``NUM_WARMUP`` eager runs precede capture so triton JIT compile / autotune
  happen outside capture.
- Any capture failure (e.g. a backend syncing or reading host values inside
  its prep) permanently disables the glue graph and falls back to eager.
- Backends whose prep computes values on the HOST each replay (e.g. the
  DFlash-family host-fed fast verify plans) must never be glued: capture
  records only device ops, so the host-written plan inputs would replay
  frozen at their capture-time values. Note the failure is SILENT — capture
  succeeds, outputs stay correct, only accept length collapses. Callers must
  gate such configurations off before routing here
  (``decode_cuda_graph_runner`` force-disables the glue for DFlash-family
  spec).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class MetadataGlueGraph:
    NUM_WARMUP = 2

    def __init__(
        self,
        device,
        capture_error_mode: str = "global",
        check_replays: bool = False,
    ):
        self.device = device
        self.disabled = False
        self._states: Dict[Any, dict] = {}
        self._capture_stream = None
        self._capture_error_mode = capture_error_mode
        self._check_replays = check_replays
        self._num_checks = 0

    def reset(self):
        """Drop captured graphs (call when the runner recaptures its graphs —
        static buffers and backend state may have been rebuilt)."""
        self._states.clear()

    @staticmethod
    def _leaves(attn_backend) -> List[Any]:
        backends = [attn_backend]
        if attn_backend.attn_backend_list is not None:
            backends.extend(attn_backend.attn_backend_list)
        return backends

    def run(self, attn_backend, fb_view, key) -> None:
        """Run ``init_forward_metadata_out_graph`` for this replay, through the
        captured glue graph once it is ready."""
        st = self._states.get(key)
        if st is None:
            st = {"warmups": 0, "graph": None, "meta": None}
            self._states[key] = st

        if st["graph"] is not None:
            for backend, metadata in st["meta"]:
                backend.forward_metadata = metadata
            st["graph"].replay()
            if self._check_replays:
                self._check(attn_backend, fb_view, key, st["meta"])
            return

        if st["warmups"] < self.NUM_WARMUP:
            st["warmups"] += 1
            attn_backend.init_forward_metadata_out_graph(fb_view)
            return

        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(
                graph,
                stream=self._capture_stream,
                capture_error_mode=self._capture_error_mode,
            ):
                attn_backend.init_forward_metadata_out_graph(fb_view)
        except Exception:
            logger.warning(
                "Metadata glue-graph capture failed for key %s; falling back "
                "to eager metadata prep permanently.",
                key,
                exc_info=True,
            )
            self.disabled = True
            # Ops under a failed capture were recorded, not executed — run
            # this step's prep for real.
            attn_backend.init_forward_metadata_out_graph(fb_view)
            return

        st["meta"] = [(b, b.forward_metadata) for b in self._leaves(attn_backend)]
        st["graph"] = graph
        # Capture records without executing; replay once to do this step's prep.
        graph.replay()

    def _check(self, attn_backend, fb_view, key, meta) -> None:
        """Compare this replay's metadata tensors with an eager build of the same
        step; the eager build rewrites the buffers, so the forward reads its values."""
        replayed = {
            path: tensor.clone()
            for i, (_, metadata) in enumerate(meta)
            for path, tensor in _metadata_tensors(metadata, f"[{i}]").items()
        }
        attn_backend.init_forward_metadata_out_graph(fb_view)
        reference = {
            path: tensor
            for i, backend in enumerate(self._leaves(attn_backend))
            for path, tensor in _metadata_tensors(
                backend.forward_metadata, f"[{i}]"
            ).items()
        }
        mismatches = [
            path
            for path in replayed.keys() | reference.keys()
            if path not in replayed
            or path not in reference
            or replayed[path].shape != reference[path].shape
            or not torch.equal(replayed[path], reference[path])
        ]
        self._num_checks += 1
        if mismatches:
            logger.warning(
                "Metadata glue-graph replay for %s differs from the eager build in "
                "%d tensors: %s",
                key,
                len(mismatches),
                sorted(mismatches)[:8],
            )
        elif self._num_checks % 100 == 1:
            logger.info(
                "Metadata glue-graph replay matches the eager build (%d replays "
                "checked).",
                self._num_checks,
            )


def _metadata_tensors(metadata: Any, root: str) -> Dict[str, torch.Tensor]:
    """Every tensor reachable from ``metadata`` through dicts, sequences and
    dataclass fields, by path; shared objects are visited once."""
    tensors: Dict[str, torch.Tensor] = {}
    visited: set = set()

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, torch.Tensor):
            tensors[path] = obj
            return
        if id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, dict):
            for name, value in obj.items():
                visit(value, f"{path}[{name!r}]")
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                visit(value, f"{path}[{i}]")
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for field in dataclasses.fields(obj):
                visit(getattr(obj, field.name), f"{path}.{field.name}")

    visit(metadata, root)
    return tensors
