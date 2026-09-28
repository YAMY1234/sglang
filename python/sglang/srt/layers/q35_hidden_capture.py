"""Opt-in eager diagnostic: persist final-normalized LM-head inputs, no edits to outputs."""
import functools
import json
import os
from pathlib import Path

_ENABLED = bool(os.environ.get("Q35_HIDDEN_CAPTURE_DIR"))
_state = {"qid": None, "events": [], "seen": {}, "call": 0}


def _flush():
    if _state["qid"] is None or not _state["events"]:
        return
    import torch
    root = Path(os.environ["Q35_HIDDEN_CAPTURE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"question-{_state['qid']:03d}.pt"
    partial = destination.with_suffix(".partial")
    torch.save({"question_id": _state["qid"], "events": _state["events"]}, partial)
    partial.replace(destination)
    print(f"Q35_CAPTURE saved question={_state['qid']} events={len(_state['events'])}", flush=True)


def capture_lm_head_inputs(func):
    if not _ENABLED:
        return func

    @functools.wraps(func)
    def wrapped(self, input_ids, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        import torch
        context = json.loads(Path(os.environ["Q35_HIDDEN_CAPTURE_CONTEXT"]).read_text())
        qid = context["question_id"]
        if qid != _state["qid"]:
            _flush()
            _state.update(qid=qid, events=[], seen={})
        record = None
        if qid >= 0 and hasattr(logits_metadata, "positions") and hidden_states.numel():
            assert not torch.cuda.is_current_stream_capturing(), "Capture requires eager execution"
            role = getattr(self, "q35_capture_role", "target")
            mode = logits_metadata.forward_mode.name
            positions = logits_metadata.positions.detach().flatten().cpu()
            ids = input_ids.detach().flatten().cpu()
            assert len(positions) == len(hidden_states) == len(ids)
            prompt = not ("VERIFY" in mode or "DECODE" in mode or "DRAFT_EXTEND" in mode)
            kind = "prompt" if prompt else "generation"
            seen = _state["seen"].setdefault((role, kind), set())
            limit = 64 if prompt else 128
            selected = []
            # Distinct positions per role/question/stratum; repeated rejected prefixes are not double-counted.
            for i, pos in enumerate(positions.tolist()):
                if pos not in seen and len(seen) < limit:
                    selected.append(i)
                    seen.add(pos)
            if selected:
                _state["call"] += 1
                record = {"call": _state["call"], "role": role, "mode": mode, "kind": kind,
                          "indices": selected, "positions": positions[selected], "input_ids": ids[selected],
                          "hidden": hidden_states[selected].detach().cpu(),
                          "all_positions": positions, "all_input_ids": ids}
                spec = getattr(logits_metadata, "spec_info", None)
                for name in ("draft_token", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling"):
                    tensor = getattr(spec, name, None)
                    if isinstance(tensor, torch.Tensor):
                        record[name] = tensor.detach().cpu().clone()
                _state["events"].append(record)
        output = func(self, input_ids, hidden_states, lm_head, logits_metadata, *args, **kwargs)
        if record is not None and output.next_token_logits is not None:
            record["actual_bf16_argmax"] = output.next_token_logits.argmax(-1).detach().cpu()
        return output
    return wrapped
