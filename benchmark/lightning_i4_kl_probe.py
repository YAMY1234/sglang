"""Test-only port of KDA FrozenReplyKL; no production backend replacement.

Original FP32 logits cross TP1 services through per-reply files. Candidates
use their actual shallow/emitter/commit/decode path, with frozen original
tokens. Full-vocabulary FP64 KL is measured BEFORE forcing those tokens.
"""
import hashlib
import json
from pathlib import Path

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class FrozenReplyKL(CustomLogitProcessor):
    def __init__(self):
        self.counts = {}
        self.reference = {}
        self.tokens = {}
        self.metrics = {}

    def __call__(self, logits, custom_param_list=None):
        assert custom_param_list and len(custom_param_list) == logits.shape[0]
        assert torch.isfinite(logits).all()
        for row, p in enumerate(custom_param_list):
            root, arm, sample = Path(p['root']), p['arm'], p['sample_id']
            steps, samples = p['steps'], p['samples']
            key = (str(root), arm, sample)
            refkey = (str(root), sample)
            step = self.counts.get(key, 0)
            assert step == len(p['__req__'].output_ids) and step <= steps
            self.counts[key] = step + 1
            if arm == 'original':
                if step == 0:
                    self.reference[refkey] = []
                    self.tokens[refkey] = []
                self.reference[refkey].append(logits[row].float().cpu())
                self.tokens[refkey].append(int(logits[row].argmax()))
                if step == steps:
                    data = dict(logits=torch.stack(self.reference[refkey]),
                                tokens=self.tokens[refkey])
                    self.reference[refkey] = data['logits']
                    pending = root / f'reference-{sample}.pt.pending'
                    torch.save(data, pending)
                    pending.replace(root / f'reference-{sample}.pt')
            else:
                if refkey not in self.reference:
                    data = torch.load(root / f'reference-{sample}.pt',
                                      map_location='cpu', weights_only=True)
                    assert data['logits'].shape == (steps + 1, logits.shape[-1])
                    self.reference[refkey], self.tokens[refkey] = data['logits'], data['tokens']
                if step:
                    assert p['__req__'].output_ids[-1] == self.tokens[refkey][step - 1]
                logp = torch.log_softmax(self.reference[refkey][step].to(
                    device=logits.device, dtype=torch.float64), -1)
                logq = torch.log_softmax(logits[row].double(), -1)
                kl = (logp.exp() * (logp - logq)).sum().item()
                assert kl >= -1e-12 and __import__('math').isfinite(kl)
                self.metrics.setdefault(key, []).append(kl)
                logits[row].fill_(-float('inf'))
                logits[row, self.tokens[refkey][step]] = 0
            if step == steps:
                receipt = dict(arm=arm, sample_id=sample, decode_steps=steps,
                    skip_prefill_prediction=True, vocab=logits.shape[-1],
                    tokens=self.tokens[refkey],
                    token_sha256=hashlib.sha256(json.dumps(self.tokens[refkey]).encode()).hexdigest(),
                    kl_direction='original || candidate', log_softmax_and_sum_dtype='float64')
                if arm != 'original':
                    receipt.update(prefill_kl=self.metrics[key][0],
                        kl_by_step=self.metrics[key][1:],
                        mean_kl=sum(self.metrics[key][1:]) / steps)
                path = root / f'{arm}-{sample}.json'
                path.write_text(json.dumps(receipt) + '\n')
        return logits


def cpu_self_test(root):
    """Exercise actual forcing, disk handoff, step alignment and KL math."""
    from types import SimpleNamespace
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    original, replay, candidate = FrozenReplyKL(), FrozenReplyKL(), FrozenReplyKL()
    source = torch.tensor([[1., 2., 3.], [-2., 1., 4.], [5., -1., 0.]])
    for arm, processor, shift in [('original', original, 0), ('replay', replay, 0),
                                  ('candidate', candidate, .25)]:
        req = SimpleNamespace(output_ids=[])
        for step in range(3):
            raw = source[step:step+1].clone(); raw[:, 0] += shift
            out = processor(raw, [dict(root=str(root), arm=arm, sample_id=0,
                steps=2, samples=1, __req__=req)])
            req.output_ids.append(int(out.argmax()))
        assert req.output_ids == source.argmax(-1).tolist()
    replay = json.loads((root / 'replay-0.json').read_text())
    candidate = json.loads((root / 'candidate-0.json').read_text())
    assert replay['mean_kl'] == 0 and replay['prefill_kl'] == 0
    q = source.double().clone(); q[:, 0] += .25
    lp, lq = source.double().log_softmax(-1), q.log_softmax(-1)
    expected = (lp.exp() * (lp-lq)).sum(-1)
    assert abs(candidate['mean_kl'] - expected[1:].mean().item()) < 1e-14
    assert candidate['prefill_kl'] == expected[0].item()
    # Serialization is the endpoint's actual import/instantiate mechanism.
    assert isinstance(CustomLogitProcessor.from_str(FrozenReplyKL.to_str()), FrozenReplyKL)
