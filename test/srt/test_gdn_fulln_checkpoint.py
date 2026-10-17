"""docs/223 §11 gate (a): multi-turn radix replay of the full-N prompt-end checkpoint (CPU).

The next turn of an agent conversation re-renders the previous prompt's last (generation-prompt) token, so it shares
only the first P - 1 prompt tokens. The legacy route checkpoints at floor((P-1)/64)*64; the full-N route used to
checkpoint at floor(P/64)*64, which for P % 64 == 0 is P itself and is never reachable (docs/223 §10).

Replays real S2 4557497 C1 conversation chains through the real ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend
and MambaAttnBackendBase._init_track_ssm_indices, with a minimal radix: a turn donates its prompt-end checkpoint
(cache_unfinished_req: key = prompt[:mamba_last_track_seqlen]); a new turn hits
the longest donated checkpoint that prefixes it and gets mamba_branching_seqlen = page floor of its KV match when that
is longer than the hit (MambaRadixCache._match_prefix_helper). Under the prompt-only state cache decode donates nothing
(cache_unfinished_req resets mamba_last_track_seqlen, so the finish insert has cache_len 0): the radix paths are exactly
the donated prompt-end keys. Checks:
  * fixed full-N (switch 1) hits == legacy hits on every turn (no shortfall), and both equal the S32 cached_tokens
    the server logged; the old full-N (switch 0) reproduces the arm's logged cached_tokens (negative control);
  * label: whenever the backend would copy last_recurrent_state (aligned track), the tracked position is the true last
    position of that forward (full-N: prefix + length; legacy N-1 route: prefix + length - 1), else the h index names
    exactly the donated depth.
"""

import os
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

CHUNK = 64

# (prompt length, S32 logged cached_tokens, arm logged cached_tokens) per turn, S2 4557497 C1 node 0; the first turn
# is the warm turn whose checkpoint both routes already share. n1 logs the same chains (its S32 prompts carry one extra
# token); 160860 is the n1-only cascade turn. 'p3' is the P3 line's length set (65 / 128 / 129 / 4096 / 4097).
CHAINS = {
    "159232": [(137595, None, None), (159232, 137536, 137536), (160334, 159168, 137536), (160860, 160320, 159168)],
    "229056": [(222647, None, None), (229056, 222592, 222592), (229326, 228992, 222592), (232277, 229312, 228992),
               (235491, 232256, 232256), (236146, 235456, 235456), (236416, 236096, 236096),
               (240907, 236352, 236096), (243160, 240896, 236352)],
    "37440": [(37171, None, None), (37440, 37120, 37120), (37581, 37376, 37120), (38683, 37568, 37376)],
    "synthetic": [(1000, None, None), (1281, None, None), (1345, None, None), (2048, None, None), (2049, None, None),
                  (2112, None, None), (4096, None, None), (4161, None, None)],
    "p3": [(65, None, None), (128, None, None), (129, None, None), (4096, None, None), (4097, None, None)],
}


def _setup():
    server_args = ServerArgs(model_path="dummy", page_size=CHUNK)
    server_args._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(server_args)


def _track(prompt, hit, branching, fulln, end=None):
    sp = SamplingParams(max_new_tokens=1)
    sp.normalize(None)
    req = Req(rid="r", origin_input_text="", origin_input_ids=array("q", prompt), sampling_params=sp, vocab_size=1 << 30)
    req.prefix_indices = torch.arange(hit, dtype=torch.int64)
    req.set_extend_range(hit, len(prompt) if end is None else end)
    req.kv.mamba_ping_pong_track_buffer = torch.tensor([0, 1], dtype=torch.int64)
    req.kv.mamba_next_track_idx = 0
    req.mamba_branching_seqlen = branching
    req._pfactor_agg_contract = fulln
    batch = ScheduleBatch(reqs=[req])
    batch.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(mamba_chunk_size=CHUNK))
    batch.tree_cache = SimpleNamespace(page_size=CHUNK)
    batch.req_to_token_pool = MagicMock()
    batch.req_to_token_pool._prefill_prompt_only_state_cache = True
    batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = 1
    entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
    return entry, req.kv.mamba_last_track_seqlen


def _backend_source(entry, hit, forward_len):
    """('final', position) or ('h', position) as MambaAttnBackendBase._init_track_ssm_indices picks it."""
    fb = SimpleNamespace(mamba_track_mask=torch.tensor([bool(entry.track_mask)]),
                         extend_seq_lens=torch.tensor([forward_len]), mamba_track_indices=torch.tensor([7]),
                         mamba_track_seqlens=torch.tensor([entry.track_seqlen]), extend_prefix_lens=torch.tensor([hit]))
    me = SimpleNamespace(mamba_chunk_size=CHUNK, device="cpu")
    out = MambaAttnBackendBase._init_track_ssm_indices(me, torch.tensor([3]), fb)
    _, h_src, _, _, final_src, _ = out[:6]
    if final_src.numel():
        return "final", hit + forward_len
    if h_src.numel():
        idx = int(h_src[0])
        assert idx <= (forward_len - 1) // CHUNK, "h index past the forward's chunk states"
        return "h", hit + idx * CHUNK
    return None, None


def replay(lengths, mode):
    """mode: legacy | fulln_old | fulln_new. Returns per-turn (hit, donated depth)."""
    os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "0" if mode == "fulln_old" else "1"
    fulln = mode != "legacy"
    fresh = iter(range(10 ** 7, 10 ** 9))
    paths, ckpts, out, prev = [], set(), [], None
    for k, P in enumerate(lengths):
        if prev is None:
            prompt = list(range(P))
        else:  # the next turn re-renders the previous prompt's last token
            prompt = prev[:-1] + [next(fresh) for _ in range(P - (len(prev) - 1))]
        hit = max([n for n in ckpts if n < len(prompt) and tuple(prompt[:n]) in ckpts_keys[n]] or [0]) \
            if k else 0
        kv = max([_common(prompt, p) for p in paths] or [0]) // CHUNK * CHUNK
        branching = kv if kv > hit and kv > 0 else None
        entry, depth = _track(prompt, hit, branching, fulln)
        forward_len = len(prompt) - hit - (0 if fulln else 1)  # legacy is the N-1 + boundary route
        src, pos = _backend_source(entry, hit, forward_len) if entry.track_mask else (None, None)
        if entry.track_mask:
            assert pos == depth, f"{mode} turn {k}: backend {src} position {pos} != donated depth {depth}"
            ckpts.add(depth)
            ckpts_keys.setdefault(depth, set()).add(tuple(prompt[:depth]))
            paths.append(prompt[:depth])
        out.append((hit, depth, src))
        prev = prompt
    return out


ckpts_keys = {}


def _common(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class FullNCheckpointReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def setUp(self):
        ckpts_keys.clear()
        self.saved = os.environ.get("SGLANG_GDN_FULLN_CKPT_N_MINUS_1")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("SGLANG_GDN_FULLN_CKPT_N_MINUS_1", None)
        else:
            os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = self.saved

    def _run(self, name):
        lengths = [p for p, _, _ in CHAINS[name]]
        res = {}
        for mode in ("legacy", "fulln_old", "fulln_new"):
            ckpts_keys.clear()
            res[mode] = replay(lengths, mode)
        return res

    def test_server_chains(self):
        for name in ("159232", "229056", "37440"):
            with self.subTest(chain=name):
                res = self._run(name)
                hits = {m: [h for h, _, _ in r] for m, r in res.items()}
                s32 = [c for _, c, _ in CHAINS[name][1:]]
                arm = [c for _, _, c in CHAINS[name][1:]]
                self.assertEqual(hits["legacy"][1:], s32, "legacy replay must reproduce the S32 log")
                self.assertEqual(hits["fulln_old"][1:], arm, "old full-N replay must reproduce the arm log")
                self.assertEqual(hits["fulln_new"], hits["legacy"], "fixed full-N must hit like legacy")
                self.assertGreater(sum(a - b for a, b in zip(hits["legacy"], hits["fulln_old"])), 0)

    def test_synthetic_alignments_and_labels(self):
        for name in ("synthetic", "p3"):
            with self.subTest(chain=name):
                self._synthetic(name)

    def test_non_final_chunk_keeps_its_end(self):
        # a non-final 128-token chunk of a 4097-token prompt: the forward really ends at 128 -> last_recurrent_state
        for flag in ("0", "1"):
            os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = flag
            entry, depth = _track(list(range(4097)), 0, None, True, end=128)
            self.assertEqual(depth, 128)
            self.assertEqual(_backend_source(entry, 0, 128), ("final", 128))

    def _synthetic(self, name):
        res = self._run(name)
        self.assertEqual([h for h, _, _ in res["fulln_new"]], [h for h, _, _ in res["legacy"]])
        aligned_turns = [k for k, p in enumerate(CHAINS[name]) if p[0] % CHUNK == 0 and k]
        self.assertTrue(aligned_turns)
        for k in aligned_turns:  # the old route donated an unreachable P; the next turn fell short
            if k + 1 < len(res["fulln_old"]):
                self.assertLess(res["fulln_old"][k + 1][0], res["legacy"][k + 1][0])
        # fixed full-N never takes last_recurrent_state for a depth short of the forward's end
        for hit, depth, src in res["fulln_new"]:
            if src == "final":
                self.assertTrue(False, "fixed full-N final-chunk checkpoint must come from h")

    def test_switch_off_is_the_old_extent(self):
        os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "0"
        _, depth = _track(list(range(2048)), 1024, None, True)
        self.assertEqual(depth, 2048)
        os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "1"
        entry, depth = _track(list(range(2048)), 1024, None, True)
        self.assertEqual(depth, 1984)
        self.assertEqual(_backend_source(entry, 1024, 1024), ("h", 1984))
        entry, depth = _track(list(range(2049)), 1024, None, True)  # P - 1 aligned: forced onto h
        self.assertEqual(depth, 2048)
        self.assertEqual(entry.track_seqlen, 2049)
        self.assertEqual(_backend_source(entry, 1024, 1025), ("h", 2048))


if __name__ == "__main__":
    unittest.main()
