"""#905: publishing a reused slot's private deep mapping clears the tail past the new reservation.

Positions beyond the reservation previously kept an earlier occupant's released private locations; a write there
(the overlap scheduler's extra verify of a finished request) landed in physical units owned by other pages and
corrupted a cached prefix's Scheme-C latent row ("malformed TP gap8 stream before scatter").
"""
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.flashnext_latent_pool import FlashNextLatentPool


class PrivateTailTest(unittest.TestCase):
    def test_tail_cleared_on_publication(self):
        table = torch.zeros((4, 256), dtype=torch.int32)
        table[2, :192] = torch.arange(1000, 1192, dtype=torch.int32)      # earlier occupant: three pages
        pool = SimpleNamespace(pending_mappings={2: [5]}, deep_req_to_token=table, page_size=64, device="cpu",
                               qsa_compress_ratio=4,
                               deep=SimpleNamespace(qsa_key_state_buffer_pool=[torch.ones(16, 2)],
                                                    qsa_rope_position_buffer=torch.ones(16)))
        FlashNextLatentPool.prepare_request_mappings(pool, torch.tensor([2]))
        self.assertEqual(table[2, :64].tolist(), list(range(5 * 64, 6 * 64)))
        self.assertEqual(int(table[2, 64:].abs().sum()), 0)
        self.assertEqual(table[1].abs().sum().item(), 0)                     # other rows untouched
        self.assertEqual(pool.pending_mappings, {})


if __name__ == "__main__":
    unittest.main()
