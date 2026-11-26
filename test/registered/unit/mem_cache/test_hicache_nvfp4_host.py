"""The NVFP4 HiCache host tier (--hicache-host-kv-dtype nvfp4).

FP8 rows encode to the linear-scale NVFP4 KV layout with each block scale equal
to the block's FP8 amax, decode back within NVFP4 error, and round-trip
GPU -> host -> GPU through the MLA KV and DSA index-key host pools.
"""

import unittest

import torch

from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
    prepare_host_pool_config,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.nvfp4_host import (
    DSAIndexerPoolNVFP4Host,
    MLATokenToKVPoolNVFP4Host,
    nvfp4_row_codec,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import break_fp4_bytes, nvfp4_kv_quantize
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_GLOBAL_SCALE = torch.tensor([1.0 / 6.0])
_PAGE_SIZE = 64


def _random_fp8(shape, device="cpu"):
    """FP8 E4M3 bytes without the NaN codes 0x7F / 0xFF."""
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    return torch.where((x & 0x7F) == 0x7F, torch.zeros_like(x), x)


def _probe_rows(dim):
    """Every finite FP8 byte, zero and -0 rows, subnormal-only and +-448 blocks,
    E2M1 ties, then random rows."""
    finite = torch.tensor(
        [b for b in range(256) if b not in (0x7F, 0xFF)], dtype=torch.uint8
    )
    rows = finite.repeat(8 * dim // finite.numel() + 1)[: 8 * dim].view(8, dim)
    rows = rows.clone()
    rows[0] = 0
    rows[1] = 0x80
    rows[2] = torch.arange(1, 8, dtype=torch.uint8).repeat(dim // 7 + 1)[:dim]
    rows[3, ::16] = 0x7E
    rows[4, ::16] = 0xFE
    # amax 384 (0x7C) with 320 (0x7A) and 64 (0x60): scaled values 5.0 and 1.0.
    rows[5] = torch.tensor([0x7C, 0x7A, 0x60, 0xFA] * (dim // 4), dtype=torch.uint8)
    return torch.cat((rows, _random_fp8((56, dim))))


def _reference(rows):
    """(E2M1 values, E4M3 scale bytes, decoded FP8 values) of CPU FP8 byte rows."""
    n, dim = rows.shape
    x = rows.view(torch.float8_e4m3fn).float()
    fp4, sf = nvfp4_kv_quantize(x, _GLOBAL_SCALE)
    vals = break_fp4_bytes(fp4).view(n, dim // 16, 16)
    scale = sf.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    decoded = ((vals * scale) * _GLOBAL_SCALE).view(n, dim)
    decoded = decoded.to(torch.bfloat16).to(torch.float8_e4m3fn).float()
    # The sign of a zero E2M1 code is not part of the contract; compare values.
    return break_fp4_bytes(fp4), sf, decoded


def _fp8_values(rows):
    return rows.cpu().view(torch.float8_e4m3fn).float()


def _token_indices(pages, device):
    return torch.cat(
        [torch.arange(p * _PAGE_SIZE, (p + 1) * _PAGE_SIZE) for p in pages]
    ).to(device)


class TestNVFP4HostTier(CustomTestCase):
    def setUp(self):
        if not (torch.cuda.is_available() and is_cuda()):
            self.skipTest("The NVFP4 host tier requires CUDA.")
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def test_codec_matches_reference_within_nvfp4_error(self):
        for dim in (128, 576):
            with self.subTest(dim=dim):
                rows = _probe_rows(dim)
                n, data = rows.shape[0], dim // 2
                codec = nvfp4_row_codec(dim, "cuda")
                # The startup probe passed, so the Triton kernels are in use.
                self.assertEqual(codec.backend, "triton")
                idx = torch.arange(n, device="cuda")
                enc = torch.zeros(
                    (1, n, codec.row_bytes), dtype=torch.uint8, device="cuda"
                )
                codec.encode_rows(rows.cuda(), idx, enc, page_size=n)
                dec = torch.zeros((n, dim), dtype=torch.uint8, device="cuda")
                codec.decode_rows(enc[0], idx, dec)
                enc = enc[0].cpu()

                ref_vals, ref_sf, ref_decoded = _reference(rows)
                self.assertTrue(torch.equal(enc[:, data:], ref_sf))
                self.assertTrue(torch.equal(break_fp4_bytes(enc[:, :data]), ref_vals))
                self.assertTrue(torch.equal(_fp8_values(dec), ref_decoded))

                # Each block scale is the block's FP8 amax, so positive unless the
                # block is zero; values stay within half an E2M1 step plus the
                # FP8 rounding of the result.
                x = _fp8_values(rows).view(n, dim // 16, 16)
                amax = x.abs().amax(-1)
                scales = enc[:, data:].view(torch.float8_e4m3fn).float()
                self.assertTrue(torch.equal(scales, amax))
                self.assertTrue((scales[amax > 0] > 0).all())
                err = (_fp8_values(dec).view(n, dim // 16, 16) - x).abs()
                self.assertTrue((err <= amax.unsqueeze(-1) * 0.25 + 2**-9).all())

    def test_host_pools_round_trip_fp8_kv_and_index_keys(self):
        layer_num = 2
        device_pool = DSATokenToKVPool(
            size=_PAGE_SIZE * 8,
            page_size=_PAGE_SIZE,
            kv_lora_rank=128,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=32,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=160,
            index_head_dim=128,
        )
        kv_host = MLATokenToKVPoolNVFP4Host(
            device_pool.get_device_pool_infos()[0].buffer_info,
            2.0,
            0,
            _PAGE_SIZE,
            "page_first_direct",
            pin_memory=True,
            device="cpu",
            override_kv_cache_dim=device_pool.kv_cache_dim,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            target_layer_num=layer_num,
        )
        self.addCleanup(kv_host.destroy)
        index_config = prepare_host_pool_config(
            root=device_pool,
            full_layer_mapping={i: i for i in range(layer_num)},
            transfer_layer_id_max=layer_num,
            transfer_page_size=_PAGE_SIZE,
        ).sidecars[0]
        index_host = DSAIndexerPoolNVFP4Host(
            index_config.buffer_info,
            layer_mapping=index_config.device_to_host_layer,
            target_device_layer_num=layer_num,
            num_host_pages=kv_host.page_num,
            transfer_page_size=_PAGE_SIZE,
            layout="page_first_direct",
            buffer_device=device_pool.device,
            dtype=device_pool.store_dtype,
            pin_memory=True,
            device="cpu",
        )
        self.addCleanup(index_host.destroy)
        self.assertEqual(kv_host.kv_cache_dim, 80 + 10)
        self.assertEqual(index_host.indexer_page_stride_size, _PAGE_SIZE * (72 + 4))

        key_bytes = _PAGE_SIZE * 128
        for kv, index in zip(
            device_pool.kv_buffer, device_pool.index_k_with_scale_buffer
        ):
            kv.copy_(_random_fp8(kv.shape, "cuda"))
            index[:, :key_bytes] = _random_fp8((index.shape[0], key_bytes), "cuda")
            index[:, key_bytes:] = torch.randint(
                0, 256, (index.shape[0], index.shape[1] - key_bytes), device="cuda"
            ).to(torch.uint8)
        kv_before = [kv.cpu() for kv in device_pool.kv_buffer]
        index_before = [index.cpu() for index in device_pool.index_k_with_scale_buffer]

        src_pages, host_pages, dst_pages = [1, 2, 3], [0, 2, 4], [4, 5, 6]
        host = _token_indices(host_pages, "cpu")
        # cudaMemcpyBatchAsync, which the direct copies use, refuses the legacy
        # default stream; the HiCache controller runs them on its own streams.
        with torch.cuda.stream(torch.cuda.Stream()):
            for pool in (kv_host, index_host):
                h, d = pool.prepare_transfer_indices(
                    host, _token_indices(src_pages, "cuda"), "direct"
                )
                pool.backup_from_device_all_layer(device_pool, h, d, "direct")
                h, d = pool.prepare_transfer_indices(
                    host, _token_indices(dst_pages, "cuda"), "direct"
                )
                for layer_id in range(layer_num):
                    pool.load_to_device_per_layer(device_pool, h, d, layer_id, "direct")
        torch.cuda.synchronize()

        for layer in range(layer_num):
            kv_after = device_pool.kv_buffer[layer].cpu()
            index_after = device_pool.index_k_with_scale_buffer[layer].cpu()
            for s, hp, d in zip(src_pages, host_pages, dst_pages):
                src = slice(s * _PAGE_SIZE, (s + 1) * _PAGE_SIZE)
                dst = slice(d * _PAGE_SIZE, (d + 1) * _PAGE_SIZE)
                ref_vals, ref_sf, ref_decoded = _reference(kv_before[layer][src, 0])
                # Host page: [page, layer, token, 1, (E2M1 | E4M3 scales)].
                stored = kv_host.kv_buffer[hp, layer, :, 0]
                self.assertTrue(torch.equal(stored[:, 80:], ref_sf))
                self.assertTrue(torch.equal(break_fp4_bytes(stored[:, :80]), ref_vals))
                self.assertTrue(torch.equal(_fp8_values(kv_after[dst, 0]), ref_decoded))

                keys = index_before[layer][s, :key_bytes].view(_PAGE_SIZE, 128)
                _, _, ref_keys = _reference(keys)
                got_keys = index_after[d, :key_bytes].view(_PAGE_SIZE, 128)
                self.assertTrue(torch.equal(_fp8_values(got_keys), ref_keys))
                # Index-key scales are FP32 bytes and stay exact.
                self.assertTrue(
                    torch.equal(
                        index_after[d, key_bytes:], index_before[layer][s, key_bytes:]
                    )
                )

    def test_pagewise_load_matches_per_layer_load(self):
        """The page-wise load (SGLANG_HICACHE_PAGEWISE_LOAD) decodes the same FP8
        bytes as the per-layer load, from whole pages or from runs of layers."""
        layer_num = 3
        device_pool = DSATokenToKVPool(
            size=_PAGE_SIZE * 20,
            page_size=_PAGE_SIZE,
            kv_lora_rank=128,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=32,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=160,
            index_head_dim=128,
        )
        kv_host = MLATokenToKVPoolNVFP4Host(
            device_pool.get_device_pool_infos()[0].buffer_info,
            2.0,
            0,
            _PAGE_SIZE,
            "page_first_direct",
            pin_memory=True,
            device="cpu",
            override_kv_cache_dim=device_pool.kv_cache_dim,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            target_layer_num=layer_num,
        )
        self.addCleanup(kv_host.destroy)
        self.assertTrue(kv_host.pagewise_load_ok("direct"))
        # Every row starts random, so a load that misses a row is caught.
        for kv in device_pool.kv_buffer:
            kv.copy_(_random_fp8(kv.shape, "cuda"))

        host = _token_indices([0, 1, 2, 5], "cpu")
        per_layer_pages = [4, 5, 6, 7]
        # Destination pages -> the layer lists of the page-wise loads into them.
        pagewise_loads = {
            (8, 9, 10, 11): [[0, 1, 2]],
            (12, 13, 14, 15): [[2, 0], [1]],
        }
        with torch.cuda.stream(torch.cuda.Stream()):
            h, d = kv_host.prepare_transfer_indices(
                host, _token_indices([1, 2, 3, 16], "cuda"), "direct"
            )
            kv_host.backup_from_device_all_layer(device_pool, h, d, "direct")
            h, d = kv_host.prepare_transfer_indices(
                host, _token_indices(per_layer_pages, "cuda"), "direct"
            )
            for layer_id in range(layer_num):
                kv_host.load_to_device_per_layer(device_pool, h, d, layer_id, "direct")
            for pages, layer_lists in pagewise_loads.items():
                h, d = kv_host.prepare_transfer_indices(
                    host, _token_indices(pages, "cuda"), "direct"
                )
                for layer_ids in layer_lists:
                    kv_host.load_to_device_layers_pagewise(device_pool, h, d, layer_ids)
        torch.cuda.synchronize()

        for layer in range(layer_num):
            kv = device_pool.kv_buffer[layer].cpu()
            want = kv[_token_indices(per_layer_pages, "cpu")]
            for pages in pagewise_loads:
                got = kv[_token_indices(pages, "cpu")]
                self.assertTrue(torch.equal(got, want), (layer, pages))


if __name__ == "__main__":
    unittest.main()
