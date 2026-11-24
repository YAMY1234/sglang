"""A coalesced staged write-back of host pages straddling two cudaHostRegister ranges must succeed and round-trip
bitwise (it failed with "CUDA error: invalid argument" when one merged copy spanned both ranges)."""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf,
    transfer_hicache_all_layer_staged_lf_pf,
    transfer_hicache_one_layer,
    transfer_hicache_one_layer_mla,
)
from sglang.srt.mem_cache.pool_host.common import (
    _CUDA_HOST_REGISTERED_RANGES_ATTR,
    _cuda_host_register,
    _cuda_host_unregister,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda(),
    reason="The staged write-back registration test requires CUDA.",
)

PAGE_SIZE = 64
NUM_PAGES = 64
NUM_DEVICE_PAGES = 256


@pytest.fixture(autouse=True)
def _chunked_registration(monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB", "1")
    monkeypatch.setenv("SGLANG_OPT_HICACHE_BATCH_WRITE_BACK_COPY", "1")


def _token_indices(pages, device):
    pages = torch.as_tensor(pages, dtype=torch.int64).view(-1, 1)
    return (
        (pages * PAGE_SIZE + torch.arange(PAGE_SIZE).view(1, -1)).reshape(-1).to(device)
    )


def _straddling_pages(first_page_of_second_range):
    return list(
        range(
            first_page_of_second_range - NUM_PAGES // 2,
            first_page_of_second_range + NUM_PAGES // 2,
        )
    )


def test_mha_v_half_straddles_registration_boundary():
    # The DSpark draft sidecar's shape: 5 layers x 16 kv heads x 64, fp8 bytes.
    layers, heads, dim = 5, 16, 64
    size = 2048 * PAGE_SIZE  # 2 x 671 MB of host memory: two 1 GiB-chunked ranges
    host = torch.zeros((2, size, layers, heads, dim), dtype=torch.uint8)
    granularity = PAGE_SIZE * layers * heads * dim
    _cuda_host_register(host, granularity)
    try:
        ranges = getattr(host, _CUDA_HOST_REGISTERED_RANGES_ATTR)
        assert len(ranges) == 2
        v_offset = size * layers * heads * dim
        host_pages = _straddling_pages((ranges[0][1] - v_offset) // granularity)
        gen = torch.Generator(device="cuda").manual_seed(0)
        dev_k = [
            torch.randint(
                0,
                256,
                (NUM_DEVICE_PAGES * PAGE_SIZE, heads, dim),
                dtype=torch.uint8,
                device="cuda",
                generator=gen,
            )
            for _ in range(layers)
        ]
        dev_v = [
            torch.randint(
                0,
                256,
                (NUM_DEVICE_PAGES * PAGE_SIZE, heads, dim),
                dtype=torch.uint8,
                device="cuda",
                generator=gen,
            )
            for _ in range(layers)
        ]
        k_ptrs = torch.tensor(
            [t.data_ptr() for t in dev_k], dtype=torch.uint64, device="cuda"
        )
        v_ptrs = torch.tensor(
            [t.data_ptr() for t in dev_v], dtype=torch.uint64, device="cuda"
        )
        src = _token_indices(
            torch.randperm(NUM_DEVICE_PAGES)[:NUM_PAGES].tolist(), "cuda"
        )
        dst = _token_indices(host_pages, "cpu")
        staging_k = torch.empty(
            (NUM_PAGES * PAGE_SIZE, layers, heads * dim),
            dtype=torch.uint8,
            device="cuda",
        )
        staging_v = torch.empty_like(staging_k)

        transfer_hicache_all_layer_staged_lf_pf(
            k_ptrs,
            v_ptrs,
            src,
            dst,
            staging_k,
            staging_v,
            host[0],
            host[1],
            page_size=PAGE_SIZE,
        )
        torch.cuda.synchronize()
        for layer in range(layers):
            assert torch.equal(host[0][dst, layer].cuda(), dev_k[layer][src])
            assert torch.equal(host[1][dst, layer].cuda(), dev_v[layer][src])

        reload = _token_indices(list(range(100, 100 + NUM_PAGES)), "cuda")
        for layer in range(layers):
            new_k, new_v = (
                torch.zeros_like(dev_k[layer]),
                torch.zeros_like(dev_v[layer]),
            )
            transfer_hicache_one_layer(
                new_k,
                new_v,
                reload,
                host[0].transpose(0, 1)[layer],
                host[1].transpose(0, 1)[layer],
                dst.cuda(),
                element_dim=heads * dim,
                page_size=PAGE_SIZE,
            )
            torch.cuda.synchronize()
            assert torch.equal(new_k[reload], dev_k[layer][src])
            assert torch.equal(new_v[reload], dev_v[layer][src])
    finally:
        _cuda_host_unregister(host)


def test_mla_straddles_registration_boundary():
    layers, element = 8, 576
    size = 4096 * PAGE_SIZE  # 1.21 GB of host memory: two 1 GiB-chunked ranges
    host = torch.zeros((size, layers, element), dtype=torch.uint8)
    granularity = PAGE_SIZE * layers * element
    _cuda_host_register(host, granularity)
    try:
        ranges = getattr(host, _CUDA_HOST_REGISTERED_RANGES_ATTR)
        assert len(ranges) == 2
        host_pages = _straddling_pages(ranges[0][1] // granularity)
        gen = torch.Generator(device="cuda").manual_seed(0)
        dev = [
            torch.randint(
                0,
                256,
                (NUM_DEVICE_PAGES * PAGE_SIZE, element),
                dtype=torch.uint8,
                device="cuda",
                generator=gen,
            )
            for _ in range(layers)
        ]
        ptrs = torch.tensor(
            [t.data_ptr() for t in dev], dtype=torch.uint64, device="cuda"
        )
        src = _token_indices(
            torch.randperm(NUM_DEVICE_PAGES)[:NUM_PAGES].tolist(), "cuda"
        )
        dst = _token_indices(host_pages, "cpu")
        staging = torch.empty(
            (NUM_PAGES * PAGE_SIZE, layers, element), dtype=torch.uint8, device="cuda"
        )

        transfer_hicache_all_layer_mla_staged_lf_pf(
            ptrs, src, dst, staging, host, page_size=PAGE_SIZE
        )
        torch.cuda.synchronize()
        for layer in range(layers):
            assert torch.equal(host[dst, layer].cuda(), dev[layer][src])

        reload = _token_indices(list(range(100, 100 + NUM_PAGES)), "cuda")
        for layer in range(layers):
            new = torch.zeros_like(dev[layer])
            transfer_hicache_one_layer_mla(
                new,
                reload,
                host.transpose(0, 1)[layer],
                dst.cuda(),
                element_dim=element,
                page_size=PAGE_SIZE,
            )
            torch.cuda.synchronize()
            assert torch.equal(new[reload], dev[layer][src])
    finally:
        _cuda_host_unregister(host)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
