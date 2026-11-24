# L3 DRAM pool service

`nixl_dram_l3.py` is the DRAM pool that the prefill `--hicache-storage-backend nixl_dram` client connects to. Run one instance per NUMA node on each decode node, bound to that NUMA node, in the same container image as the workers, and start it before the SGLang workers. The deployment ran 8 instances (4 decode nodes x 2 NUMA nodes). It needs only the Python standard library and the `nixl` package that ships in the image.

```bash
export UCX_NET_DEVICES=<RDMA_DEVICES_UCX> UCX_MEM_MMAP_HOOK_MODE=none SGLANG_HICACHE_L3_LIFETIME_LEDGER=0
for N in 0 1; do
  numactl --cpunodebind=$N --membind=$N python3 nixl_dram_l3.py \
    --numa $N --segment-gib 331 --slab-mib 512 --port $((18800 + N)) \
    --advertise-host <THIS_NODE_IP> --registry-dir <L3_REGISTRY_DIR> \
    --ucx-devices <RDMA_DEVICES> --link-layer <InfiniBand|Ethernet> --expect-ports <PORT_COUNT> \
    --stats-interval-s 60 --grow-step-gib 64 --grow-max-gib 502 --grow-guard-pct 5 --grow-margin-gib 16 \
    --grow-hold-s 60 --grow-after-file <WORK_DIR>/server_ready.json --grow-max-gib-frontend 437 &
done
```

- Each instance prints a line starting with `L3B_READY` when it is ready. Start the workers after all of them are ready.
- The prefill side finds the services through `<L3_REGISTRY_DIR>` and expects `nixl_dram_expected_services` of them (8 here).
- Growth beyond the 331 GiB segment starts once `--grow-after-file` exists. Touch that file when all SGLang workers report ready.
