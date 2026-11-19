"""Write arms.json: server args and env of the two prefill arms (PP8, TEP8+DCP8) and the TEP8 prims_ts variant.

Every flag is listed here; launch_server.py turns one arm into a `sglang.launch_server` command. Shared settings are
in COMMON_*; each arm adds or overrides only its layout-specific entries. The draft model path is left as the
placeholder @DRAFT_MODEL_PATH@ and filled in by launch_server.py from the environment.
usage: python3 make_arms.py [OUT_DIR]   (default: this directory). Prints a field diff of pp8 vs tep8.
"""
import json
import os
import sys

COMMON_ARGS = {
    # PD prefill with Mooncake; the client sends the fake bootstrap host, so the fake KV sender is used
    "disaggregation-transfer-backend": "mooncake",
    "disaggregation-decode-allocation-policy": "early",
    # model, KV, speculative decoding (DFLASH with block verification, same draft for both arms)
    "served-model-name": "moonshotai/Kimi-K3.1-Pro",
    "trust-remote-code": True,
    "reasoning-parser": "kimi_k3",
    "tool-call-parser": "kimi_k3",
    "kv-cache-dtype": "fp8_e4m3",
    "page-size": 64,
    "speculative-algorithm": "DFLASH",
    "speculative-draft-model-path": "@DRAFT_MODEL_PATH@",
    "speculative-draft-attention-backend": "cutedsl_mla",
    "speculative-draft-kv-cache-dtype": "fp8_e4m3",
    "speculative-num-draft-tokens": 6,
    "speculative-use-block-verification": True,
    # DSA attention backends (the TEP8 screen picked trtllm over prims_ts, see README)
    "dsa-prefill-backend": "trtllm",
    "dsa-decode-backend": "trtllm",
    # cache: HiCache host tier with write-back, Mamba (KDA) state cache, shortest-prefill-first
    "enable-hierarchical-cache": True,
    "hicache-size": 230,
    "hicache-write-policy": "write_back",
    "mamba-radix-cache-strategy": "extra_buffer",
    "mamba-track-interval": 256,
    "mamba-full-memory-ratio": 1.2,
    "mamba-ssm-dtype": "float32",
    "schedule-policy": "shortest-prefill-first",
    # misc
    "enable-cache-report": True,
    "enable-metrics": True,
    "gc-threshold": [700, 10, 1000],
    "watchdog-timeout": 3600,
    "model-loader-extra-config": "{\"enable_multithread_load\":true,\"num_threads\":4}",
    "weight-loader-disable-mmap": True,
    "weight-loader-drop-cache-after-load": False,
    "weight-loader-prefetch-checkpoints": False,
}

COMMON_ENV = {
    "CUDA_MODULE_LOADING": "EAGER",
    "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "1800",
    "PYTHONUNBUFFERED": "1",
    "PYTHONNOUSERSITE": "1",
    # multi-node NVLink domain (VR200: 2 nodes x 4 GPUs in one NVLink domain)
    "MC_FORCE_MNNVL": "1",
    "NCCL_MNNVL_ENABLE": "1",
    "NCCL_CUMEM_ENABLE": "1",
    "NVSHMEM_REMOTE_TRANSPORT": "none",
    "WITH_NVIDIA_PEERMEM": "0",
    "SGLANG_NUMA_BIND_V2": "0",
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
    "SGLANG_WARMUP_BF16_GEMM_MAX_M": "32768",
    "SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB": "32768",
    "SGLANG_DCP_PUSH_COMBINE_MAX_ROWS": "1536",
    "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION": "0.2",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
    "DG_MEGA_MOE_SM107_K64": "0",
    # KDA (linear attention) prefill
    "SGLANG_K31_KDA_PREFILL_FI_SM107": "1",
    "SGLANG_KDA_PREFILL_CONV_QKV_PLANES": "1",
    # layout-independent cache switches (both arms)
    "SGLANG_MAMBA_PATH_CAP_COUNTS_HOST": "1",
    "SGLANG_HICACHE_MAMBA_HOST_SCALE": "0.65",
    "SGLANG_HICACHE_WB_EVICT_BATCH": "1",
    "SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE": "1",
    "SGLANG_HICACHE_LOAD_BLOCK_QUOTA": "64",
    "SGLANG_HICACHE_MAMBA_BACKUP_BLOCK_QUOTA": "16",
    "SGLANG_HICACHE_HOST_USAGE_LOG_INTERVAL": "60",
}

PP8 = {
    "args": {
        "pipeline-parallel-size": 8,
        "tensor-parallel-size": 1,
        "ep-size": 1,
        "moe-runner-backend": "flashinfer_mxfp4",
        "chunked-prefill-size": 16384,
        "max-prefill-tokens": 16384,
        "mem-fraction-static": 0.90,
        "mamba-max-states-per-path": 3,
    },
    "env": {
        "SGLANG_PP_LAYER_PARTITION": "12,12,12,12,12,12,11,10",
        "SGLANG_PP_PD_PREFILL_RELEASE_RELAY": "1",
        "SGLANG_PP_PD_PREFILL_EARLY_SEND": "1",
    },
}

TEP8 = {  # public-stack TEP8+DCP8 with its best public flags
    "args": {
        "tensor-parallel-size": 8,
        "dcp-size": 8,
        "dcp-comm-backend": "fi_a2a",
        "ep-size": 8,
        "moe-a2a-backend": "megamoe",
        "chunked-prefill-size": 32768,
        "max-prefill-tokens": 32768,
        "mem-fraction-static": 0.88,
        "mamba-max-states-per-path": 1,
    },
    "env": {
        "SGLANG_K3_SP_COLLECTIVE": "1",
        "SGLANG_K3_SP_ATTN_RES": "1",
        "SGLANG_K3_SP_GEMM_RS_MIN_TOKENS": "16384",
        "SGLANG_DSA_DCP_PREFILL_FUSED_GATHER": "1",
        "SGLANG_K31_DSA_PREFILL_ROW_SPLIT": "1",
        "SGLANG_K31_DSA_PER_WINDOW_LOGITS": "1",
    },
}

TEP8_PRIMS = {  # screened and not selected (prims_ts / trtllm = 0.999)
    "args": {"dsa-prefill-backend": "prims_ts", "dsa-decode-backend": "prims_ts"},
    "env": {"SGLANG_ENABLE_PRIMS_TS_SM107": "1"},
}


def build(*layers):
    args, env = dict(COMMON_ARGS), dict(COMMON_ENV)
    for layer in layers:
        args.update(layer["args"])
        env.update(layer["env"])
    return {"args": args, "env": env}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    arms = {"pp8": build(PP8), "tep8": build(TEP8), "tep8-prims": build(TEP8, TEP8_PRIMS)}
    with open(os.path.join(out, "arms.json"), "w") as f:
        json.dump(arms, f, indent=1, sort_keys=True)
        f.write("\n")
    for kind in ("args", "env"):
        keys = sorted(set(arms["pp8"][kind]) | set(arms["tep8"][kind]))
        same = [k for k in keys if arms["pp8"][kind].get(k) == arms["tep8"][kind].get(k)]
        print(f"== {kind}: {len(keys)} keys, {len(same)} identical")
        for k in keys:
            a, b = arms["pp8"][kind].get(k), arms["tep8"][kind].get(k)
            if a != b:
                print(f"  {k:52s} pp8={a!s:28.28s} tep8={b!s:.40s}")


if __name__ == "__main__":
    main()
