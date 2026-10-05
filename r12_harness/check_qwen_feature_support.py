#!/usr/bin/env python3
"""Fail closed unless staging=1 is supported by Qwen3.8's hybrid MHA pool."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import tarfile


FIXED_SOURCE_SHA = "10edb05e72afb79ecfdf7dbe7603f90ba05ad3ee"
SOURCE_FILES = {
    "utils": "python/sglang/srt/disaggregation/utils.py",
    "prefill": "python/sglang/srt/disaggregation/prefill.py",
    "decode": "python/sglang/srt/disaggregation/decode.py",
    "memory_pool": "python/sglang/srt/mem_cache/memory_pool.py",
    "kv_configurator": "python/sglang/srt/mem_cache/kv_cache_configurator.py",
}


def read_sources(source_root: pathlib.Path | None, archive: pathlib.Path | None):
    if source_root is not None:
        return {
            name: source_root.joinpath(relative).read_text()
            for name, relative in SOURCE_FILES.items()
        }, f"tree:{source_root}"
    assert archive is not None
    result = {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile()]
        for name, relative in SOURCE_FILES.items():
            matches = [member for member in members if member.name.endswith(relative)]
            assert len(matches) == 1, (relative, [member.name for member in matches])
            stream = bundle.extractfile(matches[0])
            assert stream is not None
            result[name] = stream.read().decode()
    return result, f"archive:{archive}"


def require_source_contract(source: dict[str, str]) -> list[str]:
    checks = {
        "hybrid_pool_has_use_mla_false_default": re.search(
            r"class HybridLinearKVPool\(KVCache\):.*?use_mla: bool = False",
            source["memory_pool"], re.S,
        ),
        "hybrid_non_mla_uses_mha_pool": re.search(
            r"elif not use_mla:\s+TokenToKVPoolClass = MHATokenToKVPool",
            source["memory_pool"],
        ),
        "configurator_passes_model_mla_mode": re.search(
            r"if qsa_profile is None:\s+pool_class = HybridLinearKVPool\s+"
            r'extra_args\["use_mla"\] = self\.use_mla_backend',
            source["kv_configurator"],
        ),
        "staging_unwraps_hybrid_pool": re.search(
            r"if isinstance\(kv_pool, HybridLinearKVPool\):\s+kv_pool = kv_pool\.full_kv_pool",
            source["utils"],
        ),
        "staging_accepts_mha_pool": re.search(
            r"if not isinstance\(kv_pool, MHATokenToKVPool\):\s+return None",
            source["utils"],
        ),
        "prefill_rejects_only_mla": re.search(
            r"if envs\.SGLANG_DISAGG_STAGING_BUFFER\.get\(\):\s+"
            r"if self\.is_mla_backend:\s+raise RuntimeError",
            source["prefill"],
        ),
        "decode_rejects_only_mla": re.search(
            r"if self\.enable_staging and self\.is_mla_backend:\s+raise RuntimeError",
            source["decode"],
        ),
    }
    missing = [name for name, match in checks.items() if match is None]
    assert not missing, f"source contract missing: {missing}"
    return sorted(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=pathlib.Path)
    source.add_argument("--source-archive", type=pathlib.Path)
    parser.add_argument("--rendered-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    source_text, source_mode = read_sources(args.source_root, args.source_archive)
    checks = require_source_contract(source_text)
    paths = sorted(args.rendered_root.glob("*-prefill-rank0.out")) + sorted(
        args.rendered_root.glob("*-decode-rank0.out")
    )
    assert len(paths) == 6, (args.rendered_root, len(paths))
    roles = {"prefill": 0, "decode": 0}
    for path in paths:
        _, raw = path.read_text().strip().split(" ", 1)
        tokens = shlex.split(raw)
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=1") == 1, path
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=0") == 0, path
        assert tokens[tokens.index("--attention-backend") + 1] == "trtllm_mha", path
        assert tokens[tokens.index("--linear-attn-prefill-backend") + 1] == "flashinfer", path
        assert tokens[tokens.index("--linear-attn-decode-backend") + 1] == "flashinfer", path
        roles["prefill" if "-prefill-" in path.name else "decode"] += 1
    assert roles == {"prefill": 3, "decode": 3}, roles
    proof = {
        "fixed_source_sha": FIXED_SOURCE_SHA,
        "source_mode": source_mode,
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pool": "HybridLinearKVPool(use_mla=False) -> MHATokenToKVPool",
        "source_contract_checks": checks,
        "launch_commands_checked": len(paths),
        "role_counts": roles,
        "staging": {
            "configured_value": 1,
            "configured_configuration_supported": True,
            "verdict": "PASS_ENABLED_SUPPORTED_FEATURE",
        },
        "verdict": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(
        "FEATURE_SUPPORT_GATE_PASS model=Qwen/Qwen3.8-Flash-Next "
        "pool=HybridLinearKVPool(use_mla=False) staging=1 commands=6 "
        "enabled_staging_support=SUPPORTED"
    )


if __name__ == "__main__":
    main()
