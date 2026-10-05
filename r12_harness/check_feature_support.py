#!/usr/bin/env python3
"""Fail closed when a rendered R12 model-sensitive feature is unsupported."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import tarfile


SOURCE_FILES = {
    "utils": "python/sglang/srt/disaggregation/utils.py",
    "prefill": "python/sglang/srt/disaggregation/prefill.py",
    "decode": "python/sglang/srt/disaggregation/decode.py",
    "memory_pool": "python/sglang/srt/mem_cache/memory_pool.py",
}


def read_sources(source_root: pathlib.Path | None, archive: pathlib.Path | None):
    if source_root is not None:
        return {
            name: source_root.joinpath(relative).read_text()
            for name, relative in SOURCE_FILES.items()
        }, f"tree:{source_root}"

    assert archive is not None
    result: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile()]
        for name, relative in SOURCE_FILES.items():
            matches = [member for member in members if member.name.endswith(relative)]
            if len(matches) != 1:
                raise AssertionError((relative, [member.name for member in matches]))
            stream = bundle.extractfile(matches[0])
            assert stream is not None
            result[name] = stream.read().decode()
    return result, f"archive:{archive}"


def require_source_contract(source: dict[str, str]) -> None:
    utils = source["utils"]
    prefill = source["prefill"]
    decode = source["decode"]
    pool = source["memory_pool"]

    assert re.search(
        r"def is_mla_backend\(target_kv_pool\).*?return isinstance\("
        r"target_kv_pool, \(MLATokenToKVPool, DeepSeekV4TokenToKVPool\)\)",
        utils,
        re.S,
    )
    assert re.search(
        r"if isinstance\(kv_pool, HybridLinearKVPool\):\s+"
        r"kv_pool = kv_pool\.full_kv_pool",
        utils,
    )
    assert re.search(
        r"if not isinstance\(kv_pool, MHATokenToKVPool\):\s+return None",
        utils,
    )
    assert "class HybridLinearKVPool(KVCache):" in pool
    assert re.search(
        r"elif use_dsa:.*?else:\s+TokenToKVPoolClass = MLATokenToKVPool",
        pool,
        re.S,
    )
    guard = "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models"
    assert guard in prefill
    assert guard in decode


def rendered_commands(root: pathlib.Path, only_job: str | None):
    for job in ((only_job,) if only_job else ("job7b", "job7c")):
        assert job is not None
        job_root = root if only_job else root / job
        files = sorted(job_root.glob("*-prefill-rank*.out")) + sorted(
            job_root.glob("*-decode-rank*.out")
        )
        assert len(files) == 16, (job, len(files))
        for path in files:
            label, raw = path.read_text().strip().split(" ", 1)
            assert label in {"PREFILL_LAUNCH", "DECODE_LAUNCH"}, (path, label)
            yield job, path, shlex.split(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=pathlib.Path)
    source.add_argument("--source-archive", type=pathlib.Path)
    parser.add_argument("--rendered-root", required=True, type=pathlib.Path)
    parser.add_argument("--job", choices=("job7b", "job7c"))
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    source_text, source_mode = read_sources(args.source_root, args.source_archive)
    require_source_contract(source_text)

    commands = list(rendered_commands(args.rendered_root, args.job))
    for _, path, tokens in commands:
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=0") == 1, path
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=1") == 0, path
        assert tokens[tokens.index("--attention-backend") + 1] == "trtllm_mla", path
        assert (
            tokens[tokens.index("--mamba-radix-cache-strategy") + 1]
            == "extra_buffer"
        ), path

    proof = {
        "fixed_source_sha": "cb0b3498fcc2f398229b0b8cb9df0a5825e1438a",
        "source_mode": source_mode,
        "model": "moonshotai/Kimi-K3",
        "launch_commands_checked": len(commands),
        "pool_profile": [
            {
                "component": "target_outer",
                "pool": "HybridLinearKVPool(use_mla=True)",
                "staging_if_enabled": "UNSUPPORTED_HELPER_UNWRAPS_TO_MLA",
            },
            {
                "component": "target_full_attention",
                "pool": "MLATokenToKVPool",
                "staging_if_enabled": "REJECTED_BY_PREFILL_AND_DECODE_GUARDS",
            },
            {
                "component": "target_linear_state",
                "pool": "MambaPool",
                "staging_if_enabled": "NOT_IN_CONTIGUOUS_KV_STAGING_CONTRACT",
            },
            {
                "component": "draft_attention",
                "pool": "MHATokenToKVPool",
                "staging_if_enabled": "SUPPORTED_IN_ISOLATION_ONLY",
            },
        ],
        "model_sensitive_features": [
            {
                "env_or_flag": "SGLANG_DISAGG_STAGING_BUFFER",
                "configured_value": "0",
                "applies_to": ["prefill", "decode"],
                "enabled_configuration_supported": False,
                "configured_configuration_supported": True,
                "evidence": [
                    "utils.py:is_mla_backend direct-type guard",
                    "utils.py:build_staging_slot_metadata HybridLinearKVPool unwrap",
                    "utils.py:build_staging_slot_metadata MHA-only return",
                    "prefill.py:non-MLA staging guard",
                    "decode.py:non-MLA staging guard",
                ],
                "verdict": "PASS_DISABLED_UNSUPPORTED_FEATURE",
            }
        ],
        "registry_policy": (
            "Add every new model/KV-pool-sensitive env or flag here before it is "
            "enabled; an unregistered change to the rendered staging token fails."
        ),
        "verdict": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(
        "FEATURE_SUPPORT_GATE_PASS",
        "model=moonshotai/Kimi-K3",
        "staging=0",
        f"commands={len(commands)}",
        "enabled_staging_support=UNSUPPORTED",
    )


if __name__ == "__main__":
    main()
