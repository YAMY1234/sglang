#!/usr/bin/env python3

import argparse
import json
import pathlib
import shlex


CACHE_ENV = {
    "CUDA_CACHE_PATH",
    "HF_HOME",
    "SGLANG_CACHE_DIR",
    "SGLANG_JIT_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
}
PP_ENV = {
    "SGLANG_PP_COMM_OVERLAP": "1",
    "SGLANG_PP_LAYER_PARTITION": "24,23,23,23",
}
VALUE_FLAGS = {
    "--chunked-prefill-size",
    "--ep-size",
    "--max-prefill-tokens",
    "--pp-size",
    "--tp-size",
}
BOOL_FLAGS = {"--disable-overlap-schedule"}


def load(path: pathlib.Path) -> tuple[str, list[str]]:
    label, raw = path.read_text().strip().split(" ", 1)
    return label, shlex.split(raw)


def split_command(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    assert tokens[0] == "env", tokens[:4]
    python_index = tokens.index("python3")
    env = {}
    for token in tokens[1:python_index]:
        key, value = token.split("=", 1)
        assert key not in env, key
        env[key] = value
    return env, tokens[python_index:]


def normalize_cache_env(env: dict[str, str], service: str) -> dict[str, str]:
    out = dict(env)
    for key in CACHE_ENV:
        assert key in out, key
        out[key] = out[key].replace(f"/{service}/", "/<SERVICE>/")
    return out


def flag_value(tokens: list[str], flag: str) -> str:
    assert tokens.count(flag) == 1, (flag, tokens)
    return tokens[tokens.index(flag) + 1]


def strip_topology(tokens: list[str]) -> list[str]:
    out = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in VALUE_FLAGS:
            index += 2
            continue
        if token in BOOL_FLAGS:
            index += 1
            continue
        out.append(token)
        index += 1
    return out


def prove(root: pathlib.Path, pp_chunk: str) -> dict:
    ranks = []
    for rank in (0, 1):
        b_label, b_tokens = load(root / f"B-main-prefill-rank{rank}.out")
        a_label, a_tokens = load(root / f"A-main-prefill-rank{rank}.out")
        assert b_label == a_label == "PREFILL_LAUNCH"
        b_env, b_args = split_command(b_tokens)
        a_env, a_args = split_command(a_tokens)
        b_env = normalize_cache_env(b_env, "B-main")
        a_env = normalize_cache_env(a_env, "A-main")
        for key, value in PP_ENV.items():
            assert key not in b_env, (rank, key, b_env)
            assert a_env.pop(key) == value, (rank, key, a_env)
        assert a_env == b_env, (rank, a_env, b_env)
        assert flag_value(b_args, "--tp-size") == "8"
        assert flag_value(b_args, "--ep-size") == "8"
        assert "--pp-size" not in b_args
        assert "--disable-overlap-schedule" not in b_args
        assert flag_value(b_args, "--chunked-prefill-size") == "8192"
        assert flag_value(b_args, "--max-prefill-tokens") == "8192"
        assert flag_value(a_args, "--tp-size") == "2"
        assert flag_value(a_args, "--ep-size") == "2"
        assert flag_value(a_args, "--pp-size") == "4"
        assert a_args.count("--disable-overlap-schedule") == 1
        assert flag_value(a_args, "--chunked-prefill-size") == pp_chunk
        assert flag_value(a_args, "--max-prefill-tokens") == pp_chunk
        assert strip_topology(a_args) == strip_topology(b_args), rank

        b_decode_label, b_decode_tokens = load(root / f"B-main-decode-rank{rank}.out")
        a_decode_label, a_decode_tokens = load(root / f"A-main-decode-rank{rank}.out")
        assert b_decode_label == a_decode_label == "DECODE_LAUNCH"
        b_decode_env, b_decode_args = split_command(b_decode_tokens)
        a_decode_env, a_decode_args = split_command(a_decode_tokens)
        b_decode_env = normalize_cache_env(b_decode_env, "B-main")
        a_decode_env = normalize_cache_env(a_decode_env, "A-main")
        assert a_decode_env == b_decode_env, rank
        assert a_decode_args == b_decode_args, rank
        ranks.append(
            {
                "rank": rank,
                "prefill_only_differences": {
                    "env": PP_ENV,
                    "topology": {
                        "A": "TP2xEP2xPP4 disable-overlap-schedule",
                        "B": "TP8xEP8",
                    },
                    "chunk": {"A": int(pp_chunk), "B": 8192},
                },
                "decode_semantic_diff": [],
            }
        )
    return {
        "cache_identity_normalized": "service only",
        "ranks": ranks,
        "verdict": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--pp-chunk", choices=("8192", "16384"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    record = prove(args.root, args.pp_chunk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        "SEMANTIC_PROOF_PASS",
        f"root={args.root}",
        f"pp_chunk={args.pp_chunk}",
        "prefill_diff=pp_env+topology+chunk",
        "decode_diff=none",
    )


if __name__ == "__main__":
    main()
