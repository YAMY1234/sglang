#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_launch_lib.sh"

: "${R12_ROLE:?prefill|decode}"
: "${R12_SERVICE:?service generation}"
rank=${SLURM_PROCID:?}
r12_common_env
r12_build_role_cache_env "$R12_SERVICE" "$R12_ROLE" "$rank"
cache=$R12_ROLE_CACHE
mkdir -p "$cache"/{sglang,sglang-jit,xdg,triton,torchinductor,torch-extensions,cuda}

env "${R12_ENV[@]}" "${R12_ROLE_ENV[@]}" python3 - "$R12_SERVICE" "$R12_ROLE" "$rank" <<'PY'
import hashlib
import os
import pathlib
import socket
import sys

from transformers import AutoProcessor, AutoTokenizer  # noqa: F401

service, role, rank = sys.argv[1:]
AutoTokenizer.from_pretrained("/model", trust_remote_code=True)

modules = pathlib.Path(os.environ["HF_HOME"]) / "modules" / "transformers_modules" / "model"
encoding = sorted(modules.glob("*/encoding_k3.py"))
assert encoding, f"encoding_k3.py missing below {modules}"
rows = []
for encoding_path in encoding:
    tokenizer_path = encoding_path.with_name("tokenization_kimi.py")
    assert tokenizer_path.is_file(), tokenizer_path
    rows.append(
        {
            "module_dir": str(encoding_path.parent),
            "encoding_sha256": hashlib.sha256(encoding_path.read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        }
    )
assert len(rows) == 1, rows
row = rows[0]
print(
    "DYNAMIC_MODULE_PREWARM_PASS",
    f"service={service}",
    f"role={role}",
    f"rank={rank}",
    f"host={socket.gethostname()}",
    f"hf_home={os.environ['HF_HOME']}",
    f"module_dir={row['module_dir']}",
    f"encoding_sha256={row['encoding_sha256']}",
    f"tokenizer_sha256={row['tokenizer_sha256']}",
)
PY
