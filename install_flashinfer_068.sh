#!/bin/bash
# install_flashinfer_068.sh
# Upgrade FlashInfer to v0.6.8 inside the v0.5.10.post1 SGLang container.
#
# Usage (inside container via srt-slurm setup_script):
#   bash install_flashinfer_068.sh
#
# Why:
# - Container ships flashinfer 0.6.7.post3. We need 0.6.8 for:
#   * PR #2679 (BF16 GDN MTP kernel on SM100+ — the unblock commit in
#     SGLang dispatches to it)
#   * PR #2908 (state checkpointing in chunk_gated_delta_rule)
# - Both flashinfer-python AND flashinfer-cubin must bump together, else
#   FlashInfer's env.py raises a version-mismatch RuntimeError at import.
# - --no-deps keeps the container's pinned torch==2.9.1, cuda-python==12.9,
#   numpy, transformers==5.3.0 intact. Pulling pypi "flashinfer-python"
#   transitively upgrades torch to 2.11 which breaks everything else.

set -euo pipefail

# Allow FlashInfer to load even if there is a transient version-check
# during install. SGLang also sets this at runtime; set it here so the
# smoke test below works regardless of container env ordering.
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "[install_flashinfer_068] Before:"
pip show flashinfer-python flashinfer-cubin 2>&1 | grep -E "^Name|^Version" || echo "(not installed)"

# Uninstall both to avoid leaving stale metadata.
pip uninstall -y flashinfer flashinfer-python flashinfer-cubin 2>/dev/null || true

# Install both companion packages at matching version, without touching
# the container's existing torch / cuda / numpy / transformers.
pip install --no-cache-dir --no-deps \
    flashinfer-python==0.6.8 \
    flashinfer-cubin==0.6.8

echo "[install_flashinfer_068] After:"
pip show flashinfer-python flashinfer-cubin 2>&1 | grep -E "^Name|^Version"

# Smoke-test the imports and the symbols that Phase 0 depends on.
python - <<'PY'
import os
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import flashinfer
print(f"[install_flashinfer_068] flashinfer version: {flashinfer.__version__}")

from flashinfer.gdn_decode import gated_delta_rule_mtp
from flashinfer.gdn_prefill import chunk_gated_delta_rule
print("[install_flashinfer_068] gated_delta_rule_mtp: OK")
print("[install_flashinfer_068] chunk_gated_delta_rule: OK")

# The BF16 MTP kernel the unblock commit dispatches to on SM100+.
try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule_mtp as _bf16_mtp,
    )
    print("[install_flashinfer_068] gdn_decode_bf16_state.gated_delta_rule_mtp: OK")
except ImportError as e:
    print(f"[install_flashinfer_068] WARN: bf16 MTP symbol missing: {e}")
PY

echo "[install_flashinfer_068] Done."
