#!/bin/bash
# install_flashinfer_068.sh
# Install FlashInfer v0.6.8 on top of the v0.5.10.post1 SGLang container.
#
# Usage (inside container via srt-slurm pre-command or manual):
#   bash install_flashinfer_068.sh
#
# Why:
# - The v0.5.10.post1 container ships an older FlashInfer (v0.6.4-ish).
# - Amey's GDN BF16 MTP kernel (PR #2679) landed in FlashInfer v0.6.8.
# - PR #2908 (state checkpointing in chunk_gated_delta_rule) also in v0.6.8 —
#   required by Phase 0 partial-recompute path (`chunk_gated_delta_rule` with
#   `initial_state=h_0`, k/v slices, `output_final_state=True`).

set -euo pipefail

echo "[install_flashinfer_068] Current flashinfer version:"
python -c "import flashinfer; print(flashinfer.__version__)" 2>&1 || echo "(not installed)"

# Uninstall old version to avoid ABI / metadata conflicts
pip uninstall -y flashinfer flashinfer-python 2>/dev/null || true

# Install v0.6.8 from PyPI.
# --no-cache-dir: avoid stale wheel caches.
# --force-reinstall: overwrite even if a version happens to be present.
pip install --no-cache-dir --force-reinstall flashinfer-python==0.6.8

echo "[install_flashinfer_068] Installed flashinfer version:"
python -c "import flashinfer; print(flashinfer.__version__)"

# Smoke-test the GDN MTP symbol we depend on (Phase 0 uses this kernel).
python - <<'PY'
try:
    from flashinfer.gdn_decode import gated_delta_rule_mtp
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    print("[install_flashinfer_068] gated_delta_rule_mtp:",
          gated_delta_rule_mtp is not None)
    print("[install_flashinfer_068] chunk_gated_delta_rule:",
          chunk_gated_delta_rule is not None)
except ImportError as e:
    print(f"[install_flashinfer_068] SMOKE TEST FAILED: {e}")
    raise SystemExit(1)
PY

echo "[install_flashinfer_068] Done."
