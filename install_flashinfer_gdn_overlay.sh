#!/bin/bash
# install_flashinfer_gdn_overlay.sh
# Overlay GDN BF16 MTP kernel onto the container's FlashInfer installation.
#
# Why we need this:
# - SGLang main pins flashinfer_python==0.6.7.post3 (pyproject.toml:30).
#   Upgrading to 0.6.8 breaks SGLang's flashinfer_trtllm_moe call
#   (trtllm_fp8_block_scale_moe takes 27 args in 0.6.8, SGLang passes 29).
# - PR #2679 (BF16 GDN MTP kernel) only landed in 0.6.8.
# - Workaround: keep the container's 0.6.7.post3 binaries intact, but
#   overlay the GDN Python files from a fork branch that rebases
#   PR #2679 onto the v0.6.7.post3 tag. CUDA kernels are JIT-compiled
#   from Python source, so overlaying Python is enough.
#
# Pattern inspired by the existing install-flashinfer-main.sh on lyris.

set -eux

echo "=== Before: flashinfer version ==="
python3 -c "import flashinfer; print(f'flashinfer {flashinfer.__version__}')" \
  || echo "flashinfer not installed"

FLASHINFER_SRC="/flashinfer-src"
if [ ! -d "$FLASHINFER_SRC" ]; then
    echo "ERROR: FlashInfer source not mounted at $FLASHINFER_SRC"
    echo "Ensure the recipe has:"
    echo "  extra_mount:"
    echo "    - /lustre/.../flashinfer-gdn-mtp:/flashinfer-src"
    exit 1
fi

echo "=== Overlay GDN Python files from fork (branch gdn-mtp-v0.6.7.post3) ==="
INSTALLED_DIR=$(python3 -c "import flashinfer, os; print(os.path.dirname(flashinfer.__file__))")
echo "Installed flashinfer at: $INSTALLED_DIR"

# Entry point (Python dispatch into the BF16 MTP kernel)
cp -v "$FLASHINFER_SRC/flashinfer/gdn_decode.py" "$INSTALLED_DIR/gdn_decode.py"

# Full set of GDN kernel sources (includes gdn_decode_bf16_state.py)
cp -v "$FLASHINFER_SRC/flashinfer/gdn_kernels/"*.py "$INSTALLED_DIR/gdn_kernels/"

# Clear stale JIT cache; the kernel signatures changed with PR #2679 and
# reusing the old cache would ABI-mismatch at runtime.
echo "=== Clearing stale GDN JIT cache ==="
CACHE_BASE="${FLASHINFER_WORKSPACE_BASE:-/root/.cache/flashinfer}"
for cache_dir in \
    "$CACHE_BASE/gdn_decode_bf16_state" \
    "$CACHE_BASE/gdn_decode_pretranspose" \
    "$CACHE_BASE/gdn_decode_nontranspose" \
    "$CACHE_BASE/gdn_decode_mtp"; do
    if [ -d "$cache_dir" ]; then
        echo "Removing stale cache: $cache_dir"
        rm -rf "$cache_dir"
    fi
done
# Also nuke the __pycache__ for the overlay targets so the new .py is used.
rm -f "$INSTALLED_DIR/__pycache__/gdn_decode."*.pyc || true
rm -f "$INSTALLED_DIR/gdn_kernels/__pycache__/"*.pyc || true

# Belt and suspenders — the SGLang runtime sets this too, but setting it
# here prevents the smoke test below from crashing on a transient mismatch.
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "=== Verify GDN MTP BF16 state kernel is available ==="
python3 - <<'PY'
import os
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
import flashinfer
print(f"flashinfer version: {flashinfer.__version__}")
from flashinfer.gdn_decode import gated_delta_rule_mtp
print("gated_delta_rule_mtp: OK")
from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as _bf16_mtp,
)
print("gdn_decode_bf16_state.gated_delta_rule_mtp: OK")
from flashinfer.gdn_prefill import chunk_gated_delta_rule
print("chunk_gated_delta_rule: OK")
PY

echo "=== install_flashinfer_gdn_overlay.sh complete ==="
