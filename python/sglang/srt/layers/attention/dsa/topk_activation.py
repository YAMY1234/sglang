"""P-only native-load proof and bounded successful host-dispatch counters."""

import base64
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import socket
import time

logger = logging.getLogger(__name__)
MARKER = "KERNEL_AB_CALL "
NATIVE_SHA = "f2b7bc2e09c4331fa5aa7f3624e06ee2d89f62d8854d651f7d970289b7497b36"
API_SHA = "9093ea0735ac596fdb522648f078130d4a23b4ade862f6d678de3576543e6d06"
OP = "sgl_kernel.fast_topk_transform_fused_overflow_prefix"
LEGACY_OP = "sgl_kernel.fast_topk_transform_fused"
# Packaged builds (no ARIA receipt): the sha of each overflow binary, by file name.
NATIVE_MANIFEST = Path(__file__).with_name("topk_overflow_native.sha256")
_native = None
_counts = {"candidate": 0, "legacy": 0}
_first_call_epoch = None
_snapshots = 0


def load_native():
    global _native
    if _native is not None and _native["pid"] == os.getpid():
        return _native
    import torch
    import sgl_kernel

    # With the receipt (frozen packs): the frozen binary and one receipted source commit.
    # Without it (a packaged image): the binary listed in NATIVE_MANIFEST.
    receipt = os.environ.get("ARIA_SOURCE_IMPORT_EXPECTED_B64")
    binary = Path(os.environ["TOPK_OVERFLOW_NATIVE_SO"]).resolve()
    api_path = Path(os.environ["TOPK_OVERFLOW_API_SOURCE"]).resolve()
    actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    native_sha = NATIVE_SHA if receipt is not None else _manifest_sha(binary.name)
    if actual != native_sha or hashlib.sha256(api_path.read_bytes()).hexdigest() != API_SHA:
        raise RuntimeError("overflow native/API differs from the frozen build component")
    if str(torch.__version__) != "2.15.0+cu134" or torch.version.cuda != "13.4":
        raise RuntimeError("overflow native runtime differs from its build receipt")
    torch.ops.load_library(str(binary))
    if str(binary) not in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}:
        raise RuntimeError("overflow DSO was not actually loaded")
    schema = str(torch.ops.sgl_kernel.fast_topk_transform_fused_overflow_prefix.default._schema)
    spec = importlib.util.spec_from_file_location("overflow_pd_pinned_api", api_path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    sgl_kernel.fast_topk_transform_fused_overflow_prefix = api.fast_topk_transform_fused_overflow_prefix
    if receipt is not None:
        expected = json.loads(base64.b64decode(receipt))
        identities = [x["identity"] for x in expected["overlays"] if x["component"] == "sglang"]
        if len(identities) != 1 or not identities[0].get("git_commit"):
            raise RuntimeError("one complete P candidate source commit is required")
        job_id, source_commit = os.environ["SLURM_JOB_ID"], identities[0]["git_commit"]
    else:
        job_id, source_commit = os.environ.get("SLURM_JOB_ID"), os.environ.get("SGLANG_BUILD_COMMIT")
    _native = dict(schema="kernel-ab-load/1", job_id=job_id,
                   hostname=socket.gethostname(), pid=os.getpid(), epoch=time.time(),
                   source_commit=source_commit, native_sha256=actual,
                   binary=str(binary), api=str(api_path), op_symbol=OP,
                   operator_schema=schema, actual_load=True)
    logger.warning("KERNEL_AB_LOAD " + json.dumps(_native, sort_keys=True))
    return _native


def _manifest_sha(name):
    for line in NATIVE_MANIFEST.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == name:
            return fields[0]
    raise RuntimeError(f"{NATIVE_MANIFEST.name} lists no sha for {name}")


def note_dispatch(kind, score):
    """After op return only; no tensor read, CUDA sync, or replay-count claim."""
    global _first_call_epoch, _snapshots
    _counts[kind] += 1
    total = sum(_counts.values())
    if _first_call_epoch is None:
        _first_call_epoch = time.time()
    if _snapshots >= 24 or (total & (total - 1) and not
                           (kind == "candidate" and _counts[kind] == 1)):
        return
    from sglang.srt.distributed.parallel_state import get_pp_group

    native = load_native()
    job, host, pid = native["job_id"], socket.gethostname(), os.getpid()
    rank = get_pp_group().rank_in_group
    ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    switch = os.environ.get("SGLANG_DSA_TOPK_OVERFLOW_PREFIX", "0")
    row = dict(schema="kernel-ab-call/1", job_id=job,
        arm="on" if switch == "1" else "off", kernel_id="overflow-prefix-v2",
        switch_env="SGLANG_DSA_TOPK_OVERFLOW_PREFIX", switch_value=switch,
        hostname=host, pp_rank=rank, pid=pid, process_start_ticks=ticks,
        step_id=os.environ.get("SLURM_STEP_ID"), identity_key=[job, host, pid, rank],
        source_commit=native["source_commit"], native_sha256=native["native_sha256"],
        op_symbol=OP if _counts['candidate'] else LEGACY_OP,
        last_op_symbol=OP if kind == 'candidate' else LEGACY_OP,
        first_call_epoch=_first_call_epoch, snapshot_epoch=time.time(),
        new_op_calls=_counts["candidate"], legacy_op_calls=_counts["legacy"],
        eligible_calls=total, rows=score.shape[0], rowmap=os.environ.get(
            "SGLANG_DSA_LEGACY_TOPK_ROWMAP", "0"),
        scope="successful_host_dispatch; not device completion or CUDA graph replay count")
    logger.warning(MARKER + json.dumps(row, sort_keys=True))
    _snapshots += 1
