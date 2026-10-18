# SPDX-License-Identifier: Apache-2.0
"""Check a CUDA ReplaySSM stage against the current operator and FP64 math."""

import argparse
import ctypes as C
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess

import torch

import bench_gdn_half_layer_r2 as stock
from check_gdn_verify_numerics import reference


POINTER_FIELDS = (
    "q k v a b a_log dt_bias out checkpoint d_cache k_cache g_cache rawv_cache "
    "rawk_cache beta_cache query_start_loc state_indices replay_indices "
    "write_pos cache_base is_flush"
).split()
INT_FIELDS = "h hv null_block_id".split()
BOOL_FIELDS = (
    "gates_fp32 dt_bias_fp32 state_indices_int64 replay_indices_int64"
).split()
STRIDE_FIELDS = (
    "q_t k_t v_t a_t b_t o_t state_slot d_slot k_slot g_slot rawv_slot "
    "rawk_slot beta_slot qsl state_indices replay_indices"
).split()


class Params(C.Structure):
    _fields_ = (
        [(name, C.c_void_p) for name in POINTER_FIELDS]
        + [(name, C.c_int) for name in INT_FIELDS]
        + [(name, C.c_bool) for name in BOOL_FIELDS]
        + [("storage", C.c_int), ("scale", C.c_float)]
        + [("stride_" + name, C.c_int64) for name in STRIDE_FIELDS]
    )


def emit(**record):
    print(json.dumps(record), flush=True)


def bind(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.gdn_replay_probe_params_bytes.restype = C.c_size_t
    lib.gdn_replay_probe_scratch_bytes.restype = C.c_size_t
    lib.gdn_replay_probe_params_init.argtypes = [C.POINTER(Params)]
    lib.gdn_replay_probe_params_init.restype = None
    lib.gdn_replay_probe_param_offsets.argtypes = [C.POINTER(C.c_uint64), C.c_size_t]
    lib.gdn_replay_probe_param_offsets.restype = C.c_size_t
    lib.gdn_replay_probe_launch.argtypes = [
        C.POINTER(Params), C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p
    ]
    lib.gdn_replay_probe_launch.restype = C.c_int
    lib.gdn_replay_probe_attributes.argtypes = [C.c_int, C.POINTER(C.c_uint64)]
    lib.gdn_replay_probe_attributes.restype = C.c_int
    lib.gdn_replay_probe_compare_bytes.argtypes = [
        C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p
    ]
    lib.gdn_replay_probe_compare_bytes.restype = C.c_int
    assert lib.gdn_replay_probe_params_bytes() == C.sizeof(Params)
    n = len(Params._fields_)
    offsets = (C.c_uint64 * n)()
    assert lib.gdn_replay_probe_param_offsets(offsets, n) == n
    assert list(offsets) == [getattr(Params, name).offset for name, _ in Params._fields_]
    attributes = {}
    for role in (0, 1):
        values = (C.c_uint64 * 8)()
        rc = lib.gdn_replay_probe_attributes(role, values)
        assert rc == 0, (role, rc)
        attributes[role] = dict(zip(
            ("registers", "local_bytes", "shared_bytes", "max_threads",
             "binary_version", "ptx_version", "params_bytes", "scratch_bytes"),
            map(int, values),
        ))
    emit(kind="native_abi", bytes=C.sizeof(Params), fields=n, offsets=list(offsets),
         attributes=attributes, role_pool_admitted=80 <= attributes[1]["registers"] <= 128)
    return lib, attributes


def clone_case(case):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in case.items()}


def prepare(tp, seed):
    layer = stock.Layer(tp, seed)
    qkvz = stock.apply_fp8_linear_bmm_flashinfer(layer.x, layer.wq, layer.sw, layer.sx)
    ba = torch.nn.functional.linear(layer.x, layer.wba)
    mixed, _, b, a = stock.qwen3_5_gdn_prefill_projection_views(
        qkvz, ba, layer.h, layer.hv, 128, 128)
    conv = stock.causal_conv1d_update(
        mixed.view(1, 7, layer.dim).transpose(1, 2), layer.conv_state, layer.conv_w,
        activation="silu", conv_state_indices=layer.case["ssm_state_indices"],
        intermediate_conv_window=layer.window,
        intermediate_state_indices=layer.case["replay_indices"],
        token_major_out=True, block_n=128,
    ).transpose(1, 2).view(7, layer.dim)
    q, k, v = conv.split((layer.h * 128, layer.h * 128, layer.hv * 128), dim=-1)
    case = clone_case(layer.case)
    case.update(q=q.view(7, layer.h, 128), k=k.view(7, layer.h, 128),
                v=v.view(7, layer.hv, 128), a=a, b=b)
    # Construct actual high/low representations for history rather than using
    # independently sampled low parts with the magnitude of the high parts.
    k_precise = case["k_cache"].float()
    k_precise /= (k_precise.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    case["k_cache"] = k_precise.bfloat16()
    case["rawk_cache"] = (k_precise - case["k_cache"].float()).bfloat16()
    d_precise = case["d_cache"].float() + case["rawv_cache"].float() * 0.001
    case["d_cache"] = d_precise.bfloat16()
    case["rawv_cache"] = (d_precise - case["d_cache"].float()).bfloat16()
    torch.cuda.synchronize()
    return case


def make_params(lib, case):
    p = Params()
    lib.gdn_replay_probe_params_init(C.byref(p))
    mapping = {
        "a_log": "A_log", "checkpoint": "checkpoint_state",
        "state_indices": "ssm_state_indices", "replay_indices": "replay_indices",
    }
    for field in POINTER_FIELDS:
        value = case[mapping.get(field, field)]
        setattr(p, field, 0 if value is None else value.data_ptr())
    p.h, p.hv = case["q"].shape[1], case["v"].shape[1]
    p.gates_fp32 = case["a"].dtype == torch.float32
    p.dt_bias_fp32 = case["dt_bias"].dtype == torch.float32
    p.state_indices_int64 = case["ssm_state_indices"].dtype == torch.int64
    p.replay_indices_int64 = case["replay_indices"].dtype == torch.int64
    p.storage = 1
    p.scale = 128 ** -0.5
    stride_mapping = {
        "q_t": "q", "k_t": "k", "v_t": "v", "a_t": "a", "b_t": "b", "o_t": "out",
        "state_slot": "checkpoint_state", "d_slot": "d_cache", "k_slot": "k_cache",
        "g_slot": "g_cache", "rawv_slot": "rawv_cache", "rawk_slot": "rawk_cache",
        "beta_slot": "g_cache", "qsl": "query_start_loc",
        "state_indices": "ssm_state_indices", "replay_indices": "replay_indices",
    }
    for field, key in stride_mapping.items():
        setattr(p, "stride_" + field, case[key].stride(0))
    return p


def launch(lib, params, phase, status, role):
    rc = lib.gdn_replay_probe_launch(
        C.byref(params), 1, role, phase.data_ptr(), status.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    assert rc == 0, ("launch", role, rc)


def tensor_bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


def snapshots(case):
    return {key: tensor_bytes(value).clone()
            for key, value in case.items() if isinstance(value, torch.Tensor)}


def byte_equal(a, b):
    return {key: bool(torch.equal(tensor_bytes(value), tensor_bytes(b[key])))
            for key, value in a.items() if isinstance(value, torch.Tensor)}


def active_masks(case):
    masks = {key: torch.zeros_like(case[key], dtype=torch.bool)
             for key in ("out", "d_cache", "rawv_cache", "k_cache", "rawk_cache", "g_cache")}
    masks["out"].fill_(True)
    replay = int(case["replay_indices"][0])
    wp, base = int(case["write_pos"][replay]), int(case["cache_base"][replay])
    for token in range(case["max_spec_len"]):
        if wp + token < 16:
            physical = (base + wp + token) & 15
            for name in ("d_cache", "rawv_cache", "k_cache", "rawk_cache", "g_cache"):
                masks[name][replay, :, physical] = True
    return masks


def untouched_checks(case, before, masks):
    result = {}
    for key, value in case.items():
        if not isinstance(value, torch.Tensor):
            continue
        original = before[key].view(value.dtype).reshape(value.shape)
        if key in masks:
            result[key] = bool(torch.equal(tensor_bytes(value[~masks[key]]),
                                          tensor_bytes(original[~masks[key]])))
        else:
            result[key] = bool(torch.equal(tensor_bytes(value), before[key]))
    return result


def error(actual, expected):
    a, e = actual.double(), expected.double()
    diff = a - e
    rows = diff.reshape(-1, diff.shape[-1])
    expected_rows = e.reshape_as(rows)
    return dict(
        finite=bool(torch.isfinite(a).all()),
        rel_l2=float(diff.norm() / e.norm().clamp_min(1e-30)),
        mean_row_rel_l2=float((rows.norm(dim=-1) / expected_rows.norm(dim=-1).clamp_min(1e-30)).mean()),
        max_abs=float(diff.abs().max()),
    )


def delta_values(case):
    replay = int(case["replay_indices"][0])
    wp, base = int(case["write_pos"][replay]), int(case["cache_base"][replay])
    rows = []
    for token in range(case["max_spec_len"]):
        if wp + token < 16:
            physical = (base + wp + token) & 15
            rows.append(case["d_cache"][replay, :, physical].double() +
                        case["rawv_cache"][replay, :, physical].double())
    return torch.stack(rows) if rows else None


def timing(lib, attributes, args):
    assert args.layers == 45 and args.eviction_mib >= 256
    assert 80 <= attributes[1]["registers"] <= 128
    lib.gdn_replay_probe_evict.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p]
    lib.gdn_replay_probe_evict.restype = C.c_int
    lib.gdn_replay_probe_l2_bytes.argtypes = [C.POINTER(C.c_uint64)]
    lib.gdn_replay_probe_l2_bytes.restype = C.c_int
    l2_bytes = C.c_uint64()
    assert lib.gdn_replay_probe_l2_bytes(C.byref(l2_bytes)) == 0
    seed = int(args.seeds)
    prepared = [prepare(args.tp, seed + layer) for layer in range(args.layers)]
    arms = {name: [clone_case(case) for case in prepared]
            for name in ("A", "D", "R", "N")}
    params = {name: [make_params(lib, case) for case in cases]
              for name, cases in arms.items() if name in ("D", "R")}
    phase = torch.zeros(1, device="cuda", dtype=torch.int32)
    eviction = torch.zeros(args.eviction_mib << 20, device="cuda", dtype=torch.uint8)

    def run(name, index):
        if name in ("A", "N"):
            stock.gdn_replayssm_spec_decode(
                **arms[name][index], block_v=32, num_warps=2, pdl=True)
        else:
            rc = lib.gdn_replay_probe_launch(
                C.byref(params[name][index]), 1, int(name == "R"), phase.data_ptr(),
                None, torch.cuda.current_stream().cuda_stream)
            assert rc == 0, (name, index, rc)

    for name in arms:
        for index in range(args.layers):
            run(name, index)
    torch.cuda.synchronize()
    stock_null = [byte_equal(a, n) for a, n in zip(arms["A"], arms["N"])]
    role_null = [byte_equal(d, r) for d, r in zip(arms["D"], arms["R"])]
    assert all(all(row.values()) for row in stock_null)
    assert all(all(row.values()) for row in role_null)
    emit(kind="timing_setup", seed=seed, layers=args.layers,
         eviction_bytes=eviction.numel(), l2_bytes=l2_bytes.value,
         allocated_case_bytes_per_arm=sum(
             value.numel() * value.element_size()
             for case in arms["A"] for value in case.values()
             if isinstance(value, torch.Tensor)),
         read_checkpoint_bytes_per_arm=args.layers * (64 // args.tp) * 128 * 128 * 2,
         stock_null_all45=True, dedicated_role_all45=True,
         mode="single stage; 256 MiB or larger eviction before every event pair; no full fusion claim",
         method="one graph/arm, each bank: eviction -> external start -> stage -> external stop",
         sample_statistic="mean of 45 independent-bank event durations; median of 9 samples")
    graphs, events = {}, {}
    for name in arms:
        events[name] = [(torch.cuda.Event(enable_timing=True, external=True),
                         torch.cuda.Event(enable_timing=True, external=True))
                        for _ in range(args.layers)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index, (start, end) in enumerate(events[name]):
                rc = lib.gdn_replay_probe_evict(
                    eviction.data_ptr(), eviction.numel(),
                    torch.cuda.current_stream().cuda_stream)
                assert rc == 0
                start.record()
                run(name, index)
                end.record()
        graphs[name] = graph
    for graph in graphs.values():
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()

    def clocks():
        return subprocess.check_output([
            "nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,power.draw",
            "--format=csv,noheader,nounits"], text=True).strip()

    for mirror, order in enumerate((("A", "N", "D", "R"), ("R", "D", "N", "A"))):
        for name in order:
            before = clocks()
            samples, per_bank = [], []
            for _ in range(9):
                graphs[name].replay()
                torch.cuda.synchronize()
                durations = [start.elapsed_time(end) * 1000
                             for start, end in events[name]]
                samples.append(statistics.mean(durations))
                per_bank.append(durations)
            emit(kind="timing", arm=name, seed=seed, mirror=mirror,
                 median_us=statistics.median(samples), samples_us=samples,
                 range_us=[min(samples), max(samples)], per_bank_us=per_bank,
                 clock_sm_mhz_power_w_before=before, clock_sm_mhz_power_w_after=clocks())
    emit(kind="timing_complete", seed=seed, source_math_unchanged=True,
         full_fused_condition_a="NOT_MEASURED", serving_claim="NONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=8)
    parser.add_argument("--seeds", default="341,342,343")
    parser.add_argument("--histories", default="0,1,7,9")
    parser.add_argument("--bias-dtypes", default="float32,bfloat16")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--layers", type=int, default=45)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--compare-library")
    args = parser.parse_args()
    lib, attributes = bind(args.library)
    compare_lib = bind(args.compare_library)[0] if args.compare_library else None
    files = [Path(__file__), Path(__file__).with_suffix(".cu"),
             Path(__file__).with_name("gdn_replayssm_stage.cuh"),
             Path(inspect.getsourcefile(stock)), Path(inspect.getsourcefile(reference)),
             Path(inspect.getsourcefile(stock.gdn_replayssm_spec_decode))]
    emit(kind="metadata", args=vars(args), torch=torch.__version__, cuda=torch.version.cuda,
         capability=torch.cuda.get_device_capability(),
         sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         mode="operator correctness; synthetic FP8 projection followed by stock production convolution",
         numerics="numeric; diagnostics are not an accuracy or serving qualification",
         timing=("cold-bank standalone timing diagnostic requested" if args.timing
                 else "not measured; independent cold banks required before a speed claim"))
    if args.timing:
        timing(lib, attributes, args)
        return
    role_admitted = 80 <= attributes[1]["registers"] <= 128
    phase = torch.zeros(1, device="cuda", dtype=torch.int32)
    status = torch.zeros(2, device="cuda", dtype=torch.uint64)
    expected_tiles = 4 * (64 // args.tp)
    failures = 0
    numerical_misses = 0
    for seed in map(int, args.seeds.split(",")):
        prepared = prepare(args.tp, seed)
        for history in map(int, args.histories.split(",")):
            for bias_name in args.bias_dtypes.split(","):
                initial = clone_case(prepared)
                initial["write_pos"][int(initial["replay_indices"][0])] = history
                bias_dtype = getattr(torch, bias_name)
                initial["dt_bias"] = torch.linspace(-0.2, 0.2, 64 // args.tp,
                                                    device="cuda").to(bias_dtype)
                expected_out, expected_delta = reference(initial)
                stock_a, stock_n = clone_case(initial), clone_case(initial)
                stock.gdn_replayssm_spec_decode(**stock_a, block_v=32, num_warps=2, pdl=True)
                stock.gdn_replayssm_spec_decode(**stock_n, block_v=32, num_warps=2, pdl=True)
                stock_null = byte_equal(stock_a, stock_n)
                assert all(stock_null.values()), stock_null
                before = snapshots(initial)
                masks = active_masks(initial)
                operator_reference = None
                if compare_lib is not None:
                    operator_reference = clone_case(initial)
                    reference_params = make_params(compare_lib, operator_reference)
                    status.zero_()
                    phase.zero_()
                    launch(compare_lib, reference_params, phase, status, 0)
                    torch.cuda.synchronize()
                    assert status.cpu().tolist() == [expected_tiles, expected_tiles]
                variants = {}
                dedicated_case = None
                for role in (0, 1) if role_admitted else (0,):
                    candidate = clone_case(initial)
                    params = make_params(lib, candidate)
                    status.zero_()
                    phase.zero_()
                    launch(lib, params, phase, status, role)
                    torch.cuda.synchronize()
                    counters = status.cpu().tolist()
                    assert counters == [expected_tiles, expected_tiles], counters
                    checks = untouched_checks(candidate, before, masks)
                    comparisons = byte_equal(candidate, stock_a)
                    out_error = error(candidate["out"], expected_out)
                    floor_out = error(stock_a["out"], expected_out)
                    observed_delta, floor_delta = delta_values(candidate), delta_values(stock_a)
                    count = 0 if observed_delta is None else observed_delta.shape[0]
                    measured = dict(out=out_error)
                    floor = dict(out=floor_out)
                    if count:
                        measured["delta"] = error(observed_delta, expected_delta[:count])
                        floor["delta"] = error(floor_delta, expected_delta[:count])
                    ratios = {key: {metric: measured[key][metric] / max(floor[key][metric], 1e-30)
                                    for metric in ("rel_l2", "mean_row_rel_l2", "max_abs")}
                              for key in measured}
                    numeric_floor_ok = all(
                        values["rel_l2"] <= 1.02 and values["mean_row_rel_l2"] <= 1.02
                        and values["max_abs"] <= 1.10 for values in ratios.values())
                    numerical_misses += not numeric_floor_ok
                    finite = all(values["finite"] for values in measured.values())
                    passed = all(checks.values()) and finite
                    reference_bytes = None
                    if operator_reference is not None:
                        reference_bytes = byte_equal(candidate, operator_reference)
                        passed = passed and all(reference_bytes.values())
                    failures += not passed
                    variants[role] = dict(counters=counters, untouched=checks,
                                          stock_bytes=comparisons, error=measured,
                                          stock_floor=floor, ratios=ratios,
                                          numeric_floor_ok=numeric_floor_ok,
                                          plumbing_pass=passed)
                    if reference_bytes is not None:
                        variants[role]["reference_bytes"] = reference_bytes
                    if role == 0:
                        dedicated_case = candidate
                    else:
                        same = byte_equal(candidate, dedicated_case)
                        variants[role]["dedicated_bytes"] = same
                        assert all(same.values()), same
                emit(kind="cell", tp=args.tp, seed=seed, history=history,
                     cache_base=15, bias_dtype=bias_name, variants=variants,
                     stock_null=stock_null, role_pool_admitted=role_admitted)

                # Runtime phase and capture must keep their device pointers live.
                candidate = clone_case(initial)
                params = make_params(lib, candidate)
                status.zero_()
                phase.zero_()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(lib, params, phase, status, 0)
                graph.replay()
                torch.cuda.synchronize()
                assert status.cpu().tolist() == [expected_tiles, expected_tiles]
                eager_bytes = byte_equal(candidate, dedicated_case)
                assert all(eager_bytes.values()), eager_bytes
                phase.fill_(1)
                status.zero_()
                frozen = snapshots(candidate)
                graph.replay()
                torch.cuda.synchronize()
                assert status.cpu().tolist() == [0, 0]
                assert all(torch.equal(tensor_bytes(candidate[key]), value)
                           for key, value in frozen.items())
                phase.zero_()
                corrupted = candidate["out"].clone()
                corrupted.view(torch.uint8).reshape(-1)[0] ^= 1
                counts = torch.empty(2, device="cuda", dtype=torch.uint64)
                rc = lib.gdn_replay_probe_compare_bytes(
                    candidate["out"].data_ptr(), corrupted.data_ptr(),
                    corrupted.numel() * corrupted.element_size(), counts.data_ptr(),
                    torch.cuda.current_stream().cuda_stream)
                assert rc == 0
                torch.cuda.synchronize()
                assert counts.cpu().tolist() == [1, 0]
                emit(kind="controls", seed=seed, history=history, bias_dtype=bias_name,
                     graph_matches_eager=True, runtime_phase_negative_caught=True,
                     planted_byte_negative_caught=True)
    emit(kind="summary", plumbing_failures=failures, numerical_floor_misses=numerical_misses,
         role_pool_admitted=role_admitted, accuracy_gate="PENDING", timing="NOT_RUN")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
