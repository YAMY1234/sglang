"""CPU-only decision for the complete September 30 Lightning variance cell.

Two repeats estimate a noise scale, not absence of bias. The window mean +/-
2 SEM check is the descriptive check used in docs/159, not a confidence interval.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

NAMES = ("reference1", "reference2", "duet", "duet2", "stock1", "stock2", "off", "off2")


def noise_multiple(difference, noise):
    return difference / noise if noise else (0.0 if difference == 0 else None)


def rms(values):
    return math.sqrt(sum(x * x for x in values) / len(values))


def _pair(means, windows, candidate, reference):
    a, b = candidate
    c, d = reference
    delta = (means[a] + means[b]) / 2 - (means[c] + means[d]) / 2
    ref_noise = abs(means[c] - means[d])
    own_noise = abs(means[a] - means[b])
    differences = [
        (x + y - z - t) / 2
        for x, y, z, t in zip(windows[a], windows[b], windows[c], windows[d])
    ]
    ref_differences = [y - x for x, y in zip(windows[c], windows[d])]
    own_differences = [y - x for x, y in zip(windows[a], windows[b])]
    multiple = noise_multiple(abs(delta), ref_noise)
    window_multiple = noise_multiple(rms(differences), rms(ref_differences))
    mean = statistics.mean(differences)
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    interval = [mean - 2 * sem, mean + 2 * sem]
    positive = sum(x > 0 for x in differences)
    negative = sum(x < 0 for x in differences)
    # All-zero is valid. A same-sign displacement is not round-to-round noise.
    no_offset = interval[0] <= 0 <= interval[1] and (
        (positive > 0 and negative > 0) or all(x == 0 for x in differences)
    )
    scale_pass = all(x is not None and x < 10 for x in (multiple, window_multiple))
    return {
        "signed_difference": delta,
        "abs_difference": abs(delta),
        "reference_self_difference": ref_noise,
        "candidate_self_difference": own_noise,
        "over_reference_noise": multiple,
        "over_candidate_noise": noise_multiple(abs(delta), own_noise),
        "over_joint_noise": noise_multiple(abs(delta), max(ref_noise, own_noise)),
        "window_rms": rms(differences),
        "reference_window_self_rms": rms(ref_differences),
        "candidate_window_self_rms": rms(own_differences),
        "window_rms_over_reference_noise": window_multiple,
        "window_positive": positive,
        "window_negative": negative,
        "descriptive_mean_plus_minus_2sem": interval,
        "no_systematic_offset_observed": no_offset,
        "single_digit_multiples": scale_pass,
        "pass": scale_pass and no_offset,
    }


def assess(cells):
    """Validate all eight complete arms, then apply the variance decision."""
    if any(name not in cells for name in NAMES):
        raise ValueError(
            "two complete rounds of reference, DUET, stock and off are required"
        )
    baseline = cells["reference1"]["windows"]
    flattened, windows = {}, {}
    for name in NAMES:
        cell = cells[name]
        if not cell["complete"] or len(cell["windows"]) != 32:
            raise ValueError(f"{name}: incomplete 32-window cell")
        for row, expected in zip(cell["windows"], baseline):
            if len(row["losses"]) != 256 or len(row["target"]) != 256:
                raise ValueError(f"{name}: incomplete continuation")
            if (row["id"], row["prompt_sha256"], row["target"]) != (
                expected["id"],
                expected["prompt_sha256"],
                expected["target"],
            ):
                raise ValueError(f"{name}: window/token identity mismatch")
        flattened[name] = [x for row in cell["windows"] for x in row["losses"]]
        if not all(math.isfinite(x) for x in flattened[name]):
            raise ValueError(f"{name}: non-finite loss")
        windows[name] = [sum(row["losses"]) / 256 for row in cell["windows"]]
    if len({row["id"] for row in baseline}) != 32:
        raise ValueError("window IDs must be unique")
    means = {name: sum(values) / len(values) for name, values in flattened.items()}
    duet = _pair(means, windows, ("duet", "duet2"), ("reference1", "reference2"))
    off = _pair(means, windows, ("off", "off2"), ("stock1", "stock2"))
    stock_noise = [abs(a - b) for a, b in zip(flattened["stock1"], flattened["stock2"])]
    off_delta = [abs(a - b) for a, b in zip(flattened["stock1"], flattened["off"])]
    off2_delta = [abs(a - b) for a, b in zip(flattened["stock1"], flattened["off2"])]
    per_window = []
    for i, row in enumerate(baseline):
        v = {name: windows[name][i] for name in NAMES}
        per_window.append(
            {
                "id": row["id"],
                "reference_round_difference": v["reference2"] - v["reference1"],
                "duet_round_difference": v["duet2"] - v["duet"],
                "stock_round_difference": v["stock2"] - v["stock1"],
                "off_round_difference": v["off2"] - v["off"],
                "duet_reference_difference": (
                    v["duet"] + v["duet2"] - v["reference1"] - v["reference2"]
                )
                / 2,
                "off_stock_difference": (
                    v["off"] + v["off2"] - v["stock1"] - v["stock2"]
                )
                / 2,
            }
        )
    passed = duet["pass"] and off["pass"]
    return {
        "status": "pass" if passed else "fail",
        "decision": "round-to-round-variance",
        "protocol": "round-to-round-variance-20260930",
        "groups": 8,
        "windows": 32,
        "continuation_tokens_per_group": 8192,
        "means": means,
        "round_mean": {
            "reference": (means["reference1"] + means["reference2"]) / 2,
            "duet": (means["duet"] + means["duet2"]) / 2,
            "stock": (means["stock1"] + means["stock2"]) / 2,
            "off": (means["off"] + means["off2"]) / 2,
        },
        "duet_reference": duet,
        "off_stock": off,
        "duet_reference_signed_difference": duet["signed_difference"],
        "duet_reference_abs_difference": duet["abs_difference"],
        "reference_self_difference": duet["reference_self_difference"],
        "duet_self_difference": duet["candidate_self_difference"],
        "duet_reference_over_reference_noise": duet["over_reference_noise"],
        "duet_reference_over_joint_noise": duet["over_joint_noise"],
        "off_stock_signed_difference": off["signed_difference"],
        "off_stock_abs_difference": off["abs_difference"],
        "stock_self_difference": off["reference_self_difference"],
        "off_self_difference": off["candidate_self_difference"],
        "off_stock_over_stock_noise": off["over_reference_noise"],
        "off_stock_over_joint_noise": off["over_joint_noise"],
        "zero_noise_nonzero_difference": {
            "duet_reference": duet["over_reference_noise"] is None,
            "off_stock": off["over_reference_noise"] is None,
        },
        "legacy_record_only": {
            "duet1_reference1_abs_difference": abs(means["duet"] - means["reference1"]),
            "absolute_threshold": 0.002,
            "abs_round_mean_difference_le_0p002": duet["abs_difference"] <= 0.002,
            "flags_off_bitwise": max(off_delta) == 0,
            "off2_flags_off_bitwise": max(off2_delta) == 0,
            "flags_off_max_token_delta": max(off_delta),
            "off2_flags_off_max_token_delta": max(off2_delta),
            "stock_repeat_max_token_noise": max(stock_noise),
            "flags_off_same_noise": all(d <= n for d, n in zip(off_delta, stock_noise)),
        },
        "per_window": per_window,
        "interpretation_limit": "Two repeats do not prove absence of bias; window mean +/- 2SEM is descriptive, not a calibrated confidence interval.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory containing the eight measured JSON cells",
    )
    parser.add_argument("--report", default="variance-decision.json")
    args = parser.parse_args()
    result = assess(
        {name: json.loads((args.out / (name + ".json")).read_text()) for name in NAMES}
    )
    smoke = json.loads((args.out / "accuracy-first-options.json").read_text())
    result["accuracy_first_options"] = smoke["status"]
    result["accuracy_first_rounds"] = smoke.get("rounds_complete", 0)
    if smoke["status"] != "pass" or result["accuracy_first_rounds"] < 2:
        result["status"] = "fail"
    (args.out / args.report).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, allow_nan=False))
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
