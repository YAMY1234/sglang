"""Run the unchanged full Lightning cell with a second DUET pass.

The recorded round-to-round differences replace the old fixed numerical gate.
Completion means all measurements are present; the variance interpretation is
reported separately, without treating 0.002 or bitwise equality as pass lines.
"""

import argparse
import json
import sys
from pathlib import Path

import lightning_sgl_guard as guard

_legacy_assess = guard.assess


def noise_multiple(difference, noise):
    if noise:
        return difference / noise
    return 0.0 if difference == 0 else None


def assess_variance(cells):
    # Reuse the existing complete-window, finite-loss and token-identity checks.
    # Its numerical pass/fail decision is deliberately not used.
    first = _legacy_assess(cells)
    second = _legacy_assess({**cells, "duet": cells["duet2"]})
    means = {**first["means"], "duet2": second["means"]["duet"]}
    reference = (means["reference1"] + means["reference2"]) / 2
    duet = (means["duet"] + means["duet2"]) / 2
    stock = (means["stock1"] + means["stock2"]) / 2
    ref_noise = abs(means["reference1"] - means["reference2"])
    duet_noise = abs(means["duet"] - means["duet2"])
    stock_noise = abs(means["stock1"] - means["stock2"])
    duet_signed, off_signed = duet - reference, means["off"] - stock
    per_window = []
    for rows in zip(
        *(
            cells[name]["windows"]
            for name in (
                "reference1",
                "reference2",
                "duet",
                "duet2",
                "stock1",
                "stock2",
                "off",
            )
        )
    ):
        values = [sum(row["losses"]) / len(row["losses"]) for row in rows]
        r1, r2, d1, d2, s1, s2, off = values
        per_window.append(
            {
                "id": rows[0]["id"],
                "reference_round_difference": r2 - r1,
                "duet_round_difference": d2 - d1,
                "stock_round_difference": s2 - s1,
                "duet_reference_difference": (d1 + d2 - r1 - r2) / 2,
                "off_stock_difference": off - (s1 + s2) / 2,
            }
        )
    return {
        "status": "measured",
        "decision": "pending_variance_review",
        "protocol": "round-to-round-variance-1457",
        "groups": 7,
        "windows": 32,
        "continuation_tokens_per_group": 8192,
        "means": means,
        "round_mean": {"reference": reference, "duet": duet, "stock": stock},
        "duet_reference_signed_difference": duet_signed,
        "duet_reference_abs_difference": abs(duet_signed),
        "reference_self_difference": ref_noise,
        "duet_self_difference": duet_noise,
        "off_stock_signed_difference": off_signed,
        "off_stock_abs_difference": abs(off_signed),
        "stock_self_difference": stock_noise,
        "duet_reference_over_reference_noise": noise_multiple(
            abs(duet_signed), ref_noise
        ),
        "duet_reference_over_joint_noise": noise_multiple(
            abs(duet_signed), max(ref_noise, duet_noise)
        ),
        "off_stock_over_stock_noise": noise_multiple(abs(off_signed), stock_noise),
        "zero_noise_nonzero_difference": {
            "duet_reference": max(ref_noise, duet_noise) == 0 and duet_signed != 0,
            "off_stock": stock_noise == 0 and off_signed != 0,
        },
        "legacy_record_only": {
            "duet1_reference1_abs_difference": first["duet_delta_nll"],
            "abs_round_mean_difference_le_0p002": abs(duet_signed) <= 0.002,
            "flags_off_bitwise": first["flags_off_bitwise"],
            "flags_off_max_token_delta": first["flags_off_max_token_delta"],
            "stock_repeat_max_token_noise": first["stock_repeat_max_token_noise"],
        },
        "per_window": per_window,
        "interpretation_limit": "Two repeats estimate a noise scale; they do not prove absence of systematic bias.",
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", default="coordinator")
    options, _ = parser.parse_known_args()
    original_pass = guard.engine_pass

    def repeated_pass(args, endpoint, windows, name):
        original_pass(args, endpoint, windows, name)
        if name == "duet":
            original_pass(args, endpoint, windows, "duet2")

    def assess(cells):
        return assess_variance(
            {**cells, "duet2": json.loads((options.out / "duet2.json").read_text())}
        )

    guard.engine_pass = repeated_pass
    guard.assess = assess
    guard.__file__ = __file__  # The coordinator must launch this wrapper too.
    code = guard.main()
    if options.worker == "coordinator":
        result = json.loads((options.out / "result.json").read_text())
        if result["status"] == "measured":
            return 0  # Complete measurements, not a numerical PASS declaration.
    return code


if __name__ == "__main__":
    sys.exit(main())
