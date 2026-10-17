# docs/219 §8 ab_measure v4 register file (WS-3, docs/223 §11, lead 23:32). Executed once per TP process with `register`
# in globals. Needs the full-N AGG route (SGLANG_GDN_AGG_FULLN_PREFILL=1) on a tree with the prompt-end checkpoint fix
# (fork ssm219-fulln-ckpt-20261017 / v8 / v9). Single variable: A = SGLANG_GDN_FULLN_CKPT_N_MINUS_1=0 (old full-N extent),
# B = 1 (the fix); everything else unchanged. Host-only, idempotent, no allocation: fullstack_policy re-reads the env on
# every extend preparation. Reading note: the label decides where THIS extend donates its radix checkpoint, so its
# effect lands on the conversation's next turn (an A-labelled P % 64 == 0 prompt leaves an unreachable checkpoint and
# the next turn recomputes up to the previous one). Read B - A token-binned / on equal-cache hit extends and list the
# shortfall extends separately (docs/223 §11.2).
import os


def _apply_a():
    os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "0"


def _apply_b():
    os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "1"


register("fulln_ckpt", _apply_a, _apply_b, scope="prefill")  # noqa: F821
