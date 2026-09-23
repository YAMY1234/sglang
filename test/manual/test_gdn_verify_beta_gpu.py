"""Compare true stock packed decode with verify on identical BF16 inputs."""
import json

import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as packed
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as verify


def delta(a, b):
    d = (a.float()-b.float()).abs()
    return dict(equal=torch.equal(a, b), different=int(torch.count_nonzero(d)), max=float(d.max()))


def main():
    torch.manual_seed(428)
    B, D, H, HV, K, V = 2, 4, 8, 24, 128, 128
    device = "cuda"
    slots = torch.tensor([1, 3], dtype=torch.int32, device=device)
    results = []
    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        mixed = torch.randn(B, D, 2*H*K+HV*V, dtype=torch.bfloat16, device=device)
        a, b = [torch.randn(B, D, HV, dtype=torch.bfloat16, device=device) for _ in range(2)]
        A_log, dt = [torch.randn(HV, dtype=torch.float32, device=device) for _ in range(2)]
        initial = torch.randn(5, HV, V, K, dtype=torch.float32, device=device)
        sequential = initial.clone()
        outputs, checkpoints = [], []
        for step in range(D):
            out = torch.empty(B, 1, HV, V, dtype=torch.bfloat16, device=device)
            packed(mixed_qkv=mixed[:, step].contiguous(), a=a[:, step].contiguous(), b=b[:, step].contiguous(),
                   A_log=A_log, dt_bias=dt, scale=K**-.5, initial_state=sequential,
                   out=out, ssm_state_indices=slots, use_qk_l2norm_in_kernel=True)
            outputs.append(out[:, 0].clone()); checkpoints.append(sequential[slots.long()].clone())
        expected_out = torch.stack(outputs, 1).reshape(1, B*D, HV, V)
        expected_state = torch.stack(checkpoints, 1)
        q, k, v = torch.split(mixed.reshape(B*D, -1), [H*K, H*K, HV*V], dim=-1)
        for tree in (False, True):
            report = dict(seed=seed, tree=tree)
            for rounded in (False, True):
                state = initial.clone()
                cache = torch.zeros(B, D, HV, V, K, dtype=torch.float32, device=device)
                out = verify(A_log=A_log, dt_bias=dt, a=a.reshape(B*D, HV), b=b.reshape(B*D, HV),
                    q=q.reshape(1, B*D, H, K), k=k.reshape(1, B*D, H, K), v=v.reshape(1, B*D, HV, V),
                    softplus_beta=1., softplus_threshold=20., use_qk_l2norm_in_kernel=True,
                    initial_state_source=state, initial_state_indices=slots,
                    cu_seqlens=torch.arange(0, (B+1)*D, D, dtype=torch.int32, device=device),
                    disable_state_update=True, intermediate_states_buffer=cache,
                    intermediate_state_indices=torch.arange(B, dtype=torch.int32, device=device), cache_steps=D,
                    retrieve_parent_token=torch.tensor([[-1,0,1,2]]*B, dtype=torch.int32, device=device) if tree else None,
                    round_beta_to_input_dtype=rounded)
                assert torch.equal(state, initial), "verify published candidate state"
                report["rounded" if rounded else "original"] = dict(output=delta(out, expected_out), state=delta(cache, expected_state))
            results.append(report)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), cases=results)), flush=True)


if __name__ == "__main__":
    main()
