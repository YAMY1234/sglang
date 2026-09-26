"""Exercise actual model forward with the production fused projection enabled."""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch
import torch
import sglang.srt.models.qwen3_5 as model
import sglang.kernels.ops.attention.fla.layernorm_gated as norm

class Norm:
    norm_before_gate=True;group_size=None;bias=None;activation='sigmoid';eps=1e-6
    def __init__(self):self.weight=torch.ones(16);self.calls=0
    def __call__(self,x,z):self.calls+=1;return x+3

proof=[]
for projection in (False,True):
    for enabled in (False,True):
        for honor in (False,True):
            for decode in (False,True):
                for batch in (1,3):
                    for rows in (1,4):
                        n=Norm();calls=[]
                        q=torch.zeros(batch,2,16);v=torch.zeros(batch,6,16);a=torch.zeros(batch,6)
                        eligible=enabled and decode and batch==1
                        def attention(forward_batch,**kw):
                            context=kw.get('decode_norm');calls.append(context)
                            if eligible:
                                assert context is not None and context[3]==rows
                                assert (context[0] is None)==(projection and decode)
                            else:assert context is None
                            raw=torch.zeros(1,batch,6,16)
                            if projection and decode:
                                assert isinstance(kw['mixed_qkv'],tuple)
                                return (raw+3,v,True) if eligible and honor else (raw,v)
                            return (raw+3,True) if eligible and honor else raw
                        instance=SimpleNamespace(num_v_heads=6,num_k_heads=2,norm=n,attn=attention,
                            _forward_input_proj=lambda hidden:(q,a),
                            fix_query_key_value_ordering=lambda x,y:(q,q,v,v,a,a),
                            out_proj=lambda x:(x,None))
                        mode=SimpleNamespace(is_decode=lambda:decode)
                        with patch.dict(os.environ,{'SGLANG_GDN_FACTORED_STEP_NORM':str(int(enabled))}), \
                             patch.object(model,'_GDN_FUSED_QKVZBA_RATIOS',()), \
                             patch.object(model,'_gdn_decode_fused_proj_conv',projection), \
                             patch.object(norm,'calc_rows_per_block',lambda m,d:rows):
                            output=model.Qwen3_5GatedDeltaNet.forward(instance,torch.zeros(batch,96),SimpleNamespace(forward_mode=mode))
                        assert output.shape==(batch,96) and torch.equal(output,torch.full_like(output,3))
                        assert n.calls==(0 if eligible and honor else 1) and len(calls)==1
                        proof.append(dict(projection=projection,enabled=enabled,honor=honor,
                            decode=decode,batch=batch,rows=rows,norm_calls=n.calls))
print(json.dumps(dict(complete=True,passed=True,device='CPU',cases=proof)),flush=True)
