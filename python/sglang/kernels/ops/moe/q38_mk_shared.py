# SPDX-License-Identifier: Apache-2.0
"""Default-off shared+SF MKM experiment. CPU AOT only; isolated local boundary.

No service dispatch is installed until the same-card fixture and performance gates pass.
Native shared BF16 K32 partials change FP32 accumulation order; qualification is numeric.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

import torch
from sglang.kernels.ops.moe import q38_mk_q35 as mk

EXTRA = ('hidden','shared_wgu','shared_wd','shared_gate','shared_part','completed',
         'combined','debug_gu','debug_act','debug_down','debug_gate','debug_routed')
class Params(C.Structure):
    _fields_ = mk.Params._fields_[:-1] + [(n,C.c_void_p) for n in EXTRA] + [('num_tokens',C.c_int)]


def load_bundle(root):
    root=Path(root)
    assert os.environ.get('SGLANG_Q38C_MK_SHARED','0')=='1'
    manifest=json.loads((root/'manifest.json').read_text())
    source=Path(__file__).resolve().parents[2]/'jit/csrc/moe/q38_mk_shared_tp4.cu'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']
    assert manifest['grid']==150 and manifest['shared']==mk.SHARED and manifest['pdl_trigger']==0
    cubin=root/'shared.cubin';assert hashlib.sha256(cubin.read_bytes()).hexdigest()==manifest['sha256']['shared.cubin']
    return mk.jit.load(cubin.read_bytes(),'q38_mkm_shared_tp4',(256,1,1),mk.SHARED)


class State:
    def __init__(self,m,device,debug=False):
        assert m in (1,4)
        kw=dict(device=device)
        self.rows=torch.empty((m,10,2560),dtype=torch.bfloat16,**kw)
        self.weights=torch.empty((m,10),dtype=torch.bfloat16,**kw)
        self.ids=torch.empty((m,10),dtype=torch.int32,**kw)
        self.partials=torch.empty((5,m,2560),dtype=torch.float32,**kw)
        self.completed=torch.zeros(1,dtype=torch.uint64,**kw)
        self.output=torch.empty((m,2560),dtype=torch.bfloat16,**kw)
        self.debug={name:torch.empty(shape,dtype=dtype,**kw) for name,shape,dtype in [
            ('gu',(m,320),torch.bfloat16),('act',(m,160),torch.bfloat16),
            ('down',(m,2560),torch.bfloat16),('gate',(m,),torch.float32),
            ('routed',(m,2560),torch.bfloat16)]} if debug else {}

    def run(self,kernel,maps,logits,xsf,hidden,layer,shared):
        tensors=(logits,self.weights,self.ids,xsf,layer['gemm1_weights_scale'],
          layer['gemm2_weights_scale'],layer['output1_scale_gate_scalar'],
          layer['output1_scale_scalar'],layer['output2_scale_scalar'],self.rows)
        pointers=[t.data_ptr() for t in tensors]+[0]
        pointers += [hidden.data_ptr(),shared['wgu'].data_ptr(),shared['wd'].data_ptr(),
                     shared['gate'].data_ptr(),self.partials.data_ptr(),self.completed.data_ptr(),self.output.data_ptr()]
        pointers += [self.debug[n].data_ptr() if n in self.debug else 0 for n in ['gu','act','down','gate','routed']]
        params=Params(*pointers,hidden.shape[0])
        mk.jit.launch(kernel,(150,1,1),[*maps,params],torch.cuda.current_stream().cuda_stream,cluster=(2,1,1))
        return self.output
