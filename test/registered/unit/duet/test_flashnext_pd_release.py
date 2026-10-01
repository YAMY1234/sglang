"""CPU regression for native release P/D policy and transient emitter inputs."""
import importlib.util
import os
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from sglang.srt.models.flash_next_duet import pd_shallow as native
from sglang.srt.models.flash_next_duet.latent import FlashNextLatentCodec
from sglang.srt.models.flash_next_duet.serving import embedding_streams


def owner(rank=0):
    return NS(fullstack=dict(duet_spec={}, gdn_rank=rank, gdn_every=rank,
                            gdn_state=f'rank:{rank}' if rank else 'dense',
                            latent='on', prefill_layer_trim=True,
                            prefill_saving_policy='kv-and-ssm'),
              fullstack_code=True, fullstack_v3_latent=False,
              fullstack_final=True, pd_shallow_role='prefill',
              config=NS(hc_count=4, hidden_size=16, vocab_size=32))


def codec():
    torch.manual_seed(1634)
    spec=dict(latent_rank=16, latent_spikes=4, latent_id_side=True)
    value=FlashNextLatentCodec(device='cpu', spec=spec, width=64)
    for name, shape in [('E',(16,64)),('D',(64,16)),('mean',(64,))]:
        value.load(name, torch.randn(shape)/8)
    return value


class NativePDReleaseTest(unittest.TestCase):
    def test_dense_recurrence_is_independent_of_codec_storage(self):
        self.assertTrue(native.dense_enabled(owner(0)))
        self.assertFalse(native.dense_enabled(owner(8)))
        value=owner(8);value.fullstack['gdn_every']=0
        self.assertTrue(native.dense_enabled(value))

    def test_transient_codec_matches_agg_bitwise_without_touching_boundary(self):
        for rank in (0,8):
            value=owner(rank);value.latent_codec=codec()
            for length, start in ((1,0),(8,0),(64,64),(65,8192)):
                streams=torch.randn(length,64).bfloat16()
                embeddings=torch.randn(length,16).bfloat16()
                before=streams.clone()
                fb=NS(positions=torch.arange(start,start+length))
                _,expected=value.latent_codec.encode_and_decode(
                    streams,fb.positions,embedding_streams(value,embeddings))
                actual=native.transient_emitter_streams(value,streams,embeddings,fb)
                self.assertTrue(torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)))
                self.assertTrue(torch.equal(streams.view(torch.uint8),before.view(torch.uint8)))

    @unittest.skipUnless(importlib.util.find_spec('twinstar_sgl'),
                         'boundary integration requires the deployed external PD helpers')
    def test_actual_boundary_attach_accepts_native_dense_and_checks_publication(self):
        from twinstar_sgl import pd_shallow as legacy
        from sglang.srt.disaggregation.state_handoff import HandoffKind,dispatch_handoff
        def pool():
            return NS(mamba_pool=NS(size=2,custom_mem_pool=None,
                register_slot_state=lambda state:None,
                mamba_cache=NS(temporal=torch.zeros(1,3,1,1,1,dtype=torch.bfloat16))))
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(AttributeError,'pd_state_handoffs'):
                legacy.attach(owner(),NS(device='cpu',req_to_token_pool=pool()))
            rp=pool();runner=NS(device='cpu',req_to_token_pool=rp)
            native.attach(owner(),runner)
            state=rp.pd_boundary_state
            self.assertEqual(set(rp.pd_state_handoffs),{HandoffKind.DENSE_BOUNDARY})
            req=NS(kv=NS(mamba_pool_idx=torch.tensor([1])))
            with self.assertRaisesRegex(RuntimeError,'not published'):
                dispatch_handoff(rp,'before_send',req)
            state.valid[1]=1
            dispatch_handoff(rp,'before_send',req)
            native.attach(owner(),runner)
            self.assertIs(rp.pd_boundary_state,state)
            self.assertEqual(len(rp.pd_state_handoffs),1)
            state.copy_slots(torch.tensor([1]),torch.tensor([2]))
            self.assertEqual(state.valid[2].item(),0)


    @unittest.skipUnless(importlib.util.find_spec('twinstar_sgl'),
                         'boundary integration requires deployed PD helpers')
    def test_real_prefill_dispatch_rejects_old_and_emits_transient_native_code(self):
        from twinstar_sgl import pd_shallow as legacy
        from twinstar_sgl import pd_emitter_graph
        from sglang.srt.model_executor import forward_context
        from sglang.srt.layers import communicator
        from sglang.srt.eplb import expert_distribution
        from sglang.srt.models import qwen4_exp
        class Layer:
            ple=None
            def __init__(self,i):self.i=i
            def __call__(self,*,hidden_states,**kw):
                return (hidden_states.repeat(1,4) if self.i==0 else hidden_states),None
        for rank in (0,8):
            value=owner(rank);value.latent_codec=codec()
            emitted=[]
            embeddings=torch.randn(4,16).bfloat16()
            raw=embeddings.repeat(1,4)
            fb=NS(extend_seq_lens_cpu=[4],extend_prefix_lens_cpu=[0],
                  return_logprob=False,batch_size=1,positions=torch.arange(4))
            body=NS(embed_tokens=lambda ids:embeddings,has_ple=False,
                    layers=[Layer(i) for i in range(32)])
            value.model=NS(model=body);value.bridges=[];value.p_layer_ids=range(31)
            value.emitter_ids=[31];value.emitters={'31':NS(
                is_attn=True,emit=lambda x,b:emitted.append(x.clone()))}
            value._boundary_lens=lambda b:[0]
            value._sub_batch=lambda *a:(fb,torch.arange(4))
            value.n_twinstar=value.n_prefix=0
            linear=NS(forward_metadata=None)
            linear.init_forward_metadata=lambda b:setattr(linear,'forward_metadata',NS())
            backend=NS(linear_attn_backend=linear,full_attn_backend=NS(
                init_forward_metadata=lambda b:None))
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ,{},clear=True))
                stack.enter_context(patch.object(forward_context,'get_attn_backend',return_value=backend))
                stack.enter_context(patch.object(communicator,'get_attn_tp_context',return_value=NS(
                    maybe_input_scattered=lambda b:nullcontext())))
                stack.enter_context(patch.object(expert_distribution,'get_global_expert_distribution_recorder',
                    return_value=NS(with_current_layer=lambda i:nullcontext())))
                stack.enter_context(patch.object(qwen4_exp,'_commit_ple_batch',return_value=None))
                stack.enter_context(patch.object(pd_emitter_graph,'try_emit',return_value=False))
                with self.assertRaisesRegex(ValueError,'served v3 latent path'):
                    legacy.prefill_extend(value,torch.arange(4),fb.positions,fb)
                self.assertEqual(emitted,[])
                out=native.prefill_extend(value,torch.arange(4),fb.positions,fb)
            _,expected=value.latent_codec.encode_and_decode(raw,fb.positions,raw)
            self.assertEqual(len(emitted),1)
            self.assertTrue(torch.equal(emitted[0].view(torch.uint8),expected.view(torch.uint8)))
            self.assertEqual(tuple(out.next_token_logits.shape),(1,32))


if __name__=='__main__':
    unittest.main()
