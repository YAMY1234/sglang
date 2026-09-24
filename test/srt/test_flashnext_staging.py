"""TRITON_INTERPRET=1 executes the real keyword transport copy entry on CPU."""
import json
import os
os.environ.setdefault("TRITON_INTERPRET", "1")
import unittest

import torch

from sglang.srt.disaggregation.flashnext_staging_manifest import Field, Manifest, LeasePool
from sglang.srt.disaggregation.flashnext_staging_kernels import LocalRows, copy_payload


def manifest(fields, tokens=257):
    return Manifest.build(room=71,generation=1,source_rank=0,source_tp=2,
        prompt_tokens=tokens,chunk_index=0,last_chunk=True,shallow_count=9,
        deep_count=8,fields=fields)


class StagingTest(unittest.TestCase):
    def test_wire_has_no_physical_map_and_rejects_corruption(self):
        m=manifest([Field(3,"K","bfloat16",(5,64,3),0,257,64),
                    Field(0,"count","int32",(1,2),0,257,0)])
        self.assertEqual(Manifest.from_bytes(m.to_bytes()),m)
        for word in (b"ptr",b"address",b"page_id",b"rows"):
            self.assertNotIn(word,m.to_bytes())
        x=json.loads(m.to_bytes());x['fields'][1]['offset']=0
        with self.assertRaises(ValueError):Manifest.from_bytes(json.dumps(x).encode())
        with self.assertRaises(ValueError):manifest([m.fields[0],m.fields[0]])

    def test_real_copy_entry_fragmented_pages_all_bytes_and_tail(self):
        gen=torch.Generator().manual_seed(91)
        for tokens in (1,63,64,65,257):
            n=(tokens+63)//64
            specs=[Field(3,"K","bfloat16",(n,64,3),0,tokens,64),
                   Field(47,"V","bfloat16",(n,64,2),0,tokens,64),
                   Field(47,"compressed_index","bfloat16",(n,16,4),0,tokens,64,compression=4),
                   Field(-1,"latent","uint8",(n,64,5),0,tokens,64,handoff_only=True),
                   Field(0,"count","int32",(1,2),0,tokens,0),
                   Field(0,"conv","float32",(1,3,4),0,tokens,0),
                   Field(31,"h31","bfloat16",(1,17),0,tokens,0)]
            m=manifest(specs,tokens)
            source={};target={};expected={}
            for f in m.fields:
                rowbytes=f.nbytes//f.shape[0]
                src=torch.randint(0,256,(13,rowbytes),dtype=torch.uint8,generator=gen)
                dst=torch.full_like(src,237)
                a=torch.tensor([11,4,8,2,9][:f.shape[0]],dtype=torch.int64)
                b=torch.tensor([3,10,1,7,5][:f.shape[0]],dtype=torch.int64)
                source[f.key]=LocalRows(src,a);target[f.key]=LocalRows(dst,b)
                expected[f.key]=src[a].clone()
            staging=torch.full((m.nbytes,),129,dtype=torch.uint8)
            # Same real entry and keyword ABI the background transport will use.
            copy_payload(manifest=m,local=source,staging=staging,gather=True)
            received=staging.clone()  # exact bulk byte-copy model
            copy_payload(manifest=m,local=target,staging=received,gather=False)
            for f in m.fields:
                view=target[f.key]
                self.assertTrue(torch.equal(view.tensor[view.rows],expected[f.key]))
                self.assertTrue(torch.all(view.tensor[0]==237))
                self.assertTrue(torch.all(staging[f.offset+f.nbytes:(f.offset+f.nbytes+255)//256*256]==0))

    def test_head_slice_scatter_uses_destination_layout(self):
        # Two outer groups, P owns heads 0..7, D owns 0..3, this payload 2..3.
        f=Field(3,"K","uint8",(3,2,2,4),0,3,1,shard_axis=2,
                head_start=2,head_end=4,total_heads=8)
        m=manifest([f],3)
        src=torch.arange(7*2*8*4,dtype=torch.int64).to(torch.uint8).view(7,2,8,4)
        dst=torch.full((8,2,4,4),221,dtype=torch.uint8)
        a=torch.tensor([6,2,4]);b=torch.tensor([1,5,3])
        staging=torch.empty(m.nbytes,dtype=torch.uint8)
        copy_payload(manifest=m,local={f.key:LocalRows(src,a,8,32,8)},staging=staging,gather=True)
        copy_payload(manifest=m,local={f.key:LocalRows(dst,b,8,16,8)},staging=staging,gather=False)
        self.assertTrue(torch.equal(dst[b,:,2:4],src[a,:,2:4]))
        self.assertTrue(torch.all(dst[b,:,:2]==221))

    def test_pool_backpressure_and_abort_drain_generation(self):
        p=LeasePool(slots=2,slot_bytes=1024)
        a=p.acquire(room=4,nbytes=1000);b=p.acquire(room=5,nbytes=500)
        self.assertIsNone(p.acquire(room=6,nbytes=100))
        p.begin(a,operation='bulk');p.abort(a)
        with self.assertRaises(RuntimeError):p.release(a)
        with self.assertRaises(ValueError):p.begin(a,operation='late-scatter')
        p.finish(a,operation='bulk');p.release(a)
        c=p.acquire(room=6,nbytes=200)
        self.assertEqual(a.slot,c.slot);self.assertGreater(c.generation,a.generation)
        with self.assertRaises(ValueError):p.finish(a,operation='late-bulk')
        p.begin(c,operation='scatter')
        with self.assertRaises(RuntimeError):p.release(c)
        p.finish(c,operation='scatter');p.release(c);p.release(b)
        self.assertEqual((p.peak_slots,p.peak_bytes,p.reserved_bytes),(2,1500,2048))


if __name__=='__main__':unittest.main()
