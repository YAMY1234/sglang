"""Default-OFF TRT prefill kernel ports; original SG outputs and stream ordering."""
from functools import lru_cache
import logging
import os

import torch

log=logging.getLogger(__name__)
_LOGGED=set()


def enabled(name='SGLANG_HCB_PREFILL_TRT_PORT'):
    return os.environ.get(name,'0').lower() in ('1','true','yes','y')


@lru_cache(maxsize=1)
def _ops():
    torch.ops.load_library(os.environ['SGLANG_HCB_PREFILL_PORT_SO'])
    return torch.ops.hcb_prefill


def _notice(kind,shape):
    if kind not in _LOGGED:
        log.info('HCB_PREFILL_PORT active kind=%s shape=%s binary=%s',kind,shape,
                 os.environ['SGLANG_HCB_PREFILL_PORT_SO'])
        _LOGGED.add(kind)


def topk_paged(logits,lengths,page_table,cuq,row_starts,batch_idx):
    """Ported TRT selector and selected-only remap; no full page-table expansion."""
    rows=logits.shape[0]
    out=torch.empty((rows,2048),dtype=torch.int32,device=logits.device)
    if rows==0:return out
    starts=row_starts if row_starts is not None else torch.zeros(rows,dtype=torch.int32,device=logits.device)
    op=_ops()
    if batch_idx is None:
        op.topk(logits,lengths,starts,page_table,cuq,out,1)
    else:
        relative=_fi_select(logits,lengths,starts) if _TOPK=='fi_gvr2' else None
        if relative is None:
            relative=torch.empty_like(out)
            # Variant2 is index selection only; the page/cu views satisfy its shared
            # tensor API but are not read by that specialization.
            op.topk(logits,lengths,starts,page_table[:1],cuq[:2],relative,2)
        elif _SHADOW:
            ref=torch.empty_like(out)
            op.topk(logits,lengths,starts,page_table[:1],cuq[:2],ref,2)
            _shadow('main',logits,starts,lengths,ref,relative)
        op.remap(relative,page_table,batch_idx.to(torch.int64).contiguous(),out)
    _notice('selector-fi-gvr2-remap' if _TOPK=='fi_gvr2' else 'selector-selected-remap',tuple(logits.shape))
    return out


# Default-OFF FlashInfer gvr_2 selector (SGLANG_HCB_PREFILL_TOPK=fi_gvr2) for the prefill DSA top-k: the HCB PAGED
# path above and the unfused draft-extend (MTP) path via fi_positions(). gvr_2's windowed output is int32 row-relative
# positions, identity + -1 tail for rows with <= 2048 positions, the same contract as the TRT port and fast_topk_v2.
_TOPK=os.environ.get('SGLANG_HCB_PREFILL_TOPK','')
_SHADOW=enabled('SGLANG_HCB_PREFILL_TOPK_SHADOW')
_SHADOW_STATS={}
_FALLBACKS=[0]


@lru_cache(maxsize=1)
def _fi():
    import time
    from flashinfer.topk_varlen.topk_varlen import _gvr2_top_k_varlen_check, top_k_varlen
    import flashinfer.topk_varlen as ftv
    t=time.perf_counter()
    ftv.warmup_prefill(2048,1<<20)
    log.info('HCB_TOPK_FI warmup_prefill(2048) done in %.1f s',time.perf_counter()-t)
    return top_k_varlen,_gvr2_top_k_varlen_check


def _fi_select(logits,lengths,starts):
    """gvr_2 positions, or None when gvr_2 refuses this layout (the caller's deployed selector serves the call)."""
    run,fits=_fi()
    # gvr_2's own problem-size rule, the one top_k_varlen raises on (metadata only, no sync). It has no row-count or
    # width bound; it refuses e.g. a 1-row chunk whose width is not a multiple of 4 (pitch = width when rows == 1).
    if not fits(logits,lengths,2048,row_starts=starts,backend='gvr_2'):
        _FALLBACKS[0]+=1; n=_FALLBACKS[0]
        if n&(n-1)==0:
            log.info('HCB_TOPK_FI fallback=%d (deployed selector) shape=%s stride=%s ptr16=%d',n,tuple(logits.shape),
                     logits.stride(),logits.data_ptr()%16)
        return None
    idx=run(logits,lengths,2048,backend='gvr_2',row_starts=starts)
    return idx[0] if isinstance(idx,(tuple,list)) else idx


def fi_positions(score,lengths,row_starts,legacy):
    """Draft-extend (MTP) unfused top-k: gvr_2 positions when the switch is on and gvr_2 takes the layout, else None
    (caller runs legacy)."""
    if _TOPK!='fi_gvr2' or not score.is_cuda:
        return None
    got=_fi_select(score,lengths,row_starts)
    if got is None:
        return None
    if _SHADOW:
        _shadow('mtp',score,row_starts,lengths,legacy(),got)
    _notice('mtp-fi-gvr2',tuple(score.shape))
    return got


def _shadow(tag,logits,starts,lengths,ref,got):
    # smoke-only: rows whose position sets differ; rows whose value multisets differ (beyond ties), and for up to 64
    # of those per call, which side torch.topk on the row's window agrees with
    def vals(p):
        v=torch.gather(logits,1,(starts.to(torch.int64)[:,None]+p.clamp(min=0).to(torch.int64)))
        return torch.sort(torch.where(p<0,torch.full_like(v,float('-inf')),v),1).values
    diff=(torch.sort(ref,1).values!=torch.sort(got,1).values).any(1)
    vr,vg=vals(ref),vals(got)
    beyond=diff&(vr!=vg).any(1)
    st=_SHADOW_STATS.setdefault(tag,[0,0,0,0,0,0])
    st[0]+=1; st[1]+=ref.shape[0]; st[2]+=int(diff.sum()); nb=int(beyond.sum()); st[3]+=nb
    if nb:
        for r in beyond.nonzero().flatten()[:64].tolist():
            s,n=int(starts[r]),int(lengths[r]); k=min(2048,n)
            truth=torch.sort(torch.topk(logits[r,s:s+n],k).values,descending=True).values
            st[4]+=int(not torch.equal(torch.sort(vg[r][vg[r]>float('-inf')],descending=True).values,truth))
            st[5]+=int(not torch.equal(torch.sort(vr[r][vr[r]>float('-inf')],descending=True).values,truth))
    if st[0]%100==1 or nb:
        log.info('HCB_TOPK_SHADOW path=%s calls=%d rows=%d differ=%d beyond_ties=%d fi_wrong=%d deployed_wrong=%d',tag,*st)

