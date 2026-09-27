"""Opt-in isolated byte oracle, independent of the staging Triton address math."""
import hashlib
import json
import os
from pathlib import Path

import torch


def logical_field_bytes(field, *, token_end=None):
    """Diagnostic split of valid token rows and transmitted page padding.

    The full wire digest and local byte comparison remain the admission gate.
    Incomplete compressed groups live in the separately audited pending ring.
    """
    if not field.tokens_per_row:
        return field.nbytes
    if field.token_start % field.compression or field.tokens_per_row % field.compression:
        raise ValueError('unaligned audited token field')
    elements_per_row = field.tokens_per_row // field.compression
    capacity = field.shape[0] * elements_per_row
    used = ((field.token_end if token_end is None else token_end) - field.token_start) // field.compression
    if not 0 <= used <= capacity or field.nbytes % capacity:
        raise ValueError('invalid audited token extent')
    return used * (field.nbytes // capacity)


def defined_field_bytes(manifest, field):
    """Diagnostic N-1 scope for the qualified P31 shallow-boundary contract.

    D executes deep token N itself. The physical packet still carries that row
    (or its incomplete compressed group); no payload or full-byte gate changes.
    """
    end=field.token_end
    if (getattr(manifest,'shallow_count',0)==9 and getattr(manifest,'deep_count',0)==8
            and field.layer in (31,35,39,43,47)
            and field.name in ('K','V','compressed_index') and field.tokens_per_row
            and end==manifest.prompt_tokens):
        end=max(field.token_start,end-1)
    return logical_field_bytes(field,token_end=end),end


def audit_payload(*, manifest, local, staging, directory, role, rank, rid):
    fields=[]
    for field in manifest.fields:
        digest=hashlib.sha256()
        logical_digest=hashlib.sha256();padding_digest=hashlib.sha256()
        logical_bytes=logical_field_bytes(field)
        defined_bytes,defined_end=defined_field_bytes(manifest,field)
        defined_digest=hashlib.sha256();boundary_digest=hashlib.sha256()
        view=local.get(field.key)
        width=field.nbytes//field.shape[0]
        step=max(1,(1<<20)//width)
        for first in range(0,field.shape[0],step):
            last=min(field.shape[0],first+step)
            wire=staging[field.offset+first*width:field.offset+last*width].cpu().numpy()
            digest.update(wire)
            split=max(0,min(len(wire),logical_bytes-first*width))
            logical_digest.update(wire[:split]);padding_digest.update(wire[split:])
            defined_split=max(0,min(len(wire),defined_bytes-first*width))
            defined_digest.update(wire[:defined_split]);boundary_digest.update(wire[defined_split:split])
            if view is None:
                if not field.handoff_only:raise ValueError('missing audited destination field')
                continue
            # index_select + host byte slicing is deliberately independent of
            # the custom kernel. Bound scratch memory to about one MiB.
            raw=view.tensor.index_select(0,view.rows[first:last]).contiguous().view(torch.uint8)
            raw=raw.reshape(last-first,-1).cpu().numpy()
            span=view.slice_bytes or width
            stride=view.group_stride or span
            actual=b''.join(raw[row,view.slice_offset+g*stride:view.slice_offset+g*stride+span].tobytes()
                            for row in range(last-first) for g in range(width//span))
            if actual!=wire.tobytes():
                raise ValueError(f'{role} staging differs from original local field {field.key}')
        fields.append(dict(layer=field.layer,name=field.name,bytes=field.nbytes,
                           sha256=digest.hexdigest(),local_compared=view is not None,
                           logical_bytes=logical_bytes,logical_sha256=logical_digest.hexdigest(),
                           defined_token_end=defined_end,defined_bytes=defined_bytes,
                           defined_sha256=defined_digest.hexdigest(),
                           boundary_bytes=logical_bytes-defined_bytes,boundary_sha256=boundary_digest.hexdigest(),
                           padding_bytes=field.nbytes-logical_bytes,padding_sha256=padding_digest.hexdigest()))
    result=dict(passed=True,role=role,rank=rank,rid=rid,manifest=json.loads(manifest.to_bytes()),fields=fields,
                manifest_sha256=hashlib.sha256(manifest.to_bytes()).hexdigest())
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    key=f'{role}-rank{rank}-room{manifest.room}-g{manifest.generation}-c{manifest.chunk_index}.json'
    temporary=path/(key+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(result,sort_keys=True)+'\n');temporary.replace(path/key)
    return result
