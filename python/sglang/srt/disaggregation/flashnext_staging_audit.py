"""Opt-in isolated byte oracle, independent of the staging Triton address math."""
import hashlib
import json
import os
from pathlib import Path

import torch


def audit_payload(*, manifest, local, staging, directory, role, rank, rid):
    fields=[]
    for field in manifest.fields:
        digest=hashlib.sha256()
        view=local.get(field.key)
        width=field.nbytes//field.shape[0]
        step=max(1,(1<<20)//width)
        for first in range(0,field.shape[0],step):
            last=min(field.shape[0],first+step)
            wire=staging[field.offset+first*width:field.offset+last*width].cpu().numpy()
            digest.update(wire)
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
                           sha256=digest.hexdigest(),local_compared=view is not None))
    result=dict(passed=True,role=role,rank=rank,rid=rid,manifest=json.loads(manifest.to_bytes()),fields=fields,
                manifest_sha256=hashlib.sha256(manifest.to_bytes()).hexdigest())
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    key=f'{role}-rank{rank}-room{manifest.room}-g{manifest.generation}-c{manifest.chunk_index}.json'
    temporary=path/(key+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(result,sort_keys=True)+'\n');temporary.replace(path/key)
    return result
