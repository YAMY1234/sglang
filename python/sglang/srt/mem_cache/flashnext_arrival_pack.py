"""Foreground batch payload/embedding gather and one owned control transfer."""
import torch


def prepare_packed(model, pool, items):
    if not items or any(item['offset'] != sum(x['stop'] for x in items[:i])
                        for i, item in enumerate(items)):
        raise ValueError('arrival segments must retain their packed order')
    locations = torch.cat([pool.request_pool.req_to_token[x['slot'], :x['stop']]
                           for x in items])
    payload = pool.load_latent(locations)
    token_ids = payload.pop('token_ids').flatten().long()
    base = model.model.model.embed_tokens(token_ids)
    if base.shape[-1] not in (model.config.hidden_size,
                             model.config.hc_count * model.config.hidden_size):
        raise ValueError('unexpected packed arrival embedding width')
    return dict(payload=payload, token_ids=token_ids, base=base,
                stop=sum(x['stop'] for x in items))


def control_rows(items, chunk):
    return [(x, start, (x['slot'], start, x['offset']))
            for x in items for start in range(0, x['stop'], chunk)]


def transfer_controls(rows, device):
    host = torch.tensor([r[2] for r in rows], dtype=torch.int64)
    if torch.device(device).type == 'cuda':
        host = host.pin_memory()
    return host, host.to(device=device, non_blocking=host.is_pinned())
