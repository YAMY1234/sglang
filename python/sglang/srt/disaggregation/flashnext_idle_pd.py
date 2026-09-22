"""Full-KV transfer descriptors for a prefill-only idle latent cache.

The P peer has shallow radix pages and request-private deep pages. The D peer
has the ordinary twelve-layer KV pool. No latent field crosses the wire.
"""
import os

DEEP_LAYERS = (31, 35, 39, 43, 47)


def enabled(pool):
    return bool(getattr(pool, "idle_only", False)) or (
        os.environ.get("TWINSTAR_IDLE_PD_FULL_KV") == "1"
        and os.environ.get("TWINSTAR_FULLSTACK", "1") != "0"
    )


def select_deep(infos, layer_ids, *, groups):
    """Keep occurrence order (all K then all V), never sort pointer arrays."""
    if len(infos) != 3 or any(len(column) != len(layer_ids) for column in infos):
        raise ValueError("idle PD buffer metadata lengths differ")
    indices = [i for i, layer in enumerate(layer_ids) if layer in DEEP_LAYERS]
    ids = [layer_ids[i] for i in indices]
    if ids != list(DEEP_LAYERS) * groups:
        raise ValueError("idle PD requires all five deep QSA layers in each tensor group")
    return tuple([column[i] for i in indices] for column in infos), ids


def deep_buffers(pool):
    source = pool.deep if getattr(pool, "idle_only", False) else pool
    kv, kv_ids = select_deep(source.get_contiguous_buf_infos(),
                            source.get_kv_layer_ids(), groups=2)
    compressed, compressed_ids = select_deep(
        source.get_qsa_compressed_state_buf_infos(),
        source.get_qsa_compressed_state_layer_ids(), groups=1)
    return (kv, kv_ids), (compressed, compressed_ids)
