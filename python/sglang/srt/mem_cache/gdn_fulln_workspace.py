"""Bounded, shared full-N publication inputs; no token-sized state cache."""

from .gdn_prefill_commit_graph import BATCH_BUCKETS


def row_capacity(chunk_tokens, max_running_requests=None):
    if chunk_tokens is None or chunk_tokens < 2:
        raise ValueError("compact full-N requires chunked_prefill_size >= 2")
    rows = min(BATCH_BUCKETS[-1], chunk_tokens // 2)
    if max_running_requests is not None:
        if max_running_requests < 1:
            raise ValueError("compact full-N requires positive max_running_requests")
        rows = min(rows, max_running_requests)
    return next(b for b in BATCH_BUCKETS if b >= rows)


def state_memory_bytes(layers, heads, v, k, capacity, element_size=4):
    """Two slabs, independent of token buckets; intermediates are separate."""
    return 2 * layers * capacity * heads * v * k * element_size


def unique_state_bytes(shared):
    storages = {}
    for tensors in shared.values():
        for tensor in tensors:
            storage = tensor.untyped_storage()
            storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class FullNWorkspace:
    def __init__(self, pool, capacity, chunk_tokens):
        self.capacity = capacity
        self.chunk_tokens = chunk_tokens
        self.slabs = {
            role: pool.a.new_zeros(
                (len(pool.layer_ids), capacity, pool.hv, pool.v, pool.k)
            )
            for role in ("normal", "tracked")
        }
        # Captures use the original bucket shapes and arithmetic. Every smaller
        # bucket aliases a prefix of the same owned slab, on the same stream.
        self.shared = {
            (role, size): [layer[:size] for layer in slab.unbind(0)]
            for role, slab in self.slabs.items()
            for size in BATCH_BUCKETS if size <= capacity
        }

    def snapshot(self, index, dense, tracked, token_count):
        if not 2 <= token_count <= self.chunk_tokens:
            raise ValueError("full-N collection exceeds its configured token bound")
        rows = dense.shape[0]
        if not 0 < rows <= min(self.capacity, token_count // 2):
            raise ValueError("full-N collection exceeds its active row bound")
        if tracked is not None and tracked.shape[0] > rows:
            raise ValueError("full-N tracking must have at most one checkpoint per row")

        def copy(role, source):
            if source is None:
                return None
            target = self.slabs[role][index, :source.shape[0]]
            if target.shape != source.shape or not source.is_floating_point():
                raise ValueError("full-N dense collection shape/type changed")
            # BatchBuffers has always copied checkpoint inputs into FP32.
            # In particular BF16 h checkpoints widen here, before k31 math.
            target.copy_(source)
            return target

        # Never retain the producer's storage through the remaining layers.
        # These views are also the publication graph inputs, not another copy.
        return copy("normal", dense), copy("tracked", tracked)
