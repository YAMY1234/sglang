"""P12 local phase selection; the canonical wire manifest never changes."""
from .flashnext_staging_manifest import align


def staged_kv_phases(manifest):
    """Partition the entire aligned packet once, retaining every original offset."""
    manifest.validate()
    early = frozenset(f.key for f in manifest.fields
                      if 0 <= f.layer < 31 and f.name in ('K', 'V')
                      and f.tokens_per_row and not f.handoff_only)
    late = frozenset(f.key for f in manifest.fields) - early
    if not early or not late:
        raise ValueError("P12 needs both shallow KV and a final state payload")

    def ranges(keys):
        result = []
        for field in manifest.fields:
            if field.key not in keys:
                continue
            size = align(field.nbytes)
            if result and result[-1][0] + result[-1][1] == field.offset:
                start, previous = result[-1]
                result[-1] = (start, previous + size)
            else:
                result.append((field.offset, size))
        return tuple(result)

    return early, late, ranges(early), ranges(late)
