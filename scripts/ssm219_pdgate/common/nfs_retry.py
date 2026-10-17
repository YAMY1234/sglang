"""docs/222 §21.1 (LHR port): LHR's /lustre/fsw/... path is NFS4 (/scratch/fsw), not Lustre.  The window harness
coordinates its two nodes through files under V4_COORD_ROOT (atomic tmp + rename on one node, read on the other); on NFS a
read right after the other node's rename can fail with EBADHANDLE (521) / ESTALE (116) / EBADCOOKIE (523) (canary 253111:
'OSError: [Errno 521] ... control/j253111-S/D0-probe.json').  Imported by v3_services: Path.read_text / read_bytes retry
those errnos (40 x 0.25 s), every other error and every other call unchanged; CMH (Lustre) never takes the retry path.
"""
import errno
import pathlib
import time

RETRY = frozenset({errno.ESTALE, 521, 523})


def _wrap(fn):
    def call(self, *args, **kwargs):
        for i in range(40):
            try:
                return fn(self, *args, **kwargs)
            except OSError as exc:
                if exc.errno not in RETRY or i == 39:
                    raise
                time.sleep(0.25)
    call._nfs_retry = True
    call.__wrapped__ = fn
    return call


for _name in ('read_text', 'read_bytes'):
    if not getattr(getattr(pathlib.Path, _name), '_nfs_retry', False):
        setattr(pathlib.Path, _name, _wrap(getattr(pathlib.Path, _name)))
