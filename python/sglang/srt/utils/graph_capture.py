"""Coordinate CUDA capture with staging and opt-in publication ownership."""
from contextvars import ContextVar
from threading import RLock, local

# Only the opt-in model forward binds an owner. Off-state uses the original lock.
compression_capture_owner = ContextVar("compression_capture_owner", default=None)


class _CaptureLock:
    def __init__(self):
        self.lock = RLock()
        self.local = local()

    def __enter__(self):
        self.lock.acquire()
        context = None
        try:
            owner = compression_capture_owner.get()
            if owner is not None:
                context = owner.capture_scope("graph_capture_lock")
                context.__enter__()
            if not hasattr(self.local, "contexts"):
                self.local.contexts = []
            self.local.contexts.append(context)
        except BaseException:
            self.lock.release()
            raise
        return self

    def __exit__(self, kind, error, traceback):
        context = self.local.contexts.pop()
        try:
            if context is not None:
                return context.__exit__(kind, error, traceback)
        finally:
            self.lock.release()


graph_capture_lock = _CaptureLock()
