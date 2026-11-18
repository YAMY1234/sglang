"""PP agreement for the external linker's prefetch mode; scheduler thread only.

Ported from GLM's PPLinkerRelay. Its state machine (per request: lookup ->
load -> ready, with ordered ``load`` / ``finish`` actions) is kept; two things
change to fit this tree:

* Only PP0 looks the store up, for every stage at once (``lookup_stages``), so
  the lookup needs no votes.
* Votes on a read's outcome go to PP0 through a ``PPConsensusStore`` (the
  prefill release ring that carries GLM's votes is off under
  SGLANG_PP_PD_PREFILL_RELEASE_RELAY). PP0's actions ride the request relay and
  every stage, PP0 included, applies them as it receives them, before its
  admission pass, so all stages see the same phase at the same pass.
"""

from dataclasses import dataclass, field


@dataclass
class PPLinkerActions:
    actions: list = field(default_factory=list)


@dataclass
class _Request:
    req: object
    cycle: int
    phase: str = "lookup"
    stopped: bool = False
    released: bool = False
    tokens: int = 0
    # This stage's vote on its read (1 landed, 2 not), once it finished.
    voted: object = None
    # PP0: a load/finish action issued and not yet applied; a stop issued.
    pending: bool = False
    stop_issued: bool = False


_LANDED, _FAILED = 1, 2


class PPLinkerRelay:
    def __init__(self, wrapper):
        self.w = wrapper
        self.rank, self.size = wrapper.pp_rank, wrapper.pp_size
        self.epoch = self.cycle = self.serial = self.issued = 0
        self.entries = {}
        self.inbox, self.outbox = [], []
        # PPConsensusStore, attached when the PP event loop starts.
        self.store = None
        self.polling = False

    def attach_store(self, store):
        self.store = store

    def begin(self, req):
        handle = req.cache_request_handle
        entry = self.entries.get(handle)
        if entry is None or entry.phase == "ready":
            self.cycle += 1
            self.entries[handle] = _Request(req, self.cycle)

    def settle(self, handle):
        """No stage queued a lookup for ``handle`` (the same decision on every
        stage, from the same tree): it is ready without a relay round."""
        entry = self.entries.get(handle)
        if entry is not None and entry.phase == "lookup" and not entry.pending:
            entry.phase = "ready"
            if entry.released:
                del self.entries[handle]

    def release(self, handle):
        entry = self.entries.get(handle)
        if entry is not None:
            entry.released = True
            if entry.phase == "ready":
                del self.entries[handle]

    def _key(self, entry):
        return f"lk{self.epoch}:{entry.cycle}"

    def report(self):
        """Every stage: vote on each read that finished here."""
        if self.store is None:
            return
        w = self.w
        for handle, entry in self.entries.items():
            if entry.phase != "load" or entry.voted is not None:
                continue
            state = w.prefetches.get(handle)
            done = state is None or state.future is None or state.future.done()
            if not done:
                continue
            landed = False
            try:
                landed = bool(
                    state
                    and state.future
                    and not state.future.cancelled()
                    and state.future.result() is True
                    and not state.discard
                )
            except Exception:
                pass
            entry.voted = _LANDED if landed else _FAILED
            self.store[self._key(entry)] = entry.voted

    def take_actions(self):
        """PP0: the actions to relay with this round's requests, or None."""
        if not self.outbox:
            return None
        actions, self.outbox = self.outbox, []
        return PPLinkerActions(actions)

    def _issue(self, handle, entry, op, value):
        self.issued += 1
        if op != "stop":
            entry.pending = True
        self.outbox.append((self.epoch, self.issued, handle, entry.cycle, op, value))

    def receive(self, actions):
        """Every stage, as the relay delivers PP0's actions (in order)."""
        self.inbox.extend(actions)
        self._apply_inbox()

    def _apply_inbox(self):
        while self.inbox:
            epoch, serial, handle, cycle, op, value = self.inbox[0]
            if epoch < self.epoch:
                self.inbox.pop(0)
                continue
            entry = self.entries.get(handle)
            if entry is None or entry.cycle != cycle:
                # Its request is not enqueued here yet; actions apply in order.
                return
            self.inbox.pop(0)
            assert epoch == self.epoch and serial == self.serial + 1, "PP linker action order"
            self.serial = serial
            self._apply(handle, entry, op, value)

    def _apply(self, handle, entry, op, value):
        w = self.w
        if op == "load":
            assert entry.phase == "lookup"
            entry.pending = False
            lookup = w.lookups.pop(handle, None)
            if value and lookup is not None:
                w._draining_lookups = True
                try:
                    w._start_prefetch_load(
                        handle,
                        lookup.key,
                        lookup.hashes,
                        lookup.start,
                        lookup.anchor_node_id,
                        lookup.anchor_lock,
                        [value],
                        lookup.timeout_s,
                    )
                finally:
                    w._draining_lookups = False
                entry.tokens = value * w.cache.page_size
            elif lookup is not None:
                # PP0 found nothing restorable past the device prefix on every stage.
                w._lookup_stats["zero_restorable"] += 1
                # Metadata-only RPCs never own private device writes.
                if lookup.future is not None:
                    lookup.future.cancel()
                w.cache.dec_lock_ref(lookup.anchor_node_id, lookup.anchor_lock)
            entry.phase = "load" if value else "ready"
        elif op == "finish":
            assert entry.phase == "load"
            entry.pending = False
            w._finish_prefetch(handle, publish=value, hold=not entry.released)
            if entry.voted is not None and self.store is not None:
                self.store.pop(self._key(entry))
            entry.phase = "ready"
        elif op == "stop":
            if entry.phase != "ready":
                entry.stopped = True
                w._cancel_prefetch(handle)
        else:
            raise AssertionError(op)
        if entry.phase == "ready" and entry.released:
            del self.entries[handle]

    def poll(self):
        """Every stage, each scheduler step: vote; PP0 also decides."""
        if self.polling:
            return
        self.polling = True
        try:
            self.report()
            if self.rank == 0 and self.store is not None:
                self._decide()
        finally:
            self.polling = False

    def _decide(self):
        w = self.w
        for handle, entry in list(self.entries.items()):
            if entry.pending:
                continue
            if entry.phase == "lookup":
                lookup = w.lookups.get(handle)
                if lookup is not None and lookup.future is not None and not lookup.future.done():
                    continue
                pages = 0
                if lookup is not None and lookup.future is not None and not (
                    entry.stopped or entry.stop_issued
                ):
                    try:
                        restorable = lookup.future.result()
                    except Exception:
                        restorable = []
                    pages = max(
                        (n for n in restorable if 0 < n <= len(lookup.hashes)), default=0
                    )
                self._issue(handle, entry, "load", pages)
            elif entry.phase == "load":
                if entry.voted is None:
                    continue
                votes = self.store.collect(self._key(entry))
                if any(v is None for v in votes):
                    continue
                publish = (
                    all(v == _LANDED for v in votes)
                    and not entry.stopped
                    and not entry.stop_issued
                )
                self._issue(handle, entry, "finish", publish)

    def progress(self, handle, stop_due):
        """Admission: True once ``handle`` needs no more waiting. The answer
        depends only on applied actions, so every stage gives the same one."""
        entry = self.entries.get(handle)
        if entry is None or entry.phase == "ready" or entry.stopped:
            return True
        finishing = entry.pending and entry.phase == "load"
        if self.rank == 0 and not entry.stop_issued and not finishing and stop_due(handle):
            # PP0 alone runs the stop policy; the stop applies everywhere in order.
            entry.stop_issued = True
            self._issue(handle, entry, "stop", None)
        return False

    def reset(self):
        # Backend reset/close must have quiesced writers before this call.
        if self.store is not None:
            for entry in self.entries.values():
                if entry.voted is not None:
                    self.store.pop(self._key(entry))
        self.epoch += 1
        self.serial = self.issued = 0
        self.entries.clear()
        self.inbox.clear()
        self.outbox.clear()
