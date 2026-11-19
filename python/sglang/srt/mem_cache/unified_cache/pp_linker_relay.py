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
import logging
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@dataclass
class PPLinkerActions:
    actions: list = field(default_factory=list)


@dataclass
class PPLinkerOffloadCounts:
    """PP0's linker offload commits of one loop iteration, appended to the request
    relay: one [count, *verdicts] row per _check_linker_events_pp call."""

    counts: list = field(default_factory=list)


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
        # SGLANG_LINKER_PP_COUNTS_CONSENSUS (GLM's offload commit): this stage's
        # finished offloads not yet committed, their absolute index head, and on
        # PP0 the head of what it has already issued.
        self.offload_head, self.offload_results = 0, []
        self.offload_issued_head = 0
        self._offload_posted = -1
        self._ack_posted = None
        # Offload actions only: applied here, and (PP0) issued. Prefetch actions
        # in flight never hold an offload commit back.
        self.offload_serial = self.offload_issued = 0
        # PP0 commit-lag stats (SGLANG_EXTERNAL_LINKER_LOOKUP_STATS).
        self._decide_step = 0
        self._covered = 0
        self._covered_at = []
        self._lag_steps, self._lag_ms = [], []
        self._locked_max = (0, 0)
        self._stats_next = time.monotonic() + 60.0

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
            if not landed and state is not None and state.abort_reason is None:
                state.abort_reason = w.local_abort_reason(state)
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
        if entry is not None and op != "stop":
            entry.pending = True
        cycle = entry.cycle if entry is not None else 0
        self.outbox.append((self.epoch, self.issued, handle, cycle, op, value))

    # ---- offload commits (GLM's report/poll 'offload', votes on the store) ----

    def _offload_key(self, index):
        return f"lko{self.epoch}:{index}"

    def offload_report(self):
        """Every stage, each step: harvest this stage's finished offloads (no
        waiting) and post the absolute count harvested; a failed one is posted
        before the count, on the same ordered channel."""
        w = self.w
        n = w.num_completed_offloads()
        if n:
            base = self.offload_head + len(self.offload_results)
            results = w.take_completed_offloads(n)
            for i, ok in enumerate(results):
                if not ok:
                    self.store[self._offload_key(base + i)] = 0
            self.offload_results.extend(results)
        done = self.offload_head + len(self.offload_results)
        if done != self._offload_posted:
            self.store[f"lkoc{self.epoch}"] = done
            self._offload_posted = done
        # GLM's acknowledgement, on the offload actions only: the epoch and the
        # number of offload commits applied here.
        if self._ack_posted != (self.epoch, self.offload_serial):
            self.store["lkoe"] = self.epoch
            self.store["lkos"] = self.offload_serial
            self._ack_posted = (self.epoch, self.offload_serial)

    def offload_decide(self):
        """PP0: commit what every stage has finished, the AND of their outcomes,
        once every stage acknowledged this epoch and every action issued so far
        (GLM: a vote predating an action cannot authorize another offload unlock)."""
        self._decide_step += 1
        stats = self.w.log_lookup_stats
        counts = self.store.collect(f"lkoc{self.epoch}")
        finished = min((c or 0) for c in counts)
        if stats and finished > self._covered:
            now = time.monotonic()
            self._covered_at.extend([(self._decide_step, now)] * (finished - self._covered))
            self._covered = finished
        if stats:
            self._log_offload_stats()
        if any(e != self.epoch for e in self.store.collect("lkoe")) or any(
            s != self.offload_issued for s in self.store.collect("lkos")
        ):
            return
        assert self.offload_issued_head == self.offload_head, "PP linker offload head drift"
        count = finished - self.offload_head
        if count <= 0:
            return
        head = self.offload_head
        successes = [
            all(v is None for v in self.store.collect(self._offload_key(head + i)))
            for i in range(count)
        ]
        self._issue(None, None, "offload", (head, successes))
        self.offload_issued_head += count
        self.offload_issued += 1
        if stats:
            now = time.monotonic()
            for step, at in self._covered_at[:count]:
                self._lag_steps.append(self._decide_step - step)
                self._lag_ms.append(1e3 * (now - at))
            del self._covered_at[:count]

    def _log_offload_stats(self):
        w = self.w
        pending = getattr(w, "pending_offloads", ())
        tokens = getattr(w, "pending_offload_tokens", {})
        locked = (len(pending), sum(tokens.get(p.lock_node_id, 0) for p in pending))
        self._locked_max = max(self._locked_max, locked)
        now = time.monotonic()
        if now < self._stats_next:
            return
        self._stats_next = now + 60.0

        def pct(xs):
            xs = sorted(xs)
            if not xs:
                return "n=0"
            return (f"n={len(xs)} p50 {xs[len(xs) // 2]:.0f} p90 "
                    f"{xs[len(xs) * 9 // 10]:.0f} max {xs[-1]:.0f}")

        logger.info(
            "Linker PP offload commits: lag steps %s; lag ms %s; locked now %d nodes "
            "%d tokens, max %d nodes %d tokens; committed head %d, finished on all %d",
            pct(self._lag_steps), pct(self._lag_ms), locked[0], locked[1],
            self._locked_max[0], self._locked_max[1], self.offload_head, self._covered,
        )
        self._lag_steps, self._lag_ms = [], []
        self._locked_max = locked

    def _apply_offload(self, value):
        head, successes = value
        assert head == self.offload_head, "PP linker offload commit order"
        n = len(successes)
        assert n <= len(self.offload_results), "PP linker offload committed before it finished here"
        self.w.commit_completed_offloads(successes)
        for i in range(n):
            if not self.offload_results[i]:
                self.store.pop(self._offload_key(head + i))
        del self.offload_results[:n]
        self.offload_head += n
        self.offload_serial += 1

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
            if op == "offload":
                self.inbox.pop(0)
                assert epoch == self.epoch and serial == self.serial + 1, "PP linker action order"
                self.serial = serial
                self._apply_offload(value)
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
                w._cancel_prefetch(handle, "stopped")
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
                    if envs.SGLANG_LINKER_PP_ADMIT_TRACE.get():
                        logger.info(
                            "PP lookup trace: rid=%s start=%d pages=%d restorable=%s -> %d",
                            handle.rid, lookup.start, len(lookup.hashes),
                            sorted(restorable)[-8:], pages,
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
        if self.store is not None:
            # This epoch's offload votes: uncommitted failures and the count.
            for i, ok in enumerate(self.offload_results):
                if not ok:
                    self.store.pop(self._offload_key(self.offload_head + i))
            self.store.pop(f"lkoc{self.epoch}")
        self.epoch += 1
        self.serial = self.issued = 0
        self._ack_posted = None
        self.offload_serial = self.offload_issued = 0
        self._covered, self._covered_at = 0, []
        self.offload_head, self.offload_results = 0, []
        self.offload_issued_head = 0
        self._offload_posted = -1
        self.entries.clear()
        self.inbox.clear()
        self.outbox.clear()
