"""PP agreement for the existing external linker; scheduler thread only.

Votes ride the PD transfer/release ring; ordered actions ride the request
relay. Neither path creates a message, a collective, or a scheduler wait.
Generic lookup/load/hold/re-query policy remains in UnifiedCacheLinkerWrapper.
"""
from dataclasses import dataclass


@dataclass
class PPLinkerActions:
    actions: list


class PPLinkerReport(dict):
    """Transfer outcomes retain their dict interface on the existing ring."""

    def __init__(self, outcomes, reports):
        super().__init__(outcomes)
        self.reports = reports


@dataclass
class _Request:
    req: object
    cycle: int
    phase: str = 'lookup'
    stopped: bool = False
    released: bool = False
    result: object = None
    tokens: int = 0
    agreed: int = 0


class PPLinkerRelay:
    def __init__(self, wrapper):
        self.w = wrapper
        self.rank, self.size = wrapper.cache.pp_rank, wrapper.cache.pp_size
        self.epoch = self.cycle = self.serial = 0
        self.entries, self.reports = {}, {}
        self.inbox, self.outbox = [], []
        self.offload_head, self.offload_results = 0, []
        self.polling = False

    def begin(self, req):
        handle = req.cache_request_handle
        entry = self.entries.get(handle)
        if entry is None or entry.phase == 'ready':
            self.cycle += 1
            self.entries[handle] = _Request(req, self.cycle)

    def release(self, handle):
        entry = self.entries.get(handle)
        if entry is not None:
            entry.released = True
            if entry.phase == 'ready':
                del self.entries[handle]

    def report(self):
        w = self.w
        # Harvest only; the tree locks are released by a shared commit action.
        self.offload_results.extend(w.take_completed_offloads(w.num_completed_offloads()))
        votes = {}
        for handle, entry in self.entries.items():
            lookup, state = w.lookups.get(handle), w.prefetches.get(handle)
            if entry.phase == 'lookup':
                done = lookup is None or lookup.future.done()
                if lookup is not None and done and entry.result is None:
                    try:
                        entry.result = lookup.future.result()
                    except Exception:
                        entry.result = []
                signature = (lookup.start, len(lookup.key), len(lookup.hashes), lookup.hashes[-1]) if lookup else None
                value = (done, signature, entry.result or [])
            elif entry.phase == 'load':
                done = state is None or state.future is None or state.future.done()
                landed = False
                try:
                    landed = bool(state and state.future and done and not state.future.cancelled()
                                  and state.future.result() is True and not state.discard)
                except Exception:
                    pass
                # A published prefix must be held on EVERY stage. Never expose
                # a partial-stage hit when one stage cannot retain its prefix.
                value = (done, landed and not entry.stopped)
            else:
                continue
            votes[handle] = (entry.cycle, entry.phase, value)
        room = w.cache.token_to_kv_pool_allocator.available_size() + w.cache.evictable_size()
        return (self.epoch, votes, self.offload_head, list(self.offload_results),
                self.serial, max(0, w.hold_cap_tokens - w.held_tokens), room)

    def take_actions(self):
        actions, self.outbox = self.outbox, []
        return PPLinkerActions(actions)

    def receive_reports(self, reports):
        if self.rank == 0:
            self.reports = reports

    def _issue(self, handle, cycle, op, value):
        action = (self.epoch, self.serial + 1, handle, cycle, op, value)
        self._apply(action)

    def _apply(self, action):
        epoch, serial, handle, cycle, op, value = action
        if epoch < self.epoch:
            return
        assert epoch == self.epoch and serial == self.serial + 1, 'PP linker action order'
        self.serial = serial
        w = self.w
        if op == 'offload':
            head, successes = value
            assert head == self.offload_head
            assert len(successes) <= len(self.offload_results)
            w.commit_completed_offloads(successes)
            del self.offload_results[:len(successes)]
            self.offload_head += len(successes)
        elif op == 'unhold':
            # The request may have been admitted (entry released) by now: drop the
            # hold if one is still there, nothing else.
            w._unhold(handle)
        else:
            entry = self.entries[handle]
            assert entry.cycle == cycle, 'stale PP linker request generation'
            if op == 'load':
                assert entry.phase == 'lookup'
                full = None
                if isinstance(value, tuple) and len(value) == 3:
                    value, ahead, full = value  # CAL: load a piece now, restore up to `full` pages
                else:
                    value, ahead = value if isinstance(value, tuple) else (value, None)
                if full is not None and handle in w.lookups and (full or handle in w._cal):
                    # full == 0 for a CAL request: its chain is gone; the target shrinks to the held prefix
                    w.cal_enter(handle, w.lookups[handle].start + full * w.cache.page_size)
                lookup = w.lookups.pop(handle, None)
                if value and lookup is not None:
                    w._draining_lookups = True
                    try:
                        w._start_prefetch_load(handle, lookup.key, lookup.hashes,
                            lookup.start, lookup.anchor_node_id, lookup.anchor_lock,
                            [value], lookup.timeout_s,
                            **({} if ahead is None else {'ahead': ahead}))
                    finally:
                        w._draining_lookups = False
                    entry.tokens = value * w.cache.page_size
                elif lookup is not None:
                    # Metadata-only RPCs never own private device writes.
                    lookup.future.cancel()
                    w.cache.dec_lock_ref(lookup.anchor_node_id, lookup.anchor_lock)
                entry.phase = 'load' if value else 'ready'
                if entry.phase == 'ready' and entry.released:
                    del self.entries[handle]
            elif op == 'park':
                assert entry.phase == 'lookup'
                w._park_lookup(handle, value)
                entry.phase = 'ready'
                if entry.released:
                    del self.entries[handle]
            elif op == 'finish':
                assert entry.phase == 'load'
                w._finish_prefetch(handle, publish=value)
                if value:
                    assert handle in w.holds, 'PP linker published without stage hold'
                entry.phase = 'ready'
                if entry.released:
                    del self.entries[handle]
            else:
                raise AssertionError(op)
        if self.rank + 1 < self.size:
            self.outbox.append(action)

    def poll(self):
        if self.polling:
            return
        self.polling = True
        try:
            if self.rank:
                actions, self.inbox = self.inbox, []
                for action in actions:
                    self._apply(action)
                return
            if len(self.reports) != self.size:
                return
            reports = dict(self.reports)
            reports[0] = self.report()
            # A vote predating an action cannot authorize another hold or
            # offload unlock. All stages acknowledge the command batch first.
            if any(r[0] != self.epoch or r[4] != self.serial for r in reports.values()):
                return
            hold_remaining = {rank: r[5] for rank, r in reports.items()}
            rooms = [r[6] for r in reports.values() if len(r) > 6]
            heads = [r[2] for r in reports.values()]
            if all(head == self.offload_head for head in heads):
                count = min(len(r[3]) for r in reports.values())
                if count:
                    successes = [all(r[3][i] for r in reports.values()) for i in range(count)]
                    self._issue(None, 0, 'offload', (self.offload_head, successes))
            w = self.w
            near_term = w.near_term_window is not None
            # Device tokens every out-of-window lookup on PP0 keeps anchored, RPC done or not.
            anchored = sum(
                lk.start for h, lk in w.lookups.items() if not w.in_near_term_window(h)
            ) if near_term else 0
            held = []
            for handle, entry in list(self.entries.items()):
                votes = [r[1].get(handle) for r in reports.values()]
                if any(v is None or v[:2] != (entry.cycle, entry.phase) for v in votes):
                    continue
                values = [v[2] for v in votes]
                if not all(v[0] for v in values):
                    continue
                if entry.phase == 'lookup':
                    signatures = [v[1] for v in values]
                    common = set(values[0][2])
                    for value in values[1:]:
                        common.intersection_update(value[2])
                    if all(signatures) and not entry.stopped:
                        assert all(s == signatures[0] for s in signatures), 'PP linker tree/key mismatch'
                        common = {n for n in common if 0 < n <= signatures[0][2]}
                    else:
                        common.clear()
                    value = max(common, default=0)
                    entry.agreed = value
                    if value and not w.in_near_term_window(handle):
                        held.append((handle, entry.cycle, value))
                        continue  # Hit agreed; its load starts once PP0 plans it near-term.
                    if value and near_term and w.stale_hit(handle):
                        # PP0's clock decides, every stage parks in action order: the
                        # re-query at admission looks the chain up afresh.
                        self._issue(handle, entry.cycle, 'park', value)
                        continue
                    load = value
                    if near_term:
                        # Every stage reserves PP0's work ahead of the request plus the
                        # evictable memory out-of-window anchors may still pin.
                        allowance = max(w.near_term_anchor_cap - anchored, 0)
                        if w.head_load_skips_allowance and handle == w.near_term_head:
                            allowance = 0  # F6c7b: the head's room is the adder's committed number
                        if w.window_load_skips_allowance and handle in w.near_term_window:
                            allowance = 0  # F6c7c: every window member (work ahead still applies)
                        load = (value, w.near_term_window.get(handle, 0) + allowance)
                    if w.cal_enabled and (value or handle in w._cal) and len(rooms) == len(reports):
                        # CAL, PP0 only: size the load by the tightest stage's room (relayed; every stage
                        # applies the same piece). A piece smaller than the minimum waits for room.
                        page = w.cache.page_size
                        ahead_tokens = load[1] if isinstance(load, tuple) else None
                        need = w.load_free_reserve_tokens
                        if handle == w.near_term_head and w.near_term_head_reserve:
                            need = min(need, w.near_term_head_reserve)
                        need = max(need, w.chunk_tail_tokens + (ahead_tokens or 0))
                        fit = max(0, (min(rooms) - need) // page)
                        if value > fit or handle in w._cal or (w.cal_force and value * page > w.cal_max_piece_tokens):
                            piece = min(value, fit, w.cal_max_piece_tokens // page)
                            if piece < value and piece * page < w.cal_min_piece_tokens:
                                piece = 0
                            load = (piece, ahead_tokens, value)
                    self._issue(handle, entry.cycle, 'load', load)
                elif entry.phase == 'load':
                    publish = all(v[1] for v in values) and all(
                        room >= entry.tokens for room in hold_remaining.values())
                    if publish:
                        for rank in hold_remaining:
                            hold_remaining[rank] -= entry.tokens
                    self._issue(handle, entry.cycle, 'finish', publish)
            while held and anchored > w.near_term_anchor_cap:
                handle, cycle, value = held.pop()  # newest first; the oldest are nearest the window
                lookup = w.lookups.get(handle)
                anchored -= lookup.start if lookup is not None else 0
                self._issue(handle, cycle, 'park', value)
            if near_term and w.window_load_skips_allowance:
                # F6c7c: window loads no longer leave the anchor allowance, so PP0 keeps the head
                # its work by parking the newest out-of-window anchors (their prefixes go back to
                # evictable); a relayed decision, never a per-stage one.
                room = (w.cache.token_to_kv_pool_allocator.available_size()
                        + w.cache.evictable_size())
                need = w.chunk_tail_tokens + w.near_term_head_work
                while held and room < need:
                    handle, cycle, value = held.pop()
                    lookup = w.lookups.get(handle)
                    room += lookup.start if lookup is not None else 0
                    self._issue(handle, cycle, 'park', value)
                    w._lookup_stats["head_room_park"] += 1
                if room < need:
                    # Out-of-window lookups whose RPC is past its timeout (hung) still pin anchors,
                    # and no other relayed path releases them while the head is blocked: end them
                    # with 'load 0' (every stage drops lookup + anchor). Lookups still within their
                    # timeout are left to agree; an agreed hit is parked above and keeps its hit.
                    pending = [(h, e) for h, e in self.entries.items()
                               if e.phase == 'lookup' and not e.stopped and not e.agreed
                               and not w.in_near_term_window(h) and w.lookup_overdue(h)]
                    pending.sort(key=lambda he: -(w.lookups[he[0]].issued_at if he[0] in w.lookups else 0.0))
                    for handle, entry in pending:
                        if room >= need:
                            break
                        lookup = w.lookups.get(handle)
                        room += lookup.start if lookup is not None else 0
                        self._issue(handle, entry.cycle, 'load', 0)
                        w._lookup_stats["head_room_release"] += 1
        finally:
            self.polling = False

    def progress(self, handle, local_progress):
        entry = self.entries.get(handle)
        verdict = 2 if entry is None or entry.phase == 'ready' else 0
        if self.rank == 0:
            if (verdict == 2 and self.w.near_term_window is not None
                    and self.w.parked_out_of_window(handle)):
                verdict = 0  # A parked hit waits for the window, then re-queries.
            elif verdict != 2 and local_progress(handle):
                verdict = 1  # PP0 alone runs the supplier's timeout/stop policy.
        verdict = self.w._pp_verdict(verdict)
        if verdict == 1:
            assert entry is not None
            entry.stopped = True
            self.w._cancel_prefetch(handle)
        return verdict != 0

    def stall_release(self, head):
        """PP0 liveness backstop: park every agreed out-of-window hit and drop
        every non-head hold, on every stage in action order."""
        w = self.w
        for handle, entry in list(self.entries.items()):
            # A stopped cycle settles by PP0's own load 0; parking it here (PP0 at the
            # loop hook, stages 1-3 in poll) would let PP0 alone reopen it.
            if (entry.phase == 'lookup' and entry.agreed and not entry.stopped
                    and not w.in_near_term_window(handle)):
                self._issue(handle, entry.cycle, 'park', entry.agreed)
            elif entry.phase == 'ready' and handle != head and handle in w.holds:
                self._issue(handle, entry.cycle, 'unhold', 0)

    def requery(self, req, local_requery):
        decision = local_requery(req) if self.rank == 0 else False
        decision = self.w._pp_verdict(int(decision))
        if self.rank and decision:
            # Same device tree and target on every stage; fail closed on a
            # divergent tree instead of silently choosing a different batch.
            assert int(local_requery(req)) == decision, 'PP linker re-query state mismatch'
        return decision  # 0 no, 1 re-query, 2 wait for the cycle to settle

    def reset(self):
        # Backend reset/close must have quiesced writers before this call.
        self.epoch += 1
        self.entries.clear()
        self.reports.clear()
        self.inbox.clear()
        self.outbox.clear()
        self.offload_results.clear()
        self.offload_head = 0
