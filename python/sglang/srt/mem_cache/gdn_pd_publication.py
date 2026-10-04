"""Opt-in P full-N publication after its forward, with mandatory reader joins.

Only one publication may own the graph's static inputs at a time. This does not
reorder the PD scheduler or wait indefinitely for a future request: an early
reader drains the queued work immediately.
"""
import logging

import torch

logger = logging.getLogger(__name__)


class CudaPublicationRuntime:
    def __init__(self, device):
        self.device = device
        # CUDA's lowest priority is 0. A default main stream may be equal;
        # ordering comes from events, not from a priority/preemption assumption.
        self.stream = torch.cuda.Stream(device=device, priority=0)

    def launch_after_forward(self, operation, inputs):
        forward_done = torch.cuda.Event()
        forward_done.record(torch.cuda.current_stream(self.device))
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(forward_done)
            try:
                operation()
            finally:
                # Even failed enqueue paths must not release live inputs while
                # already submitted device work is still using their storage.
                for tensor in inputs:
                    if tensor is not None and tensor.is_cuda:
                        tensor.record_stream(self.stream)
                done.record(self.stream)
        return done

    def join(self, done):
        torch.cuda.current_stream(self.device).wait_event(done)


class PDBatchPublication:
    def __init__(self, pool, runtime=None):
        self.pool = pool
        self.runtime = runtime
        self.pending = None
        self.ticket = None
        self.failed = None
        self.stats = dict(submitted=0, launched=0, joins=0, early_reader=0, rows=0)

    def submit(self, graph, plan, states, controls, *, eager, policy):
        if self.pending is not None:
            raise RuntimeError("PD publication inputs reused before reader join")
        if graph.include_tail or not graph.warmed:
            raise RuntimeError("PD deferred publication requires a prewarmed full-N graph")
        states, controls = tuple(states), tuple(controls)
        self.pending = graph, plan, states, controls, eager, policy
        self.stats["submitted"] += 1
        self.stats["rows"] += plan.slots.numel()

    def start_after_forward(self):
        if self.failed is not None:
            raise RuntimeError("PD publication failed; state is not publishable") from self.failed
        if self.pending is None or self.ticket is not None:
            return
        graph, plan, states, controls, eager, policy = self.pending
        if self.runtime is None:
            self.runtime = CudaPublicationRuntime(self.pool.a.device)
        inputs = [t for pair in states for t in pair]
        inputs.extend(controls)
        inputs.extend((plan.slots, plan.ring_dst, plan.dense_required_after_commit))
        try:
            self.ticket = self.runtime.launch_after_forward(
                lambda: graph.run(self.pool, plan, states, *controls,
                                  eager=eager, policy=policy), inputs)
        except BaseException as error:
            self.failed = error
            # Retain all inputs and fail every subsequent reader closed.
            raise
        self.stats["launched"] += 1

    def join(self):
        if self.pending is None:
            return
        if self.ticket is None:
            self.stats["early_reader"] += 1
            self.start_after_forward()
        self.runtime.join(self.ticket)
        self.stats["joins"] += 1
        self.pending = self.ticket = None


def install(pool):
    pool._pd_batch_publication = PDBatchPublication(pool)
    logger.info("PD full-N deferred publication enabled: after-forward event, "
                "side priority=0; all factor readers join; early reader drains immediately")
