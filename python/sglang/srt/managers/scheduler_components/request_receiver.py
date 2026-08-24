from __future__ import annotations

import os
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Union,
)

import zmq
import torch
import torch.distributed as dist
from torch.distributed import barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.runtime_context import get_disagg, get_parallel, is_ep_scale_joiner
from sglang.srt.utils import (
    PP_PYOBJ_TAG_REQUEST,
    broadcast_pyobj,
)
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.rust_server import RustServer
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy, RustServer]
    recv_from_rpc: Optional[zmq.Socket]
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: ParallelState
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_batch: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None
    pp_pyobj_inbox: Dict[int, Deque[Any]] = field(
        default_factory=lambda: defaultdict(deque),
        init=False,
        repr=False,
        compare=False,
    )
    pp_pyobj_recv_condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
        compare=False,
    )
    pp_pyobj_io_thread: Optional[threading.Thread] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    pp_pyobj_io_errors: List[BaseException] = field(
        default_factory=list,
        init=False,
        repr=False,
        compare=False,
    )
    pp_pyobj_send_queue: Deque[bytes] = field(
        default_factory=deque,
        init=False,
        repr=False,
        compare=False,
    )
    pp_pyobj_send_condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
        compare=False,
    )

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_nvtx_method("scheduler.recv_requests")
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        # Only PP0 polls tokenizer/RPC sockets. Later pipeline stages must
        # consume every forwarded control tick; applying the local receive
        # interval there leaves one empty request envelope queued per skipped
        # iteration and eventually hides real requests behind that backlog.
        if self.recv_skipper is not None and self.ps.pp_rank == 0:
            if not self.recv_skipper.handle(self.get_last_batch()):
                return []

        recv_reqs = self._pull_raw_reqs()

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        if self.ps.pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _pull_raw_reqs(self) -> Optional[List]:
        if self.ps.pp_rank == 0:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                recv_reqs = []

                # Rust ringbuffer backend: drain the in-process ring fed by the
                # embedded Rust TokenizerManager instead of a zmq socket. Same
                # non-blocking, msgpack-decoded contract as the zmq path below.
                if envs.SGLANG_RUST_SERVER.get():
                    recv_reqs.extend(
                        self.recv_from_tokenizer.drain(self.max_recv_per_poll)
                    )
                    return recv_reqs

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = sock_recv(self.recv_from_tokenizer, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = sock_recv(self.recv_from_rpc, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                recv_reqs = self.recv_pp_control(PP_PYOBJ_TAG_REQUEST)
            else:
                recv_reqs = None
        return recv_reqs

    def recv_pp_control(self, expected_channel: int) -> Any:
        """Receive one typed PP control payload from the continuously drained inbox."""
        self._ensure_pp_control_io()
        condition = self.pp_pyobj_recv_condition
        inbox = self.pp_pyobj_inbox[expected_channel]
        diagnostic = os.environ.get("SGLANG_PP_P2P_DIAGNOSTIC") == "1"
        next_report = time.monotonic() + 5.0
        with condition:
            while not inbox:
                if self.pp_pyobj_io_errors:
                    raise RuntimeError("PP control I/O failed") from (
                        self.pp_pyobj_io_errors[0]
                    )
                condition.wait(timeout=1.0 if diagnostic else None)
                now = time.monotonic()
                if diagnostic and now >= next_report:
                    inbox_sizes = dict(
                        (key, len(value))
                        for key, value in self.pp_pyobj_inbox.items()
                    )
                    print(
                        "PP_CONTROL_WAIT "
                        f"rank={self.ps.pp_rank} channel={expected_channel} "
                        f"send_queue={len(self.pp_pyobj_send_queue)} "
                        f"inbox_sizes={inbox_sizes}",
                        flush=True,
                    )
                    next_report = now + 5.0
            return inbox.popleft()

    def send_pp_control(self, channel: int, payload: Any) -> None:
        """Queue one immutable envelope for the PP control I/O owner."""
        self._ensure_pp_control_io()
        serialized = pickle.dumps([channel, payload], protocol=pickle.HIGHEST_PROTOCOL)
        with self.pp_pyobj_send_condition:
            if self.pp_pyobj_io_errors:
                raise RuntimeError("PP control I/O failed") from (
                    self.pp_pyobj_io_errors[0]
                )
            self.pp_pyobj_send_queue.append(serialized)
            self.pp_pyobj_send_condition.notify()

    def _ensure_pp_control_io(self) -> None:
        condition = self.pp_pyobj_send_condition
        with condition:
            if self.pp_pyobj_io_thread is not None:
                return
            thread = threading.Thread(
                target=self._pp_control_io_loop,
                name=f"pp-control-io-{self.ps.pp_rank}",
                daemon=True,
            )
            object.__setattr__(self, "pp_pyobj_io_thread", thread)
            thread.start()

    def _pp_control_io_loop(self) -> None:
        """Own Gloo and exchange one deterministically paired ring frame."""
        dp_offset = self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
        rank = self.ps.pp_rank * self.ps.tp_size + dp_offset
        src = ((self.ps.pp_rank - 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset
        dst = ((self.ps.pp_rank + 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset
        group = self.world_group.cpu_group
        noop = pickle.dumps([0, None], protocol=pickle.HIGHEST_PROTOCOL)
        diagnostic = os.environ.get("SGLANG_PP_P2P_DIAGNOSTIC") == "1"
        rounds = 0
        sent_counts: Dict[int, int] = defaultdict(int)
        recv_counts: Dict[int, int] = defaultdict(int)
        next_report = time.monotonic() + 5.0

        def send_serialized(serialized: bytes) -> None:
            tensor_data = torch.frombuffer(bytearray(serialized), dtype=torch.uint8)
            tensor_size = torch.tensor([tensor_data.numel()], dtype=torch.long)
            dist.send(tensor_size, dst, group=group, tag=0)
            dist.send(tensor_data, dst, group=group, tag=1)

        def recv_serialized() -> bytes:
            tensor_size = torch.zeros(1, dtype=torch.long)
            dist.irecv(tensor_size, src=src, group=group, tag=0).wait()
            size = int(tensor_size.item())
            if size <= 0:
                raise RuntimeError(f"Invalid PP control frame size: {size}")
            tensor_data = torch.empty(size, dtype=torch.uint8)
            dist.irecv(tensor_data, src=src, group=group, tag=1).wait()
            return bytes(tensor_data.numpy())

        try:
            while True:
                with self.pp_pyobj_send_condition:
                    if self.pp_pyobj_send_queue:
                        serialized = self.pp_pyobj_send_queue.popleft()
                        is_noop = False
                        if diagnostic:
                            sent_counts[pickle.loads(serialized)[0]] += 1
                    else:
                        serialized = noop
                        is_noop = True

                # PP0 initiates each round; every other stage receives then
                # forwards.  This works for any ring size and guarantees that
                # every blocking send already has a matching receive phase.
                if self.ps.pp_rank == 0:
                    send_serialized(serialized)
                    incoming = recv_serialized()
                else:
                    incoming = recv_serialized()
                    send_serialized(serialized)

                envelope = pickle.loads(incoming)
                if (
                    not isinstance(envelope, list)
                    or len(envelope) != 2
                    or not isinstance(envelope[0], int)
                ):
                    raise RuntimeError(f"Malformed PP control envelope: {envelope!r}")
                channel, payload = envelope
                rounds += 1
                if diagnostic:
                    recv_counts[channel] += 1
                if channel != 0:
                    with self.pp_pyobj_recv_condition:
                        self.pp_pyobj_inbox[channel].append(payload)
                        self.pp_pyobj_recv_condition.notify_all()

                now = time.monotonic()
                if diagnostic and now >= next_report:
                    with self.pp_pyobj_recv_condition:
                        inbox_sizes = dict(
                            (key, len(value))
                            for key, value in self.pp_pyobj_inbox.items()
                        )
                    print(
                        "PP_CONTROL_IO "
                        f"rank={self.ps.pp_rank} rounds={rounds} "
                        f"sent={dict(sent_counts)} received={dict(recv_counts)} "
                        f"send_queue={len(self.pp_pyobj_send_queue)} "
                        f"inbox_sizes={inbox_sizes}",
                        flush=True,
                    )
                    next_report = now + 5.0

                if is_noop:
                    time.sleep(0.0005)
        except BaseException as exc:
            with self.pp_pyobj_recv_condition:
                self.pp_pyobj_io_errors.append(exc)
                self.pp_pyobj_recv_condition.notify_all()
            with self.pp_pyobj_send_condition:
                self.pp_pyobj_send_condition.notify_all()

    def _broadcast_reqs_across_ranks(self, recv_reqs: Optional[List]) -> List:
        if get_parallel().enable_dp_attention:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.ps.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.ps.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = (
                get_parallel().enable_dp_attention_local_control_broadcast
                or is_ep_scale_joiner()
            )
            if _local_ctrl:
                if self.ps.attn_tp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                if self.ps.attn_cp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
            elif self.ps.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.ps.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return recv_reqs

    def unwrap_pickle_wrapper(self, recv_reqs: Optional[List]) -> None:
        if not recv_reqs:
            return

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                req.unwrap_pickle_fields()
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                for sub_req in req:
                    sub_req.unwrap_pickle_fields()

    def _apply_mm_receiver(self, recv_reqs: List) -> List:
        # Process MM requests under EPD-disaggregation mode
        if (
            self.ps.pp_rank == 0
            and get_disagg().language_only
            and get_disagg().encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                if error_code is None:
                    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                elif isinstance(error_code, HTTPStatus):
                    status_code = error_code
                else:
                    status_code = HTTPStatus(int(error_code))
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        # Unwrap shared memory features AFTER all broadcasts complete,
        # so that ShmPointerMMData metadata (not full tensor data) is what
        # gets serialized during broadcast_pyobj.
        if recv_reqs:
            if self.model_config.is_multimodal and has_shm_features(recv_reqs):
                # The broadcast source returns with its original objects while
                # peer ranks may still be unpickling ShmPointerMMData
                # (-> shm_open).  Synchronize the same CPU groups that carried
                # SHM-backed work requests before materialize() unlinks them.
                if get_parallel().enable_dp_attention:
                    if self.ps.attn_tp_size > 1:
                        barrier(group=self.attn_tp_cpu_group)
                    if self.ps.attn_cp_size > 1:
                        barrier(group=self.attn_cp_cpu_group)
                elif self.ps.tp_size > 1:
                    barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs
