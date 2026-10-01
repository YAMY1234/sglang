"""Bounded, optional CPU conversion outside the HTTP event loop.

The caller owns an isolated conversion/tokenizer instance. Capacity follows the
native future, so cancelling an HTTP coroutine cannot release a running slot.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Callable


class ConversionCapacityError(RuntimeError):
    pass


def conversion_worker_settings() -> tuple[bool, int]:
    enabled = os.environ.get("SGLANG_OPENAI_CHAT_CONVERSION_WORKER", "0")
    if enabled not in ("0", "1"):
        raise ValueError("SGLANG_OPENAI_CHAT_CONVERSION_WORKER must be 0 or 1")
    raw_limit = os.environ.get("SGLANG_OPENAI_CHAT_CONVERSION_MAX_PENDING", "128")
    if not raw_limit.isdecimal() or not 1 <= int(raw_limit) <= 4096:
        raise ValueError(
            "SGLANG_OPENAI_CHAT_CONVERSION_MAX_PENDING must be an integer in [1, 4096]"
        )
    return enabled == "1", int(raw_limit)


class ConversionWorker:
    def __init__(self, convert: Callable, max_pending: int):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._convert = convert
        self._max_pending = max_pending
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sglang-chat-conversion"
        )
        self._pending: set[Future] = set()
        self._loop = None
        self._closed = False

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def run(self, *args):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        if loop is not self._loop:
            raise RuntimeError("ConversionWorker cannot cross event loops")
        if self._closed:
            raise ConversionCapacityError("Chat conversion worker is shutting down")
        if len(self._pending) >= self._max_pending:
            raise ConversionCapacityError(
                "Chat conversion capacity is full; retry later"
            )

        context = contextvars.copy_context()
        future = self._executor.submit(context.run, partial(self._convert, *args))
        self._pending.add(future)
        future.add_done_callback(
            lambda done: loop.call_soon_threadsafe(self._pending.discard, done)
        )
        wrapped = asyncio.wrap_future(future, loop=loop)
        # A cancelled caller may never retrieve a late conversion exception.
        wrapped.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # This succeeds only for work that has not started. Running work
            # retains capacity until its native completion callback above.
            future.cancel()
            raise

    async def aclose(self):
        self._closed = True
        pending = tuple(self._pending)
        for future in pending:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if pending:
            await asyncio.shield(
                asyncio.gather(
                    *(asyncio.wrap_future(future) for future in pending),
                    return_exceptions=True,
                )
            )


class ConversionManagerView:
    """Read manager configuration while owning a separate tokenizer object."""

    def __init__(self, manager, tokenizer):
        self._manager = manager
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._manager, name)
