"""Stdlib host gates for the real bounded worker and real serving entry method."""

import ast
import asyncio
import contextvars
import importlib.util
import logging
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OPENAI = ROOT / "python/sglang/srt/entrypoints/openai"
spec = importlib.util.spec_from_file_location(
    "conversion_worker", OPENAI / "conversion_worker.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ConversionWorker = module.ConversionWorker
CapacityError = module.ConversionCapacityError


async def wait_until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


class SettingsTests(unittest.TestCase):
    def test_explicit_settings_and_invalid_startup(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.conversion_worker_settings(), (False, 128))
            os.environ["SGLANG_OPENAI_CHAT_CONVERSION_WORKER"] = "1"
            os.environ["SGLANG_OPENAI_CHAT_CONVERSION_MAX_PENDING"] = "48"
            self.assertEqual(module.conversion_worker_settings(), (True, 48))
            for value in ("", "true", "all", "2"):
                os.environ["SGLANG_OPENAI_CHAT_CONVERSION_WORKER"] = value
                with self.assertRaises(ValueError):
                    module.conversion_worker_settings()
            os.environ["SGLANG_OPENAI_CHAT_CONVERSION_WORKER"] = "0"
            for value in ("0", "-1", "4097", "1.0", "x"):
                os.environ["SGLANG_OPENAI_CHAT_CONVERSION_MAX_PENDING"] = value
                with self.assertRaises(ValueError):
                    module.conversion_worker_settings()


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_and_value_preserved(self):
        request_id = contextvars.ContextVar("request_id")
        request_id.set("request-a")
        worker = ConversionWorker(
            lambda value: (value, request_id.get(), threading.get_ident()), 2
        )
        try:
            value, rid, thread = await worker.run({"ids": [1, 2, 3]})
            self.assertEqual((value, rid), ({"ids": [1, 2, 3]}, "request-a"))
            self.assertNotEqual(thread, threading.get_ident())
        finally:
            await worker.aclose()

    async def test_cancel_running_retains_capacity_until_native_completion(self):
        entered, release = threading.Event(), threading.Event()

        def convert():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test barrier")
            return "converted"

        worker = ConversionWorker(convert, 1)
        task = asyncio.create_task(worker.run())
        try:
            await wait_until(entered.is_set)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(worker.pending, 1)
            with self.assertRaises(CapacityError):
                await worker.run()
            release.set()
            await wait_until(lambda: worker.pending == 0)
            self.assertEqual(await worker.run(), "converted")
        finally:
            release.set()
            await worker.aclose()

    async def test_cancel_queued_never_executes_and_frees_its_capacity(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def convert(value):
            calls.append(value)
            if value == "first":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test barrier")
            return value

        worker = ConversionWorker(convert, 2)
        first = asyncio.create_task(worker.run("first"))
        try:
            await wait_until(entered.is_set)
            second = asyncio.create_task(worker.run("cancelled"))
            await wait_until(lambda: worker.pending == 2)
            with self.assertRaises(CapacityError):
                await worker.run("over-capacity")
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
            await wait_until(lambda: worker.pending == 1)
            release.set()
            self.assertEqual(await first, "first")
            self.assertEqual(await worker.run("third"), "third")
            self.assertEqual(calls, ["first", "third"])
        finally:
            release.set()
            await worker.aclose()

    async def test_exception_preserved_and_capacity_recovered(self):
        def convert():
            raise ValueError("original template error")

        worker = ConversionWorker(convert, 1)
        try:
            with self.assertRaisesRegex(ValueError, "original template error"):
                await worker.run()
            await wait_until(lambda: worker.pending == 0)
        finally:
            await worker.aclose()

    async def test_close_rejects_new_work_and_waits_for_running_conversion(self):
        entered, release = threading.Event(), threading.Event()

        def convert():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test barrier")

        worker = ConversionWorker(convert, 2)
        first = asyncio.create_task(worker.run())
        try:
            await wait_until(entered.is_set)
            closing = asyncio.create_task(worker.aclose())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            with self.assertRaises(CapacityError):
                await worker.run()
            release.set()
            await first
            await closing
            await worker.aclose()  # Idempotent cleanup.
        finally:
            release.set()
            await worker.aclose()


class GenerateReqInput:
    def __init__(self, value):
        self.value = value


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


def actual_method(filename, name, namespace):
    tree = ast.parse((OPENAI / filename).read_text())
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(candidates) == 1, name
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            candidates[0],
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(tree), str(OPENAI / filename), "exec"),
        namespace,
    )
    return namespace[name]


NAMESPACE = dict(
    ConversionCapacityError=CapacityError,
    GenerateReqInput=GenerateReqInput,
    EmbeddingReqInput=type("EmbeddingReqInput", (), {}),
    HTTPException=HTTPException,
    DS32EncodingError=type("DS32EncodingError", (Exception,), {}),
    SimpleNamespace=SimpleNamespace,
    monotonic_time=time.monotonic,
    logger=logging.getLogger("conversion-worker-test"),
)


class EntryHarness:
    handle_request = actual_method("serving_base.py", "handle_request", NAMESPACE)
    _convert_in_worker = actual_method(
        "serving_chat.py", "_convert_in_worker", NAMESPACE
    )

    def __init__(self, convert, enabled):
        self.convert = convert
        self.tokenizer_manager = SimpleNamespace(
            request_logger=SimpleNamespace(log_requests=False),
            tokenizer=SimpleNamespace(chat_template="template-a"),
        )
        self._conversion_source_tokenizer = self.tokenizer_manager.tokenizer
        self._conversion_source_template = "template-a"
        self._conversion_worker = (
            ConversionWorker(self._convert_to_internal_request, 1) if enabled else None
        )
        self.dispatched = []

    def _validate_request(self, request):
        return None

    def _convert_to_internal_request(self, request, raw_request):
        return GenerateReqInput(self.convert(request, raw_request)), request

    async def _handle_non_streaming_request(self, adapted, processed, raw):
        self.dispatched.append(adapted.value)
        return adapted.value

    _handle_streaming_request = _handle_non_streaming_request

    def create_error_response(self, **kwargs):
        return kwargs


class EntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_stays_synchronous_and_enabled_uses_header_view(self):
        def convert(request, raw):
            return (threading.get_ident(), request.payload, raw.headers["x-request-id"])

        request = SimpleNamespace(stream=False, payload=[1, 2, 3])
        raw = SimpleNamespace(headers={"x-request-id": "r1"})
        off = EntryHarness(convert, False)
        on = EntryHarness(convert, True)
        try:
            baseline = await off.handle_request(request, raw)
            candidate = await on.handle_request(request, raw)
            self.assertEqual(baseline[0], threading.get_ident())
            self.assertNotEqual(candidate[0], baseline[0])
            self.assertEqual(candidate[1:], baseline[1:])
        finally:
            await on._conversion_worker.aclose()

    async def test_cancelled_entry_does_not_dispatch_after_background_finishes(self):
        entered, release = threading.Event(), threading.Event()

        def convert(request, raw):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test barrier")
            return [7, 8]

        handler = EntryHarness(convert, True)
        task = asyncio.create_task(
            handler.handle_request(SimpleNamespace(stream=True), None)
        )
        try:
            await wait_until(entered.is_set)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(handler._conversion_worker.pending, 1)
            release.set()
            await wait_until(lambda: handler._conversion_worker.pending == 0)
            self.assertEqual(handler.dispatched, [])
        finally:
            release.set()
            await handler._conversion_worker.aclose()

    async def test_original_error_mapping_is_preserved(self):
        for error, expected in (
            (ValueError("invalid template"), 400),
            (HTTPException(422, "invalid header"), 422),
        ):

            def convert(request, raw):
                raise error

            handler = EntryHarness(convert, True)
            try:
                response = await handler.handle_request(
                    SimpleNamespace(stream=False), None
                )
                self.assertEqual(response["status_code"], expected)
                self.assertEqual(handler.dispatched, [])
            finally:
                await handler._conversion_worker.aclose()

    async def test_changed_tokenizer_template_fails_before_dispatch(self):
        handler = EntryHarness(lambda *args: [1], True)
        try:
            handler.tokenizer_manager.tokenizer.chat_template = "template-b"
            response = await handler.handle_request(SimpleNamespace(stream=False), None)
            self.assertEqual(response["status_code"], 400)
            self.assertEqual(handler.dispatched, [])
        finally:
            await handler._conversion_worker.aclose()


if __name__ == "__main__":
    unittest.main()
