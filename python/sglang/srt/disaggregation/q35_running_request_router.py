"""Opt-in HTTP admission proxy for H28 + DP request balancing experiments.

Run once in front of the existing Dynamo/nginx frontend. Both experiment arms
use this proxy; only /control/policy toggles admission for new sessions.
"""

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web
from sglang.srt.disaggregation.q35_request_balancer import LoadBalancer

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}
PIN_FIELDS = {
    "backend_instance_id",
    "decode_worker_id",
    "dp_rank",
    "prefill_worker_id",
    "prefill_dp_rank",
    "routing_constraints",
    "allowed_worker_ids",
    "agent_hints",
    "router",
}
PIN_HEADERS = {
    "x-dynamo-worker-instance-id",
    "x-dynamo-dp-rank",
    "x-worker-instance-id",
    "x-dp-rank",
    "x-data-parallel-rank",
    "x-dynamo-prefill-instance-id",
    "x-prefill-instance-id",
    "x-dynamo-prefill-dp-rank",
    "x-prefill-dp-rank",
}
OTHER_SESSION_HEADERS = {"session-id", "x-session-id", "x-claude-code-session-id"}


class Router:
    def __init__(self, upstream, engines, evidence, workers=2, ranks=4):
        self.upstream = upstream.rstrip("/")
        self.engines = engines
        self.balancer = LoadBalancer(workers, ranks)
        self.evidence = Path(evidence)
        self.evidence.parent.mkdir(parents=True, exist_ok=True)
        self.log = None
        self.client = None
        self.poll_task = None
        self.sequence = 0
        self.poll_errors = {}
        self.engine_ids = {}

    def record(self, event, **fields):
        self.log.write(
            json.dumps({"event": event, "time": time.time(), **fields}) + "\n"
        )
        self.log.flush()

    async def lifecycle(self, app):
        with self.evidence.open("a", buffering=1) as self.log:
            async with ClientSession(
                timeout=ClientTimeout(total=None, sock_connect=15),
                connector=TCPConnector(limit=0),
                auto_decompress=False,
            ) as self.client:
                self.poll_task = asyncio.create_task(self.poll())
                try:
                    yield
                finally:
                    self.poll_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.poll_task
                    self.record("shutdown", status=self.balancer.status(time.time()))

    async def poll_one(self, endpoint):
        try:
            async with self.client.post(
                endpoint.rstrip("/") + "/engine/control/q35_loads",
                json={},
                timeout=ClientTimeout(total=1),
            ) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)
            worker = int(data["worker_id"])
            if endpoint in self.engine_ids and self.engine_ids[endpoint] != worker:
                raise ValueError("Engine connection identity changed")
            if (
                worker in self.engine_ids.values()
                and self.engine_ids.get(endpoint) != worker
            ):
                raise ValueError("Two endpoints returned the same worker")
            self.engine_ids[endpoint] = worker
            self.balancer.update_loads(worker, data["loads"], time.time())
            self.poll_errors.pop(endpoint, None)
        except (
            ClientError,
            TimeoutError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            self.poll_errors[endpoint] = f"{type(error).__name__}: {error}"

    async def poll(self):
        last_record = 0
        while True:
            await asyncio.gather(*(self.poll_one(e) for e in self.engines))
            if time.monotonic() - last_record >= 1:
                self.record(
                    "load",
                    status=self.balancer.status(time.time()),
                    errors=self.poll_errors,
                )
                last_record = time.monotonic()
            await asyncio.sleep(0.2)

    async def status(self, request):
        return web.json_response(
            {**self.balancer.status(time.time()), "poll_errors": self.poll_errors}
        )

    async def policy(self, request):
        # The controller is local to the frontend host, not a public API.
        if request.remote not in ("127.0.0.1", "::1"):
            raise web.HTTPForbidden(text="Local controller only")
        body = await request.json()
        if body.get("mode") == "balanced" and not self.balancer.fresh(time.time()):
            raise web.HTTPConflict(text="Incomplete or stale engine snapshots")
        if any(values[0] for values in self.balancer.loads.values()):
            raise web.HTTPConflict(text="Engine requests have not drained")
        try:
            self.balancer.set_policy(body["mode"], body["epoch"])
        except (ValueError, RuntimeError, KeyError) as error:
            raise web.HTTPConflict(text=str(error)) from error
        self.record("policy", status=self.balancer.status(time.time()))
        return await self.status(request)

    async def forward(self, request):
        raw = await request.read()
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS
        }
        # Disable upstream compression so optional worker metadata is auditable.
        headers["Accept-Encoding"] = "identity"
        admission = None
        observed = set()
        parse_buffer = b""
        session = None
        request_id = self.sequence
        self.sequence += 1
        status, error_text = None, None
        is_generation = request.method == "POST" and request.path in (
            "/v1/chat/completions",
            "/v1/completions",
        )
        if is_generation:
            try:
                body = json.loads(raw)
                ext = body.get("nvext")
                if ext is None:
                    ext = body["nvext"] = {}
                if not isinstance(ext, dict):
                    raise TypeError("nvext must be an object")
                session = (
                    request.headers.get("x-dynamo-session-id") or ""
                ).strip() or None
                parent = (
                    request.headers.get("x-dynamo-parent-session-id") or ""
                ).strip() or None
                namespace = (body.get("model"), request.headers.get("x-tenant-id"))
                session_key = (*namespace, session) if session else None
                parent_key = (*namespace, parent) if parent else None
                constrained = bool(PIN_FIELDS.intersection(ext)) or any(
                    h in request.headers for h in PIN_HEADERS | OTHER_SESSION_HEADERS
                )
                admission = self.balancer.choose(
                    session_key, parent_key, constrained, time.time()
                )
                fields = ext.get("extra_fields")
                if fields is None:
                    fields = ext["extra_fields"] = []
                if not isinstance(fields, list):
                    raise TypeError("nvext.extra_fields must be a list")
                if "worker_id" not in fields:
                    fields.append("worker_id")
                if admission.inject:
                    ext["decode_worker_id"] = admission.target.worker_id
                    ext["dp_rank"] = admission.target.dp_rank
                raw = json.dumps(body, separators=(",", ":")).encode()
            except (ValueError, TypeError, AttributeError) as error:
                if admission is not None:
                    self.balancer.release(admission)
                raise web.HTTPBadRequest(text=str(error)) from error
        try:
            async with self.client.request(
                request.method,
                self.upstream + request.path_qs,
                data=raw,
                headers=headers,
                allow_redirects=False,
            ) as upstream:
                status = upstream.status
                response = web.StreamResponse(
                    status=status,
                    headers={
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower() not in HOP_HEADERS
                    },
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if admission is not None:
                        parse_buffer += chunk
                        while b"\n" in parse_buffer:
                            line, parse_buffer = parse_buffer.split(b"\n", 1)
                            self.observe(line, observed)
                        if len(parse_buffer) > 4 * 1024 * 1024:
                            raise ValueError("Oversized response line")
                    await response.write(chunk)
                if parse_buffer:
                    self.observe(parse_buffer, observed)
                await response.write_eof()
                return response
        except BaseException as error:
            error_text = type(error).__name__ + ": " + str(error)
            raise
        finally:
            if admission is not None:
                target = admission.target
                expected = (
                    (target.worker_id, target.dp_rank) if admission.inject else None
                )
                self.balancer.release(admission)
                self.record(
                    "request",
                    request_id=request_id,
                    session=session,
                    epoch=self.balancer.epoch,
                    mode=self.balancer.mode,
                    reason=admission.reason,
                    injected=admission.inject,
                    expected=expected,
                    observed=sorted(observed),
                    matched=(expected in observed) if expected else None,
                    status=status,
                    error=error_text,
                )

    @staticmethod
    def observe(line, observed):
        if b'"worker_id"' not in line:
            return
        if line.startswith(b"data:"):
            line = line[5:].strip()
        try:
            body = json.loads(line)
            worker = body.get("nvext", {}).get("worker_id") or {}
            if (
                worker.get("decode_worker_id") is not None
                and worker.get("decode_dp_rank") is not None
            ):
                observed.add(
                    (int(worker["decode_worker_id"]), int(worker["decode_dp_rank"]))
                )
        except (ValueError, TypeError, AttributeError):
            return

    def app(self):
        app = web.Application(client_max_size=32 * 1024 * 1024)
        app.cleanup_ctx.append(self.lifecycle)
        app.router.add_get("/control/status", self.status)
        app.router.add_post("/control/policy", self.policy)
        app.router.add_route("*", "/{path:.*}", self.forward)
        return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--engine", action="append", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    router = Router(
        args.upstream, args.engine, args.evidence, len(args.engine), args.ranks
    )
    web.run_app(
        router.app(),
        host=args.host,
        port=args.port,
        access_log=None,
        handler_cancellation=True,
        # Match the existing nginx AgentX idle-connection contract. Clients
        # reuse pooled sockets after long prewarm barriers and think-time gaps.
        keepalive_timeout=600,
    )


if __name__ == "__main__":
    main()
