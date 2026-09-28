"""CPU contract tests; GPU routing and true MTP still need deployment validation."""

import asyncio
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/disaggregation"


def load(name):
    fullname = "sglang.srt.disaggregation." + name
    spec = importlib.util.spec_from_file_location(fullname, SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


policy = load("q35_request_balancer")
adapter = load("q35_dynamo_loads")
proxy = load("q35_running_request_router")


def samples(counts, timestamp=100):
    return [
        {
            "dp_rank": r,
            "num_running_reqs": n,
            "num_waiting_reqs": 0,
            "timestamp": timestamp,
        }
        for r, n in enumerate(counts)
    ]


def balancer(counts=(0, 0, 0, 0)):
    b = policy.LoadBalancer(workers=1)
    b.update_loads(50, samples(counts), 100)
    b.set_policy("balanced", "B1")
    return b


def test_burst_accounts_for_admission_before_next_snapshot():
    b = balancer((36, 58, 50, 40))
    admissions = [b.choose(str(i), None, False, 100) for i in range(100)]
    scores = [r["score"] for r in b.status(100)["ranks"]]
    assert max(scores) - min(scores) <= 1
    assert b.inflight == 100
    for a in admissions:
        b.release(a)
    assert b.inflight == 0 and not any(b.local.values())
    with pytest.raises(RuntimeError):
        b.release(admissions[0])


def test_snapshot_does_not_double_charge_existing_requests():
    b = balancer()
    active = [b.choose(str(i), None, False, 100) for i in range(8)]
    b.update_loads(50, samples((2, 2, 2, 2), 101), 101)
    assert [r["score"] for r in b.status(101)["ranks"]] == [2] * 4
    b.release(active[0])
    assert b.status(101)["ranks"][0]["score"] == 1


def test_sticky_session_parent_and_dynamo_owned_session():
    b = balancer()
    first = b.choose("parent", None, False, 100)
    b.release(first)
    b.update_loads(50, samples((999, 0, 0, 0)), 100)
    same = b.choose("parent", None, False, 100)
    child = b.choose("child", "parent", False, 100)
    assert same.target == child.target == first.target
    unknown = b.choose("other", "unseen", False, 100)
    later = b.choose("other", None, False, 100)
    assert not unknown.inject and not later.inject


def test_existing_stock_sessions_not_taken_over_and_no_live_switch():
    b = balancer()
    b.set_policy("stock", "A1")
    a = b.choose("old", None, False, 100)
    with pytest.raises(RuntimeError):
        b.set_policy("balanced", "B1")
    b.release(a)
    b.set_policy("balanced", "B1")
    assert not b.choose("old", None, False, 100).inject
    assert b.choose("new", None, False, 100).inject


def test_stale_snapshot_constraints_and_worker_restart():
    b = balancer((1, 2, 3, 4))
    assert not b.choose("stale", None, False, 104).inject
    assert not b.choose("pinned", None, True, 100).inject
    with pytest.raises(ValueError, match="identity"):
        b.update_loads(51, samples((1, 2, 3, 4)), 100)
    with pytest.raises(ValueError, match="Incomplete"):
        b.update_loads(50, samples((1, 2)), 100)
    # Idle snapshots can stay unchanged while scheduler sleeps; fresh endpoint
    # polling and local reservations are still required.
    b = balancer()
    b.update_loads(50, samples((0, 0, 0, 0), 1), 100)
    assert b.fresh(100)
    assert not b.fresh(103)


def test_adapter_refuses_wrong_source_and_installs_once(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text(
        "class H:\n    def register(self, runtime):\n"
        '        runtime.register_engine_route("control/start_profile", self.start_profile)\n'
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="mismatch"):
        adapter.install(path, "wrong")
    proof = adapter.install(path, sha)
    assert proof["before"] == sha and proof["after"] != sha
    with pytest.raises(RuntimeError, match="mismatch"):
        adapter.install(path, sha)


@pytest.mark.asyncio
async def test_streaming_proxy_pins_match_response_and_honors_alias(tmp_path):
    requests = []

    async def generate(request):
        body = await request.json()
        requests.append((body, dict(request.headers)))
        ext = body["nvext"]
        data = {
            "choices": [{"delta": {"content": "ok"}}],
            "nvext": {
                "worker_id": {
                    "decode_worker_id": ext.get("decode_worker_id", 50),
                    "decode_dp_rank": ext.get("dp_rank", 2),
                }
            },
        }
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        raw = b"data: " + json.dumps(data).encode() + b"\n\ndata: [DONE]\n\n"
        for pos in range(0, len(raw), 7):
            await response.write(raw[pos : pos + 7])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", generate)
    async with TestServer(app) as upstream:
        router = proxy.Router(
            str(upstream.make_url("/")), [], tmp_path / "routes.jsonl", workers=1
        )
        router.balancer.update_loads(
            50, samples((0, 0, 0, 0), time.time()), time.time()
        )
        async with TestClient(TestServer(router.app())) as client:
            res = await client.post(
                "/control/policy", json={"mode": "balanced", "epoch": "B1"}
            )
            assert res.status == 200
            original = {
                "model": "q35",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            }
            for _ in range(2):
                res = await client.post(
                    "/v1/chat/completions",
                    json=original,
                    headers={"x-dynamo-session-id": "session"},
                )
                assert res.status == 200 and b"[DONE]" in await res.read()
            assert requests[0][0]["messages"] == original["messages"]
            assert (
                requests[0][0]["nvext"]["dp_rank"] == requests[1][0]["nvext"]["dp_rank"]
            )
            assert "prefill_worker_id" not in requests[0][0]["nvext"]
            res = await client.post(
                "/v1/chat/completions",
                json=original,
                headers={"x-worker-instance-id": "50", "x-dp-rank": "2"},
            )
            await res.read()
            assert "decode_worker_id" not in requests[-1][0]["nvext"]
            assert router.balancer.inflight == 0
    records = [
        json.loads(s) for s in (tmp_path / "routes.jsonl").read_text().splitlines()
    ]
    routed = [r for r in records if r["event"] == "request" and r["injected"]]
    assert len(routed) == 2 and all(r["matched"] for r in routed)


@pytest.mark.asyncio
async def test_client_disconnect_releases_reservation(tmp_path):
    active = asyncio.Event()

    async def generate(request):
        await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices": []}\n\n')
        active.set()
        await asyncio.sleep(30)
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", generate)
    async with TestServer(app, handler_cancellation=True) as upstream:
        router = proxy.Router(
            str(upstream.make_url("/")), [], tmp_path / "cancel.jsonl", workers=1
        )
        router.balancer.update_loads(
            50, samples((0, 0, 0, 0), time.time()), time.time()
        )
        router.balancer.set_policy("balanced", "B1")
        async with TestClient(
            TestServer(router.app(), handler_cancellation=True)
        ) as client:
            response = await client.post(
                "/v1/chat/completions", json={"model": "q35", "stream": True}
            )
            await active.wait()
            assert router.balancer.inflight == 1
            rejected = await client.post(
                "/control/policy", json={"mode": "stock", "epoch": "A2"}
            )
            assert rejected.status == 409
            response.close()
            for _ in range(100):
                if router.balancer.inflight == 0:
                    break
                await asyncio.sleep(0.01)
            assert router.balancer.inflight == 0
