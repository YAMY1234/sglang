"""Least-outstanding-requests balancer over identical SGLang servers (data parallel: one server per GPU or TP group).
lm-eval talks to one URL; every /generate goes to the backend with the fewest requests in flight, so long generations
(AIME, thinking) do not pile up on one server the way round-robin would let them.  A backend whose connection fails is taken
out of rotation until its /health answers 200 again (checked every HEALTH_PERIOD s); a dead server would otherwise have the
fewest requests in flight and draw every new request.

--resume S1.jsonl[,S2...]: the sidecars (model.py) of an interrupted run of the same repetitions -- e.g. a job killed at its time
limit; lm-eval writes its samples only at the end of a repetition.  A /generate whose prompt, occurrence, sampling seed and
budget match a
reply such a run received is answered with that reply instead of being generated again (the seed fixes the random stream, so
the replayed reply is the sample the rerun would draw, up to the batch-variance of the kernels that already separates any two
runs); only the missing replies reach the servers.  The response carries meta_info.duet_resumed = the original request id, which
model.py records in the new sidecar.

    python -m benchmark.duet_eval.lmeval.lb --port 31001 --backends http://127.0.0.1:30601,http://127.0.0.1:30602 [--resume ...]

GET /health      200 when every backend is healthy        POST /generate    forwarded (least outstanding) or replayed (--resume)
POST /flush_cache  broadcast to all backends               GET /stats        per-backend counters, replayed replies
GET /get_server_info, /model_info  forwarded to the first backend"""
import argparse
import asyncio
import json
import time

import aiohttp
from aiohttp import web

from .common import ids_key

HEALTH_PERIOD = 30.0       # arbitrary; seconds between health checks of the backends out of rotation


def load_resume(paths):
    """sidecar records with a reply -> {(prompt key, occurrence, sampling seed, max_new_tokens): record}; model.py's request id
    is <pid>-<prompt key>-<occurrence>[-retry<n>].  A line cut by the kill that interrupted the run is skipped."""
    store, cut = {}, 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    cut += 1
                    continue
                if r.get("text_raw") is None or not r.get("rid") or r.get("finish") is None:
                    continue
                _, key, occurrence = r["rid"].split("-")[:3]
                store[(key, occurrence, r.get("seed"), r.get("max_new_tokens"))] = r
    return store, cut


class Balancer:
    def __init__(self, backends, resume=None):
        self.backends = backends
        self.resume = resume or {}
        self.replayed = 0
        self.outstanding = {b: 0 for b in backends}
        self.total = {b: 0 for b in backends}
        self.failed = {b: 0 for b in backends}
        self.down = set()
        self.started = time.time()
        self.session = None
        self.watcher = None

    async def start(self, app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                             connector=aiohttp.TCPConnector(limit=0))
        self.watcher = asyncio.create_task(self.readmit())

    async def stop(self, app):
        self.watcher.cancel()
        await self.session.close()

    async def readmit(self):
        while True:
            await asyncio.sleep(HEALTH_PERIOD)
            for b in sorted(self.down):
                try:
                    async with self.session.get(b + "/health", timeout=aiohttp.ClientTimeout(total=10)) as r:
                        if r.status == 200:
                            self.down.discard(b)
                            print(f"backend {b} healthy again, back in rotation", flush=True)
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass

    def pick(self):
        live = [b for b in self.backends if b not in self.down] or self.backends
        return min(live, key=lambda b: (self.outstanding[b], self.total[b]))

    def replay(self, body):
        """the stored reply of this request (--resume), in SGLang's /generate response format, or None"""
        try:
            payload = json.loads(body)
        except ValueError:
            return None
        ids, params, rid = payload.get("input_ids"), payload.get("sampling_params") or {}, str(payload.get("rid") or "")
        if not isinstance(ids, list) or not ids or not isinstance(ids[0], int) or rid.count("-") < 2:
            return None
        r = self.resume.get((ids_key(ids), rid.split("-")[2], params.get("sampling_seed"), params.get("max_new_tokens")))
        if r is None:
            return None
        self.replayed += 1
        return {"text": r["text_raw"], "meta_info": {"id": payload.get("rid"), "finish_reason": {"type": r["finish"]},
                                                     "prompt_tokens": r.get("prompt_tokens"),
                                                     "completion_tokens": r.get("completion_tokens"), "duet_resumed": r["rid"]}}

    async def forward(self, request):
        body = await request.read()
        if self.resume:
            stored = self.replay(body)
            if stored is not None:
                return web.json_response(stored)
        backend = self.pick()
        self.outstanding[backend] += 1
        self.total[backend] += 1
        try:
            async with self.session.post(backend + request.path_qs, data=body,
                                         headers={"Content-Type": request.headers.get("Content-Type", "application/json")}) as r:
                data = await r.read()
                if r.status >= 400:
                    self.failed[backend] += 1
                return web.Response(status=r.status, body=data, content_type=r.content_type)
        except aiohttp.ClientError as e:
            self.failed[backend] += 1
            if backend not in self.down:
                self.down.add(backend)
                print(f"backend {backend} out of rotation: {e!r}", flush=True)
            return web.Response(status=502, text=f"backend {backend} failed: {e!r}")
        finally:
            self.outstanding[backend] -= 1

    async def first(self, request):
        async with self.session.get(self.backends[0] + request.path_qs) as r:
            return web.Response(status=r.status, body=await r.read(), content_type=r.content_type)

    async def health(self, request):
        states = {}
        for b in self.backends:
            try:
                async with self.session.get(b + "/health", timeout=aiohttp.ClientTimeout(total=10)) as r:
                    states[b] = r.status
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                states[b] = repr(e)
        ok = all(s == 200 for s in states.values())
        return web.json_response(states, status=200 if ok else 503)

    async def broadcast(self, request):
        body = await request.read()
        states = {}
        for b in self.backends:
            try:
                async with self.session.post(b + request.path_qs, data=body) as r:
                    states[b] = r.status
            except aiohttp.ClientError as e:
                states[b] = repr(e)
        return web.json_response(states)

    async def stats(self, request):
        return web.json_response({"uptime_s": round(time.time() - self.started), "backends": self.backends,
                                  "outstanding": self.outstanding, "total": self.total, "failed": self.failed,
                                  "down": sorted(self.down), "resume_store": len(self.resume), "replayed": self.replayed})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backends", required=True, help="comma-separated base URLs")
    ap.add_argument("--resume", default="", help="comma-separated sidecar files of an interrupted run (see above)")
    args = ap.parse_args()
    store, cut = load_resume([p for p in args.resume.split(",") if p])
    lb = Balancer([b.strip().rstrip("/") for b in args.backends.split(",") if b.strip()], store)
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(lb.start)
    app.on_cleanup.append(lb.stop)
    app.router.add_post("/generate", lb.forward)
    app.router.add_post("/flush_cache", lb.broadcast)
    app.router.add_get("/health", lb.health)
    app.router.add_get("/stats", lb.stats)
    app.router.add_get("/get_server_info", lb.first)
    app.router.add_get("/model_info", lb.first)
    print(json.dumps({"lb": f"http://{args.host}:{args.port}", "backends": lb.backends,
                      **({"resume_store": len(store), "resume_cut_lines": cut} if args.resume else {})}), flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
