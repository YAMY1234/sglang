"""Opt-in read-only DP telemetry adapter for the frozen Dynamo integration."""

import hashlib
import json
from pathlib import Path


def register_load_route(handler, runtime):
    async def read_loads(body):
        snapshots = await handler.engine.tokenizer_manager.get_loads()
        return {
            "worker_id": int(handler.generate_endpoint.connection_id()),
            "loads": [
                {
                    key: getattr(snapshot, key)
                    for key in (
                        "timestamp",
                        "dp_rank",
                        "num_running_reqs",
                        "num_waiting_reqs",
                    )
                }
                for snapshot in snapshots
            ],
        }

    runtime.register_engine_route("control/q35_loads", read_loads)


def install(source_file, expected_sha):
    """Patch only an explicitly identified ephemeral Dynamo Python file.

    Called by the experiment's container setup, before importing Dynamo.
    The adapter and proxy run in both arms; scheduler/kernel code is identical.
    """
    path = Path(source_file)
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(f"Dynamo source mismatch: {digest} != {expected_sha}")
    anchor = b'        runtime.register_engine_route("control/start_profile", self.start_profile)'
    if original.count(anchor) != 1:
        raise RuntimeError("Expected exactly one engine route registration site")
    insertion = (
        b"        from sglang.srt.disaggregation.q35_dynamo_loads import register_load_route\n"
        b"        register_load_route(self, runtime)\n"
    )
    modified = original.replace(anchor, insertion + anchor)
    compile(modified, str(path), "exec")
    path.write_bytes(modified)
    return {
        "path": str(path),
        "before": digest,
        "after": hashlib.sha256(modified).hexdigest(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_file")
    parser.add_argument("expected_sha")
    args = parser.parse_args()
    print(json.dumps(install(args.source_file, args.expected_sha)))
