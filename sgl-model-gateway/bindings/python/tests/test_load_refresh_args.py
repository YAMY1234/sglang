"""CPU-only CLI/native boundary checks for independent load monitoring."""

import argparse
import inspect

import pytest
from sglang_router.router import Router
from sglang_router.router_args import RouterArgs
from sglang_router.sglang_router_rs import Router as NativeRouter


def test_defaults_preserve_actual_refresh_and_disable_trace():
    args = RouterArgs()
    assert args.load_refresh_interval_secs == 30
    assert args.score_trace is False
    signature = inspect.signature(NativeRouter)
    assert signature.parameters["load_refresh_interval_secs"].default == 30
    assert signature.parameters["score_trace"].default is False
    # Existing positional arguments remain first, and construction does not start workers.
    NativeRouter(["http://127.0.0.1:1"])
    Router.from_args(args)


@pytest.mark.parametrize("prefix", [False, True])
def test_refresh_and_trace_round_trip_to_native_without_starting_workers(prefix):
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=prefix)
    flag = "--router-" if prefix else "--"
    parsed = parser.parse_args(
        [
            flag + "worker-startup-check-interval",
            "7",
            flag + "load-refresh-interval-secs",
            "1",
            flag + "score-trace",
        ]
    )
    args = RouterArgs.from_cli_args(parsed, use_router_prefix=prefix)
    assert args.worker_startup_check_interval == 7
    assert args.load_refresh_interval_secs == 1
    assert args.score_trace is True
    Router.from_args(args)
