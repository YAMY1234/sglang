"""Compatibility launcher exposing the unified CLI on the pinned nightly image.

Native fork ServerArgs accepts the same options. This shim only consumes DUET
options and exports them for images predating those fields; all other arguments
are parsed by the unmodified native sglang.launch_server.
"""

import argparse
import runpy
import sys

from ._common import load

_options = load("options")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    _options.add_arguments(parser)
    args, remaining = parser.parse_known_args()
    _options.export_cli_environment(args)
    sys.argv = ["sglang.launch_server", *remaining]
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
