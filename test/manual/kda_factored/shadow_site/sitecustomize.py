"""Opt-in, test-process-only shadow instrumentation; never imported by serving defaults."""

import os

if os.environ.get("KDA_DRIFT_SHADOW") == "1":
    from drift_shadow import install

    install()
