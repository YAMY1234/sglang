"""Run-time overrides of selected switches, for A/B variants inside one loaded server.

Off unless SGLANG_Q35MF_RT_TOGGLE_FILE names a JSON object file; a caller passes its env value as the
default, so a missing file, a missing key or an unreadable file keeps the env behaviour. The file is
stat'ed once every 16 lookups and re-read when its mtime changes.
"""

import json
import os

from sglang.srt.environ import envs

_PATH = envs.SGLANG_Q35MF_RT_TOGGLE_FILE.get() or None
_state = {"mtime": None, "flags": {}, "n": 0}


def rt_flag(name: str, default: bool) -> bool:
    if _PATH is None:
        return default
    _state["n"] += 1
    if _state["n"] & 15 == 1:
        try:
            mtime = os.stat(_PATH).st_mtime_ns
            if mtime != _state["mtime"]:
                with open(_PATH) as f:
                    _state["flags"] = json.load(f)
                _state["mtime"] = mtime
        except (OSError, ValueError):
            _state["flags"], _state["mtime"] = {}, None
    value = _state["flags"].get(name)
    return default if value is None else bool(value)
