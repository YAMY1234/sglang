# Temporarily do this to avoid changing all imports in the repo
from sglang.srt.utils.common import *

# Backwards-compat re-exports for external consumers (e.g. dynamo 0.7.0)
# that still do `from sglang.srt.utils import get_local_ip_auto` etc.
# These were split into sglang.srt.utils.network when utils became a
# package. Re-exporting everything avoids per-symbol maintenance.
from sglang.srt.utils.network import *  # noqa: F401,F403
