"""Optional TwinStar instrumentation, kept outside the serving distribution."""

import importlib
import importlib.util
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(None)
def optional(name):
    qualified = name if name.startswith("twinstar.") else "twinstar_sgl." + name
    try:
        spec = importlib.util.find_spec(qualified)
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec is None:
        logger.info("Flash-Next diagnostic %s is unavailable; disabled", qualified)
        return None
    # An installed diagnostic's failures must remain visible.
    return importlib.import_module(qualified)
