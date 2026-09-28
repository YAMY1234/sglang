"""Coordinate dynamic CUDA graph capture with background staging transfers."""

from threading import RLock

# PyTorch stream pools can alias a capture stream with a long-lived transfer stream.
graph_capture_lock = RLock()
