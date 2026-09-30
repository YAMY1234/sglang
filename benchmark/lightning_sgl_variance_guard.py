"""Compatibility entry for the complete eight-group variance guard."""

from lightning_sgl_decision import assess as assess_variance
from lightning_sgl_decision import noise_multiple
from lightning_sgl_guard import main

__all__ = ["assess_variance", "main", "noise_multiple"]

if __name__ == "__main__":
    raise SystemExit(main())
