"""Shared policy for NVIDIA megakernel runtime slots."""

from sglang.srt.environ import envs


def megakernel_slot_requested(name: str) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled

    return megakernel_enabled() and name in envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.get()
