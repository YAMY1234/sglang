"""Temporary Triton codegen compatibility for CUDA capability 10.7."""


def apply_triton_sm107_family_target_patch() -> None:
    """Compile SM107 kernels for the forward-compatible SM100 family target.

    The Triton bundled in the current CUDA 13 public image maps capability 10.7
    to ``sm_107a``. Its LLVM backend does not recognize that processor yet, but
    CUDA 13.4 accepts ``sm_100f`` and the resulting cubin runs on SM107.
    """
    from triton.backends.nvidia import compiler

    if getattr(compiler, "_sglang_sm107_family_target_patched", False):
        return

    original_sm_arch_from_capability = compiler.sm_arch_from_capability
    original_make_ptx = compiler.CUDABackend.make_ptx

    def sm_arch_from_capability(capability: int) -> str:
        if capability == 107:
            return "sm_100f"
        return original_sm_arch_from_capability(capability)

    def make_ptx(self, src, metadata, opt, capability):
        ptx = original_make_ptx(self, src, metadata, opt, capability)
        if capability == 107:
            ptx = ptx.replace(".target sm_107a", ".target sm_100f")
            ptx = ptx.replace(".target sm_107f", ".target sm_100f")
        return ptx

    compiler.sm_arch_from_capability = sm_arch_from_capability
    compiler.CUDABackend.make_ptx = make_ptx
    compiler._sglang_sm107_family_target_patched = True
