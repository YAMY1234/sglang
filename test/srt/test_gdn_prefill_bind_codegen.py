"""Compile the actual packing kernel for serving GPUs without a GPU allocation."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class PackedBindCodegen(unittest.TestCase):
    def test_sm100_sm103_kernel_compiles_without_device_execution(self):
        source = Path(__file__).resolve().parents[2] / "python/sglang/srt/mem_cache/gdn_prefill_bind.py"
        script = r'''
import importlib.util, json, sys
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
spec = importlib.util.spec_from_file_location("packed_codegen", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
rows = []
for arch in (100, 103):
    for start in (0, 72):
        source = ASTSource(module._pack, {"DESC": "*i64"},
                           constexprs={"START": start, "BLOCK": 512})
        compiled = triton.compile(source, target=GPUTarget("cuda", arch, 32),
                                  options={"num_warps": 4})
        assert len(compiled.asm["cubin"]) > 0
        rows.append(dict(arch=arch, start=start, cubin_bytes=len(compiled.asm["cubin"])))
print(json.dumps(dict(passed=True, cases=rows, gpu_executions=0)))
'''
        env = {**os.environ, "TRITON_INTERPRET": "0"}
        result = subprocess.run([sys.executable, "-c", script, str(source)],
                                env=env, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout)


if __name__ == "__main__":
    unittest.main()
