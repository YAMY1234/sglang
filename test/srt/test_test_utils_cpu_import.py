"""Real helper imports must work without a visible CUDA device."""

import os
import subprocess
import sys
import unittest


class CPUHelperImportTest(unittest.TestCase):
    def test_empty_and_unset_cuda_visible_devices(self):
        for visible in ("", None):
            with self.subTest(visible=visible):
                env = dict(os.environ)
                if visible is None:
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = visible
                code = """
import os
before = os.environ.get('CUDA_VISIBLE_DEVICES')
from sglang.test import test_utils as t
assert os.environ.get('CUDA_VISIBLE_DEVICES') == before
assert t.DEFAULT_PORT_FOR_SRT_TEST_RUNNER in (10000, 20000)
assert t._test_port_device_index(None) == 0
assert t._test_port_device_index('') == 0
assert t._test_port_device_index('  ') == 0
assert t._test_port_device_index('-1') == 0
assert t._test_port_device_index('3') == 3
assert t._test_port_device_index('2,3') == 2
print('CPU_HELPER_IMPORT_PASS', repr(before))
"""
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CPU_HELPER_IMPORT_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
