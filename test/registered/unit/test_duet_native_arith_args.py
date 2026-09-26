"""DUET opt-in CLI/runtime contract; no model or CUDA execution.

Run normally in the serving CPU image to require real ServerArgs and publish().
For a workstation without SGLang dependencies, --stdlib-only runs just the CSV
parser checks and explicitly skips runtime integration (never silently skips it).
"""

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest


STDLIB_ONLY = "--stdlib-only" in sys.argv
if STDLIB_ONLY:
    sys.argv.remove("--stdlib-only")

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/arg_groups/duet_native_arith_args.py"
# test/registered/unit -> repository is parents[3]. Import the pure parser
# directly so the explicit stdlib-only mode does not initialize SGLang.
spec = importlib.util.spec_from_file_location("duet_native_arith_args_tested", SOURCE)
args_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(args_module)

if not STDLIB_ONLY:
    from sglang.srt import runtime_context as rc
    from sglang.srt.server_args import ServerArgs


class TestDuetComponentParser(unittest.TestCase):
    def test_all_components_are_canonical_immutable_tuple(self):
        self.assertEqual(args_module.parse_disabled_components("state,router,rmsnorm"),
                         ("router", "rmsnorm", "state"))
        self.assertIsInstance(args_module.parse_disabled_components("router"), tuple)

    def test_duplicates_and_whitespace(self):
        self.assertEqual(args_module.parse_disabled_components(" state, router ,state "), ("router", "state"))

    def test_empty_explicit_value_clears_overrides(self):
        self.assertEqual(args_module.parse_disabled_components(""), ())
        self.assertEqual(args_module.parse_disabled_components("  "), ())

    def test_unknown_misspelled_and_empty_entries_rejected(self):
        for value in ("routr", "Router", "all", "router,,state", ",state", "state,"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                args_module.parse_disabled_components(value)

    def test_nonstring_rejected(self):
        for value in (None, 1, ["router"], ("state",)):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                args_module.parse_disabled_components(value)


@unittest.skipIf(STDLIB_ONLY, "explicit --stdlib-only: serving image must run real CLI/runtime tests")
class TestDuetNativeArithmeticCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def publish(self, *args):
        parsed = self.parser.parse_args(["--model", "dummy", *args])
        server_args = ServerArgs.from_cli_args(parsed)
        rc.publish(server_args, role="test")
        return server_args, rc.get_exec().features

    def test_ordinary_fork_defaults_off(self):
        raw, features = self.publish()
        self.assertIs(raw.duet_native_arith, False)
        self.assertIs(features.duet_native_arith, False)
        self.assertEqual(features.duet_native_arith_disable_components, ())
        self.assertIsInstance(features.duet_native_arith_disable_components, tuple)

    def test_opt_in_and_component_ablation_publish_exact_runtime_paths(self):
        raw, features = self.publish("--duet-native-arith", "--duet-native-arith-disable-components", "state,router,router")
        self.assertIs(raw.duet_native_arith, True)
        self.assertIs(features.duet_native_arith, True)
        self.assertEqual(raw.duet_native_arith_disable_components, ("router", "state"))
        self.assertEqual(features.duet_native_arith_disable_components, ("router", "state"))
        self.assertIsInstance(features.duet_native_arith_disable_components, tuple)

    def test_disabled_component_option_does_not_enable_master(self):
        _, features = self.publish("--duet-native-arith-disable-components", "rmsnorm")
        self.assertIs(features.duet_native_arith, False)
        self.assertEqual(features.duet_native_arith_disable_components, ("rmsnorm",))

    def test_explicit_no_flag_overrides_prior_enable(self):
        _, features = self.publish("--duet-native-arith", "--no-duet-native-arith")
        self.assertIs(features.duet_native_arith, False)

    def test_invalid_cli_component_is_rejected_before_publish(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.parser.parse_args(["--model", "dummy", "--duet-native-arith-disable-components", "router,norm"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
