"""CPU namespace tests without importing GPU-dependent SGLang package init."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[2] / "python/sglang/srt/model_executor/duet_policy.py"
spec = importlib.util.spec_from_file_location("duet_cache_policy", path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class DuetPolicyTests(unittest.TestCase):
    def model(self, **overrides):
        config = {"duet_release": "/release/fn", "duet_sha256": "abc"}
        config.update(overrides)
        return SimpleNamespace(hf_config=SimpleNamespace(twinstar=config))

    def req(self, session=None, extra=None):
        return SimpleNamespace(session_id=session, extra_key=extra, rid="same-user-rid")

    def test_cold_requests_with_identical_tokens_and_rid_are_isolated(self):
        a, b = self.req(), self.req()
        policy.namespace_request(self.model(), a)
        policy.namespace_request(self.model(), b)
        self.assertNotEqual(a.extra_key, b.extra_key)

    def test_only_same_session_release_and_extra_key_share(self):
        a, b = self.req("chain1", "x"), self.req("chain1", "x")
        policy.namespace_request(self.model(), a)
        policy.namespace_request(self.model(), b)
        self.assertEqual(a.extra_key, b.extra_key)
        for model, req in ((self.model(), self.req("chain2", "x")),
                           (self.model(duet_sha256="changed"), self.req("chain1", "x")),
                           (self.model(), self.req("chain1", "y"))):
            policy.namespace_request(model, req)
            self.assertNotEqual(a.extra_key, req.extra_key)

    def test_stock_namespace_is_unchanged(self):
        req = self.req(extra="stock")
        policy.namespace_request(SimpleNamespace(hf_config=SimpleNamespace()), req)
        self.assertEqual(req.extra_key, "stock")

    def test_empty_session_is_rejected(self):
        with self.assertRaises(ValueError):
            policy.namespace_request(self.model(), self.req(""))


if __name__ == "__main__":
    unittest.main()
