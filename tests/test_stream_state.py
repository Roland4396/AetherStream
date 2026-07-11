import unittest

from aetherstream.streaming.state import ActiveStreamRegistry


class ActiveStreamRegistryTests(unittest.TestCase):
    def test_release_trace_is_idempotent_and_provider_independent(self):
        logs = []
        registry = ActiveStreamRegistry(log=logs.append)
        registry.register("caller-a", trace_id="trace-a", model="a", msg_count=1)
        registry.register("caller-b", trace_id="trace-b", model="b", msg_count=2)

        registry.release_trace("trace-a")
        registry.release_trace("trace-a")

        self.assertEqual(len(registry), 1)
        self.assertIsNone(registry.get("caller-a"))
        self.assertIsNotNone(registry.get("caller-b"))
        self.assertEqual(sum("caller_release_trace" in line for line in logs), 1)


if __name__ == "__main__":
    unittest.main()
