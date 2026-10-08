import unittest

from aetherstream.streaming.state import ActiveStreamRegistry


class ActiveStreamRegistryTests(unittest.TestCase):
    def test_latest_local_stream_supersedes_previous_same_model(self):
        logs = []
        registry = ActiveStreamRegistry(log=logs.append)
        registry.register(
            "caller-a",
            trace_id="trace-old",
            model="glm-5.2-local",
            msg_count=9,
        )
        old_event = registry.cancellation_event("caller-a", "trace-old")

        registry.register(
            "caller-a",
            trace_id="trace-new",
            model="glm-5.2-local",
            msg_count=9,
            supersede_previous=True,
        )
        new_event = registry.cancellation_event("caller-a", "trace-new")

        self.assertIsNotNone(old_event)
        self.assertTrue(old_event.is_set())
        self.assertIsNotNone(new_event)
        self.assertFalse(new_event.is_set())
        self.assertTrue(any("caller_supersede" in line for line in logs))

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
