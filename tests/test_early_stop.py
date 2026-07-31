import unittest

from aetherstream.features.early_stop import DEFAULT_EARLY_STOP_TAGS, EarlyStopMatcher


class EarlyStopMatcherTests(unittest.TestCase):
    def test_closing_leaf_is_a_default_stop_tag(self):
        matcher = EarlyStopMatcher(
            lookup=lambda *_keys: None,
            env_enabled=True,
            env_tags=None,
            env_case_sensitive=True,
        )

        text = "main response\n<closing_leaf>detached tail"

        self.assertIn("<closing_leaf>", DEFAULT_EARLY_STOP_TAGS)
        self.assertEqual(matcher.find(text), len("main response\n"))


if __name__ == "__main__":
    unittest.main()
