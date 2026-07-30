import unittest

from aetherstream.api.chat_routes import (
    _apply_configured_template_thinking,
    _drop_configured_request_fields,
)
from aetherstream.routing.model_policy import ModelPolicy


class UpstreamRequestCompatTests(unittest.TestCase):
    def test_drops_only_fields_configured_for_route(self):
        payload = {
            "model": "grok-4.5",
            "messages": [],
            "max_tokens": 60000,
            "temperature": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "stop": ["END"],
        }
        route = {
            "drop_request_fields": [
                "presence_penalty",
                "frequency_penalty",
                "stop",
            ]
        }

        removed = _drop_configured_request_fields(payload, route)

        self.assertEqual(removed, ["presence_penalty", "frequency_penalty", "stop"])
        self.assertEqual(payload["max_tokens"], 60000)
        self.assertEqual(payload["temperature"], 1)
        self.assertNotIn("stop", payload)

    def test_ignores_invalid_or_absent_field_configuration(self):
        payload = {"model": "grok-4.5", "max_tokens": 60000}

        self.assertEqual(_drop_configured_request_fields(payload, {}), [])
        self.assertEqual(_drop_configured_request_fields(payload, {"drop_request_fields": "stop"}), [])
        self.assertEqual(payload["max_tokens"], 60000)

    def test_route_can_force_template_thinking_on(self):
        payload = {
            "model": "glm-5.2-local",
            "chat_template_kwargs": {"enable_thinking": False, "other": "kept"},
        }

        changed = _apply_configured_template_thinking(payload, {"enable_thinking": True})

        self.assertTrue(changed)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True, "other": "kept"},
        )

    def test_free_claude_is_classified_for_anthropic_route(self):
        policy = ModelPolicy(
            responses_models=frozenset(),
            allowed_gpt_models=frozenset(),
            allowed_gemini_models=frozenset(),
            allowed_claude_models=frozenset(),
            responses_unsupported_fields=frozenset(),
        )

        self.assertTrue(policy.is_claude_model("free/claude-fable-5"))
        self.assertTrue(policy.is_claude_model("free/claude-opus-4-8"))
        self.assertFalse(policy.is_claude_model("free/gpt-5.5"))


if __name__ == "__main__":
    unittest.main()
