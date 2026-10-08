import copy
import unittest
from unittest.mock import AsyncMock, patch

from aetherstream.api import app
from aetherstream.api.app import (
    apply_claude_client_compat_request,
    apply_claude_model_compat_request,
    is_claude_opus55_model,
)
from aetherstream.api.chat_routes import (
    _apply_configured_reasoning,
    _apply_configured_template_thinking,
    _drop_configured_request_fields,
    _upstream_http_status,
)
from aetherstream.routing.model_policy import ModelPolicy


class UpstreamRequestCompatTests(unittest.TestCase):
    def test_recovers_upstream_http_status_from_collector_error(self):
        self.assertEqual(
            _upstream_http_status(RuntimeError('Upstream error: 400 - invalid request')),
            400,
        )
        self.assertEqual(_upstream_http_status(RuntimeError('transport failed')), 502)

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

    def test_route_can_match_local_glm_max_reasoning_parameters(self):
        payload = {
            "model": "[OR]GLM-5.2",
            "reasoning_effort": "low",
            "effort": "low",
            "chat_template_kwargs": {
                "enable_thinking": False,
                "thinking": False,
                "other": "kept",
            },
        }

        changed = _apply_configured_reasoning(payload, {
            "effort": "max",
            "reasoning_effort": "max",
            "enable_thinking": True,
            "thinking": True,
        })

        self.assertEqual(changed, [
            "effort",
            "reasoning_effort",
            "chat_template_kwargs.enable_thinking",
            "chat_template_kwargs.thinking",
        ])
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["effort"], "max")
        self.assertEqual(payload["chat_template_kwargs"], {
            "enable_thinking": True,
            "thinking": True,
            "other": "kept",
        })

    def test_model_directory_preserves_configured_reasoning_overrides(self):
        configured = [{
            "name": "youzi-glm52",
            "base_url": "http://account-pool-proxy:3200/v1",
            "include_models": ["[OR]GLM-5.2"],
            "enable_thinking": True,
            "thinking": True,
            "effort": "max",
            "reasoning_effort": "max",
        }]

        with patch.object(app, "_runtime_lookup", return_value=configured):
            route = app.get_openai_compatible_upstreams()[0]

        self.assertTrue(route["enable_thinking"])
        self.assertTrue(route["thinking"])
        self.assertEqual(route["effort"], "max")
        self.assertEqual(route["reasoning_effort"], "max")

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

    def test_claude_client_compat_preserves_nonempty_tools(self):
        tools = [
            {"name": "Read", "input_schema": {"type": "object"}},
            {"name": "submit_response_self_evaluation", "input_schema": {"type": "object"}},
        ]

        sanitized, meta = apply_claude_client_compat_request({"tools": tools})

        self.assertEqual(sanitized["tools"], tools)
        self.assertEqual(meta["tools"], "preserved:2")

    def test_fable_removes_unsupported_thinking_disabled(self):
        payload = {
            "model": "claude-fable-5",
            "thinking": {"type": "disabled"},
        }

        sanitized, meta = apply_claude_model_compat_request(payload)

        self.assertNotIn("thinking", sanitized)
        self.assertEqual(meta["thinking"], "removed_disabled_for_fable")

    def test_opus55_requires_adaptive_without_changing_effort_or_content(self):
        for model in (
            "claude-opus-5-5", "claude-opus-5-5-20260820",
            "claude-opus-5.5", "free/claude-opus-5-5",
            "anthropic/claude-opus-5.5",
            "[m1]claude-opus-5-5", "[m2]claude-opus-5-5-thinking",
            "route/vendor-claude-opus-5-5:latest", "CLAUDE-OPUS-5-5-HIGH",
            "prefix-claude-opus-5-5/suffix",
        ):
            with self.subTest(model=model):
                payload = {
                    "model": model, "stream": True, "max_tokens": 60000,
                    "temperature": 1, "thinking": {"type": "disabled"},
                    "output_config": {"effort": "max"},
                    "messages": [{"role": "user", "content": "synthetic test"}],
                }
                sanitized, meta = apply_claude_model_compat_request(payload)
                self.assertEqual(sanitized, dict(payload, thinking={"type": "adaptive"}))
                self.assertEqual(payload["thinking"], {"type": "disabled"})
                self.assertEqual(meta["thinking"], "opus55_forced_adaptive")

    def test_opus55_removes_manual_budget_and_preserves_display(self):
        payload = {
            "model": "claude-opus-5-5",
            "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "omitted"},
        }
        sanitized, _ = apply_claude_model_compat_request(payload)
        self.assertEqual(sanitized["thinking"], {"type": "adaptive", "display": "omitted"})

    def test_opus55_compat_runs_after_global_disabled_preference(self):
        payload = {"model": "claude-opus-5-5-20260820"}
        cfg = {"tools_enabled": True, "thinking_type": "disabled",
               "context_management_enabled": False, "context_management_edits": []}
        with patch.object(app, "get_claude_client_compat_settings", return_value=cfg):
            sanitized, _ = apply_claude_client_compat_request(payload)
        self.assertEqual(sanitized["thinking"], {"type": "disabled"})
        sanitized, _ = apply_claude_model_compat_request(sanitized)
        self.assertEqual(sanitized["thinking"], {"type": "adaptive"})

    def test_opus55_compat_does_not_change_other_models(self):
        for model in ("claude-opus-4-6", "claude-opus-5", "claude-sonnet-5-5", "claude-opus-5-50",
                      "[m1]claude-opus-5-51", "[m1]claude-opus-5.50"):
            with self.subTest(model=model):
                payload = {"model": model, "thinking": {"type": "disabled"}}
                sanitized, meta = apply_claude_model_compat_request(payload)
                self.assertEqual(sanitized, payload)
                self.assertEqual(meta, {})

    def test_opus55_removes_only_disabling_aliases_without_mutating_input(self):
        payload = {
            "model": "[m1]claude-opus-5-5", "reasoning_effort": "none",
            "effort": "off", "thinking": {"type": "disabled"},
            "reasoning": {"effort": "none", "summary": "auto"},
            "output_config": {"effort": "max", "other": "kept"},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False, "other": 1},
            "messages": [{"role": "user", "content": "unchanged"}],
            "max_tokens": 30000, "tools": [{"type": "function", "function": {"name": "test"}}],
        }
        original = copy.deepcopy(payload)
        sanitized, meta = apply_claude_model_compat_request(payload)
        expected = copy.deepcopy(original)
        expected.pop('reasoning_effort')
        expected.pop('effort')
        expected['thinking'] = {'type': 'adaptive'}
        expected['reasoning'] = {'summary': 'auto'}
        expected['chat_template_kwargs'] = {'other': 1}
        self.assertEqual(sanitized, expected)
        self.assertEqual(payload, original)
        self.assertIn('reasoning_effort', meta['removed'])
        self.assertEqual(apply_claude_model_compat_request(sanitized)[0], sanitized)

    def test_opus55_preserves_positive_efforts_and_drops_empty_disable_containers(self):
        for effort in ('low', 'medium', 'high', 'max'):
            with self.subTest(effort=effort):
                payload = {'model': '[m1]claude-opus-5-5', 'reasoning_effort': effort,
                           'reasoning': {'effort': effort}, 'output_config': {'effort': effort}}
                self.assertEqual(apply_claude_model_compat_request(payload)[0],
                                 dict(payload, thinking={'type': 'adaptive'}))
        payload = {'model': '[m1]claude-opus-5-5', 'reasoning': {'effort': 'none'},
                   'output_config': {'effort': 'none'}, 'chat_template_kwargs': {'thinking': False}}
        self.assertEqual(apply_claude_model_compat_request(payload)[0],
                         {'model': '[m1]claude-opus-5-5', 'thinking': {'type': 'adaptive'}})

    def test_model_match_is_scoped_to_opus55_version_with_aliases(self):
        for model in ('[m1]claude-opus-5-5', '[m2]CLAUDE-OPUS-5-5-20260820',
                      'provider/claude-opus-5-5/fast', 'claude-opus-5.5'):
            self.assertTrue(is_claude_opus55_model(model), model)
        for model in (None, '', 'claude-5-5', 'claude-sonnet-5-5', 'claude-opus-5-50',
                      'claude-opus-4-5', '[m1]claude-opus-5-51'):
            self.assertFalse(is_claude_opus55_model(model), model)


class Opus55RouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_himodels_opus55_uses_native_messages_with_cached_or_new_directory(self):
        model = "claude-opus-5-5-20260820"
        route = {"name": "account-pool-claude", "base_url": "http://test.invalid/v1"}
        directory = {"routes": {model: route}}
        for fresh in (False, True):
            with self.subTest(fresh=fresh):
                snapshots = [{"routes": {}}, directory] if fresh else [directory]
                with patch.object(app, "refresh_model_directory", AsyncMock(side_effect=snapshots)):
                    self.assertIsNone(await app.resolve_openai_compatible_route(model))

    async def test_other_models_and_explicit_temporary_channels_keep_their_route(self):
        for model, name in (("claude-opus-4-6", "account-pool-claude"),
                            ("claude-opus-5-5", "free-temporary-opus")):
            with self.subTest(model=model, name=name):
                route = {"name": name, "base_url": "http://test.invalid/v1"}
                with patch.object(app, "refresh_model_directory", AsyncMock(return_value={"routes": {model: route}})):
                    self.assertEqual(await app.resolve_openai_compatible_route(model), route)


if __name__ == "__main__":
    unittest.main()
