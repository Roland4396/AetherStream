import unittest
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from aetherstream.features.stage_warning import STAGE_INTRO, WARNING_OPEN, WARNING_CLOSE
from aetherstream.features.scene_filter import SCENE_INTRO

from aetherstream.features.opus_notes import (
    PRO_OPUS_LAST_USER_APPEND_MARKER,
    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
    PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
)
from aetherstream.features.request_injections import (
    ASSISTANT_PREFILL_CONTINUATION_TEXT,
    append_assistant_prefill_continuation,
    apply_direct_opus_note,
    apply_forced_opus_note,
    apply_pioneer_opus_note,
    is_kimi_model,
)


class RequestInjectionTests(unittest.TestCase):
    def test_illustration_only_reminder_exact_wording(self):
        reminder = '（记得插图。）'
        self.assertEqual(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER, reminder)
        self.assertEqual(PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT, '!!important!!\n' + reminder * 3 + '\n!!important!!')

    def test_kimi_model_matches_provider_prefixes_only(self):
        for model in ('kimi-k3', 'free/kimi-k3', '[NV]kimi-k3', '[G]Kimi-2.7-code',
                      'moonshotai/Kimi-K3', 'kimi-k3-fireworks'):
            with self.subTest(model=model):
                self.assertTrue(is_kimi_model(model))
        for model in (None, '', 'glm-5.3', 'claude-opus-4-6', 'not-kimi-k3'):
            with self.subTest(model=model):
                self.assertFalse(is_kimi_model(model))

    def test_kimi_reuses_shared_note_without_duplicates(self):
        for stream in (True, False):
            request = {'stream': stream, 'messages': [
                {'role': 'user', 'content': '<latest_human_message>hello</latest_human_message>'},
                {'role': 'assistant', 'content': 'prefill'},
            ]}
            kwargs = dict(selected_model='free/kimi-k3', trace_prefix='[TRACE test]',
                          route_label='kimi', log=lambda _message: None)
            apply_forced_opus_note(request, **kwargs)
            apply_forced_opus_note(request, **kwargs)
            content = request['messages'][0]['content']
            self.assertEqual(content.count(PRO_OPUS_LAST_USER_APPEND_MARKER), 1)
            self.assertEqual(content.count(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER), 3)
            self.assertEqual(request['messages'][-1], {'role': 'assistant', 'content': 'prefill'})

    def test_assistant_prefill_continuation_is_shared_and_conditional(self):
        logs = []
        request = {"messages": [{"role": "assistant", "content": "prefill"}]}

        changed = append_assistant_prefill_continuation(
            request,
            selected_model="claude-test",
            trace_prefix="[TRACE test]",
            route_label="route",
            log=logs.append,
        )

        self.assertTrue(changed)
        self.assertEqual(request["messages"][-1]["role"], "user")
        self.assertEqual(request["messages"][-1]["content"], ASSISTANT_PREFILL_CONTINUATION_TEXT)
        self.assertEqual(len(logs), 1)

    def test_pioneer_runtime_switch_prevents_all_opus_injection(self):
        request = {"messages": [{"role": "user", "content": "text"}]}
        apply_pioneer_opus_note(
            request,
            selected_model="claude-opus-4-6",
            trace_prefix="[TRACE test]",
            route_label="free_openai",
            runtime_lookup=lambda *keys: False,
            is_opus_model=lambda _model: True,
            log=lambda _message: None,
        )

        self.assertEqual(request["messages"][0]["content"], "text")

    def test_direct_opus_injection_is_idempotent(self):
        request = {
            "messages": [{
                "role": "user",
                "content": "<latest_human_message>hello</latest_human_message>",
            }]
        }
        kwargs = {
            "selected_model": "claude-opus-4-6",
            "trace_prefix": "[TRACE test]",
            "route_label": "claude_code",
            "is_opus_model": lambda _model: True,
            "log": lambda _message: None,
        }

        apply_direct_opus_note(request, **kwargs)
        apply_direct_opus_note(request, **kwargs)

        content = request["messages"][0]["content"]
        self.assertEqual(content.count(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER), 3)
        self.assertEqual(content.count(PRO_OPUS_LAST_USER_APPEND_MARKER), 1)
        self.assertNotIn('<disclaimer>', content)
        self.assertIn(PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT, content)
        self.assertNotIn("（记得最后的收尾输出）", content)

    def test_pioneer_opus_illustration_falls_back_without_latest_human_tag(self):
        request = {
            "messages": [
                {"role": "user", "content": "current SillyTavern prompt"},
                {"role": "assistant", "content": "prefill"},
            ]
        }

        apply_pioneer_opus_note(
            request,
            selected_model="claude-opus-4-6",
            trace_prefix="[TRACE test]",
            route_label="free_openai",
            runtime_lookup=lambda *keys: True,
            is_opus_model=lambda _model: True,
            log=lambda _message: None,
        )

        content = request["messages"][0]["content"]
        self.assertIn(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER, content)
        self.assertIn(PRO_OPUS_LAST_USER_APPEND_MARKER, content)

    def test_forced_opus_injection_applies_to_non_opus_channel(self):
        request = {
            "messages": [{
                "role": "user",
                "content": "<latest_human_message>hello</latest_human_message>",
            }]
        }

        apply_forced_opus_note(
            request,
            selected_model="grok-4.5",
            trace_prefix="[TRACE test]",
            route_label="configured",
            log=lambda _message: None,
        )

        content = request["messages"][0]["content"]
        self.assertIn(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER, content)
        self.assertIn(PRO_OPUS_LAST_USER_APPEND_MARKER, content)


class KimiRouteInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_chat_routes_inject_before_forwarding(self):
        from starlette.requests import Request
        from aetherstream.api import app
        from aetherstream.api.chat_routes import chat_completions

        for model, stream in (('free/kimi-k3', True), ('free/kimi-k3', False),
                              ('[NV]kimi-k3', True), ('[G]Kimi-2.7-code', False),
                              ('transsion/kimi-k3', True), ('transsion/kimi-k3', False),
                              ('free/glm-5.2', True), ('free/glm-5.2', False),
                              ('free/claude-opus-4-6', True), ('free/claude-opus-4-6', False)):
            with self.subTest(model=model, stream=stream):
                deps = copy.copy(app.chat_route_dependencies)
                # Direct handler tests bypass ASGI cleanup; do not mutate the global registry.
                deps.active_stream_registry = type(app.active_stream_registry)(log=lambda _: None)
                deps.release_active_stream_caller = deps.active_stream_registry.release
                captured = []
                logs = []
                deps.log = logs.append
                deps.save_request_log = lambda *args, **kwargs: None
                deps.replay_service = SimpleNamespace(prepare=lambda **kwargs: None)
                deps.model_policy = SimpleNamespace(
                    apply_claude_sampling_compat=lambda data: [],
                    is_model_allowed=lambda model: True,
                    is_claude_model=lambda model: 'claude' in model,
                    is_gemini_model=lambda model: False,
                )
                deps.get_claude_upstream_for_provider = lambda provider: ('test-key', 'http://test.invalid/v1')
                deps.resolve_openai_compatible_route = AsyncMock(return_value={
                    'name': 'free-antigravity' if 'claude' in model else 'test',
                    'base_url': 'http://test.invalid/v1', 'api_key': 'test-key',
                    'inject_opus_note': model == 'free/claude-opus-4-6',
                })
                deps.apply_openai_template_thinking_disabled = lambda *args: False
                deps.should_append_pro_opus46_last_user_note = lambda *args: False
                deps.should_apply_deepseek_drawing_context_filter = lambda *args: False
                deps.apply_pro_no_reasoning_payload = lambda *args: None

                async def fake_stream(**kwargs):
                    self.assertEqual(kwargs['enable_early_stop'], is_kimi_model(model))
                    captured.append(kwargs['request_data'])
                    yield 'data: [DONE]\n\n'

                async def fake_collect(**kwargs):
                    self.assertEqual(kwargs['enable_early_stop'], is_kimi_model(model))
                    captured.append(kwargs['request_data'])
                    return ('ok', model, {}, 'stop', '')

                deps.forward_chat_completions_stream = fake_stream
                deps.collect_chat_completions_stream = fake_collect
                stage = '<stage><act name="测试角色">now: 示例</act></stage>'
                scene = SCENE_INTRO + '\n<scene>\nDay:1, 12:48 | library\n【场景】library\n【氛围】quiet\n【类型】daily\n期限=none | 跳跃=无\n</scene>'
                state = '系统:\n  当前年份: 2024\n  当前日期: 4月9日\n  当前时间: 12:47\n现在的时间是: 12:47'
                user_text = ('<latest_human_message>hello\n' + scene + '\n</latest_human_message>\n'
                             + STAGE_INTRO + '\n' + stage + '\n' + scene + '\n<recall>AM0001</recall>\n' + state)
                payload = {'model': model, 'stream': stream, 'temperature': 0.7, 'top_p': 0.98, 'messages': [
                    {'role': 'user', 'content': user_text},
                ]}
                request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions',
                                   'headers': [], 'client': ('127.0.0.1', 1234)})
                request._body = json.dumps(payload).encode()
                with patch('httpx.AsyncClient.send', side_effect=AssertionError('No external requests in test')):
                    response = await chat_completions(request, deps)
                    if hasattr(response, 'body_iterator'):
                        async for _ in response.body_iterator:
                            pass
                self.assertEqual(response.status_code, 200, logs)
                self.assertEqual(len(captured), 1, logs)
                if is_kimi_model(model):
                    self.assertNotIn('temperature', captured[0])
                    self.assertNotIn('top_p', captured[0])
                else:
                    self.assertEqual(captured[0]['temperature'], 0.7)
                    self.assertEqual(captured[0]['top_p'], 0.98)
                self.assertEqual(payload['top_p'], 0.98)
                content = captured[0]['messages'][0]['content']
                expected_note = is_kimi_model(model) or model == 'free/claude-opus-4-6'
                self.assertEqual(content.count(PRO_OPUS_LAST_USER_APPEND_MARKER), int(expected_note))
                if expected_note:
                    self.assertIn(PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT, content)
                    self.assertEqual(content.count(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER), 3)
                self.assertIn(WARNING_OPEN + stage + WARNING_CLOSE, content)
                self.assertNotIn('<disclaimer>', content)
                self.assertNotIn('<scene>', content)
                self.assertNotIn(SCENE_INTRO, content)
                self.assertIn('<recall>AM0001</recall>', content)
                self.assertIn(state, content)
                self.assertTrue(any('ai_scene_filter removed=2 invalid_skipped=0' in line for line in logs))
                self.assertEqual(payload['messages'][0]['content'], user_text)
                self.assertNotIn(PRO_OPUS_LAST_USER_APPEND_MARKER, payload['messages'][0]['content'])


if __name__ == "__main__":
    unittest.main()
