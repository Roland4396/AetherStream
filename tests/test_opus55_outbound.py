"""Opus 5.5 regression checks at the outbound boundary; no real inference."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from starlette.requests import Request

from aetherstream.api import app
from aetherstream.api.chat_routes import chat_completions
from aetherstream.api.protocol_gateway import route_protocol_request
from aetherstream.features.opus_notes import (
    PRO_OPUS_LAST_USER_APPEND_TEXT,
    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
)
from aetherstream.features.request_injections import apply_forced_opus_note
from aetherstream.streaming.dedupe import ExactRequestCoalescer


class Opus55OutboundTests(unittest.IsolatedAsyncioTestCase):
    def inject_expected(self, payload):
        expected = copy.deepcopy(payload)
        apply_forced_opus_note(expected, selected_model=payload['model'], trace_prefix='test',
                               route_label='test', log=lambda _: None)
        return expected

    async def capture(self, payload, *, protocol='chat', route_kind='directory', overrides=None):
        deps = copy.copy(app.chat_route_dependencies)
        deps.active_stream_registry = type(app.active_stream_registry)(log=lambda _: None)
        deps.release_active_stream_caller = deps.active_stream_registry.release
        captured, logs = [], []
        deps.log = logs.append
        # Each outbound capture is independent; do not reuse another capture's
        # global cache when several protocol aliases produce an identical body.
        coalescer = ExactRequestCoalescer(ttl=180, log=logs.append)
        deps.run_exact_nonstream_once = coalescer.run
        deps.save_request_log = lambda *a, **k: None
        deps.replay_service = SimpleNamespace(prepare=lambda **k: None)
        route = {'name': 'pro-regression', 'base_url': 'http://offline.invalid/v1',
                 'api_key': 'offline-test-only', **(overrides or {})}
        deps.resolve_openai_compatible_route = AsyncMock(
            return_value=route if route_kind == 'directory' else None)
        deps.resolve_model_name_passthrough_upstream = lambda _: route if route_kind == 'passthrough' else None
        deps.get_claude_upstream_for_provider = lambda _: ('offline-test-only', 'http://offline.invalid/v1')
        deps.build_timed_claude_user_id = lambda **k: ('offline-user', 'offline-session', 'test', 60, 'test')
        deps.get_claude_prompt_caching_settings = lambda: {'enabled': False}
        deps.get_claude_cache_keepalive_settings = lambda: {'enabled': False}
        # Exercise the final model guard after the native global disabled policy.
        deps.apply_claude_client_compat_request = lambda p: (dict(p, thinking={'type': 'disabled'}), {})
        deps.apply_claude_output_settings = lambda p: p

        async def fake_stream(**kwargs):
            captured.append(copy.deepcopy(kwargs['request_data']))
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            yield 'data: [DONE]\n\n'

        async def fake_collect(**kwargs):
            captured.append(copy.deepcopy(kwargs['request_data']))
            return ('ok', payload['model'], {}, 'stop', '')

        async def fake_native_collect(**kwargs):
            return (*await fake_collect(**kwargs), [])

        deps.forward_chat_completions_stream = fake_stream
        deps.collect_chat_completions_stream = fake_collect
        deps.forward_anthropic_messages_as_chat_stream = fake_stream
        deps.collect_anthropic_messages_as_chat_completion = fake_native_collect
        path = {'chat': '/v1/chat/completions', 'responses': '/v1/responses', 'anthropic': '/v1/messages'}[protocol]
        request = Request({'type': 'http', 'method': 'POST', 'path': path,
                           'headers': [], 'client': ('127.0.0.1', 1234)})
        original = copy.deepcopy(payload)
        request._body = json.dumps(payload).encode()
        with patch('httpx.AsyncClient.send', side_effect=AssertionError('External network forbidden')):
            response = (await chat_completions(request, deps) if protocol == 'chat'
                        else await route_protocol_request(request, deps, protocol=protocol))
            if hasattr(response, 'body_iterator'):
                async for _ in response.body_iterator:
                    pass
        self.assertEqual(response.status_code, 200, logs)
        self.assertEqual(len(captured), 1, logs)
        self.assertEqual(payload, original)
        return captured[0], logs

    async def test_log08_parameters_no_longer_send_disabling_reasoning(self):
        # The failing log's options, with entirely synthetic prompt content.
        for streaming in (False, True):
            payload = {
                'model': '[m1]claude-opus-5-5', 'stream': streaming,
                'messages': [{'role': 'user', 'content': 'Synthetic regression request.'}],
                'temperature': 1, 'top_p': 0, 'max_tokens': 30000,
                'presence_penalty': 0, 'frequency_penalty': 0,
                'stop': [], 'reasoning_effort': 'none',
            }
            outgoing, logs = await self.capture(payload)
            expected = self.inject_expected(payload)
            expected.pop('reasoning_effort')
            expected['thinking'] = {'type': 'adaptive'}
            # Conversion defaults on here; assert the actual upstream mode,
            # not the pre-collector client mode captured by the old stub.
            expected['stream'] = True
            self.assertEqual(outgoing, expected)
            self.assertTrue(any('opus55_adaptive_compat' in line for line in logs))
            self.assertFalse(any('pro_no_reasoning_payload' in line for line in logs))

    async def test_aliases_all_protocols_stream_modes_and_routes(self):
        scenarios = (
            ('[m1]claude-opus-5-5', 'directory'),
            ('claude-opus-5-5-20260820', 'directory'),
            ('vendor/CLAUDE-OPUS-5-5-high', 'directory'),
            ('[m2]claude-opus-5-5-fast', 'passthrough'),
            ('free/[m1]claude-opus-5-5', 'free'),
            ('claude-opus-5-5', 'native'),
            ('free/claude-opus-5-5', 'native'),
        )
        for model, route_kind in scenarios:
            for streaming in (False, True):
                for protocol in ('chat', 'responses', 'anthropic'):
                    with self.subTest(model=model, route=route_kind, stream=streaming, protocol=protocol):
                        payload = {'model': model, 'stream': streaming, 'max_tokens': 30000,
                                   'messages': [{'role': 'user', 'content': 'Synthetic regression request.'}],
                                   'thinking': {'type': 'disabled'}, 'reasoning_effort': 'none'}
                        if protocol == 'responses':
                            payload['input'] = payload.pop('messages')
                            payload['max_output_tokens'] = payload.pop('max_tokens')
                            payload['reasoning'] = {'effort': 'none'}
                        outgoing, _ = await self.capture(payload, protocol=protocol, route_kind=route_kind,
                                                        overrides={'reasoning_effort': 'none', 'enable_thinking': False,
                                                                   'thinking': False})
                        self.assertEqual(outgoing['thinking'], {'type': 'adaptive'})
                        self.assertNotIn('reasoning_effort', outgoing)
                        self.assertNotEqual(outgoing.get('reasoning', {}).get('effort'), 'none')
                        self.assertNotIn('chat_template_kwargs', outgoing)
                        self.assertEqual(outgoing['max_tokens'], 30000)
                        self.assertEqual(outgoing['model'], model.removeprefix('free/'))
                        self.assertIn('Synthetic regression request.', json.dumps(outgoing['messages']))
                        self.assert_note_once(outgoing)

    async def test_pro_route_preserves_explicit_adaptive_effort_and_display(self):
        for streaming in (False, True):
            payload = {'model': '[m1]claude-opus-5-5', 'stream': streaming,
                       'messages': [{'role': 'user', 'content': 'Synthetic regression request.'}],
                       'thinking': {'type': 'adaptive', 'display': 'omitted'},
                       'reasoning_effort': 'high', 'output_config': {'effort': 'max'}}
            outgoing, _ = await self.capture(payload)
            self.assertEqual(outgoing, dict(self.inject_expected(payload), stream=True))

    async def test_non_pro_directory_also_gets_model_guard(self):
        payload = {'model': '[m1]claude-opus-5-5', 'stream': True,
                   'messages': [{'role': 'user', 'content': 'Synthetic regression request.'}],
                   'reasoning_effort': 'none'}
        outgoing, _ = await self.capture(payload, overrides={'name': 'temporary-channel'})
        self.assertEqual(outgoing['thinking'], {'type': 'adaptive'})
        self.assertNotIn('reasoning_effort', outgoing)
        self.assert_note_once(outgoing)

    def assert_note_once(self, outgoing):
        body = '\n'.join(
            message['content'] if isinstance(message.get('content'), str)
            else '\n'.join(block.get('text', '') for block in message.get('content', []) if isinstance(block, dict))
            for message in outgoing['messages'] if message.get('role') == 'user'
        )
        self.assertEqual(body.count(PRO_OPUS_LAST_USER_APPEND_TEXT), 1)
        self.assertNotIn('<disclaimer>', body)
        self.assertEqual(body.count(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER), 3)

    async def test_already_injected_with_prefill_is_not_duplicated_by_any_route(self):
        for model, route_kind in (('[m1]claude-opus-5-5', 'directory'),
                                   ('free/[m1]claude-opus-5-5', 'free'),
                                   ('claude-opus-5-5', 'native'),
                                   ('vendor/claude-opus-5-5', 'passthrough')):
            for streaming in (False, True):
                for already_injected in (False, True):
                    with self.subTest(model=model, stream=streaming, already_injected=already_injected):
                        payload = {'model': model, 'stream': streaming, 'messages': [
                            {'role': 'user', 'content': '<latest_human_message>synthetic</latest_human_message>'},
                            {'role': 'assistant', 'content': 'Synthetic prefill.'},
                        ]}
                        if already_injected:
                            payload = self.inject_expected(payload)
                        outgoing, logs = await self.capture(payload, route_kind=route_kind,
                                                            overrides={'inject_opus_note': True})
                        self.assert_note_once(outgoing)
                        self.assertTrue(any('opus55_last_user_note' in line for line in logs))

    async def test_other_models_keep_pro_disable_policy(self):
        for model in ('[m1]claude-opus-4-6', '[m1]claude-sonnet-5-5', '[m1]claude-opus-5-50'):
            for streaming in (False, True):
                with self.subTest(model=model, stream=streaming):
                    payload = {'model': model, 'stream': streaming,
                               'messages': [{'role': 'user', 'content': 'Synthetic regression request.'}],
                               'thinking': {'type': 'adaptive'}, 'reasoning_effort': 'high'}
                    outgoing, logs = await self.capture(payload)
                    self.assertNotIn('thinking', outgoing)
                    self.assertEqual(outgoing['reasoning_effort'], 'none')
                    self.assertTrue(any('pro_no_reasoning_payload' in line for line in logs))
                    self.assertFalse(any('opus55_adaptive_compat' in line for line in logs))


if __name__ == '__main__':
    unittest.main()
