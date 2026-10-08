"""Offline final-wire assertions, not only a route-configuration check."""
import copy
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from starlette.requests import Request

from aetherstream.api import app
from aetherstream.api.chat_routes import chat_completions
from aetherstream.api.protocol_gateway import route_protocol_request
from aetherstream.features.glm_thinking import apply_glm52_official_thinking, is_glm52_model
from aetherstream.observability.logging import ProxyLogger
from aetherstream.upstreams import openai_chat_completions as chat


REAL_CLIENT = httpx.AsyncClient
ALIASES = (
    'glm-5.2', '[官]glm-5.2', '[G]GLM-5.2', '[OR]GLM-5.2',
    'vendor/zai-org/GLM-5.2', 'free/glm-5.2', 'glm-5.2-local',
    'GLM5.2', 'glm_5_2', 'glm-5-2', 'glm.5.2', 'glm 5.2',
    '「新渠道」GLM-5.2-fast', 'glm-5.2-20261008', 'glm-5.2:thinking',
    '【备用】ＧＬＭ５．２',
)
OTHERS = (None, '', 'glm-5.1', 'glm-5.3', 'glm-5.20', 'glm-15.2',
          'glm-5.21', 'notglm-5.2', 'claude-opus-5-5', 'gpt-5.2', 'kimi-k3')


def disabled_payload(model):
    return {
        'model': model, 'stream': True, 'max_tokens': 987,
        'messages': [{'role': 'user', 'content': 'Synthetic offline request.'}],
        'temperature': .7, 'top_p': .9,
        'thinking': {'type': 'disabled'}, 'reasoning_effort': 'none',
        'effort': 'low', 'enable_thinking': False,
        'chat_template_kwargs': {'enable_thinking': False, 'thinking': False},
        'reasoning': {'effort': 'none'}, 'output_config': {'effort': 'low'},
    }


def stream_reply(model):
    events = [
        {'id': 'offline', 'model': model, 'choices': [{'index': 0, 'delta': {'reasoning_content': 'Synthetic reasoning.'}, 'finish_reason': None}]},
        {'id': 'offline', 'model': model, 'choices': [{'index': 0, 'delta': {'content': 'OK'}, 'finish_reason': None}]},
        {'id': 'offline', 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
    ]
    return ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n'


def json_reply(model):
    return {'id': 'offline', 'model': model, 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': 'OK', 'reasoning_content': 'Synthetic reasoning.'},
            'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}}


def make_deps(logs, saved):
    return chat.ChatCompletionsUpstreamDeps(
        log=logs.append, save_request_log=lambda *a, **kw: saved.append(copy.deepcopy(kw)),
        build_openai_sse_error=lambda *a, **kw: b'UNEXPECTED_ERROR',
        has_stop_tag=lambda _: False, find_stop_tag=lambda _: -1,
        fmt_ms=lambda *_: '0ms', release_caller=lambda *_: None,
    )


class GlmPolicyTests(unittest.TestCase):
    def test_channel_prefix_case_separator_and_suffix_do_not_disable_policy(self):
        for name in ALIASES:
            with self.subTest(model=name):
                self.assertTrue(is_glm52_model(name))
                payload = disabled_payload(name)
                self.assertTrue(apply_glm52_official_thinking(payload))
                self.assertEqual(payload['thinking'], {'type': 'enabled'})
                self.assertEqual(payload['reasoning_effort'], 'max')
                for field in ('effort', 'enable_thinking', 'chat_template_kwargs', 'reasoning', 'output_config'):
                    self.assertNotIn(field, payload)
                self.assertEqual(payload['model'], name)
                self.assertEqual(apply_glm52_official_thinking(payload), [])

    def test_other_model_families_are_byte_for_byte_unchanged(self):
        for name in OTHERS:
            with self.subTest(model=name):
                self.assertFalse(is_glm52_model(name))
                payload = disabled_payload(name)
                original = json.dumps(payload)
                self.assertEqual(apply_glm52_official_thinking(payload), [])
                self.assertEqual(json.dumps(payload), original)

    def test_preserves_messages_tools_sampling_and_official_history_control(self):
        payload = disabled_payload('GLM5.2')
        payload['thinking'] = {'type': 'disabled', 'clear_thinking': False, 'budget_tokens': 100}
        payload['tools'] = [{'type': 'function', 'function': {'name': 'offline'}}]
        payload['chat_template_kwargs']['other'] = 'keep'
        payload['output_config']['format'] = {'type': 'json_schema'}
        payload['reasoning']['summary'] = 'auto'
        before = copy.deepcopy(payload)
        previous_thinking = payload['thinking']
        previous_template = payload['chat_template_kwargs']
        apply_glm52_official_thinking(payload)
        self.assertEqual(payload['thinking'], {'type': 'enabled', 'clear_thinking': False})
        self.assertEqual(payload['chat_template_kwargs'], {'other': 'keep'})
        self.assertEqual(payload['output_config'], {'format': {'type': 'json_schema'}})
        self.assertEqual(payload['reasoning'], {'summary': 'auto'})
        for key in ('model', 'messages', 'tools', 'temperature', 'top_p', 'max_tokens', 'stream'):
            self.assertEqual(payload[key], before[key])
        self.assertEqual(previous_thinking, before['thinking'])
        self.assertEqual(previous_template, before['chat_template_kwargs'])

    def test_absent_boolean_or_legacy_thinking_becomes_official_object(self):
        for thinking in (None, False, True, 'disabled', {'type': 'adaptive'}):
            payload = {'model': '[官]glm-5.2', 'thinking': thinking}
            apply_glm52_official_thinking(payload)
            self.assertEqual(payload['thinking'], {'type': 'enabled'})
            self.assertEqual(payload['reasoning_effort'], 'max')

    def test_policy_diagnostic_is_visible_without_verbose_prompt_logging(self):
        logger = ProxyLogger.__new__(ProxyLogger)
        logger.verbose_trace = False
        self.assertTrue(logger._should_emit('[TRACE offline] glm52_official_thinking thinking=enabled reasoning_effort=max'))


class GlmWireTests(unittest.IsolatedAsyncioTestCase):
    def assert_official(self, outgoing):
        self.assertEqual(outgoing['thinking'], {'type': 'enabled'})
        self.assertEqual(outgoing['reasoning_effort'], 'max')
        for field in ('enable_thinking', 'effort', 'chat_template_kwargs', 'reasoning', 'output_config'):
            self.assertNotIn(field, outgoing)

    async def test_all_live_chat_transports_enforce_final_wire_and_log_payload(self):
        functions = (chat.forward_chat_completions_stream, chat.collect_chat_completions_stream,
                     chat.collect_chat_completions_nonstream, chat.replay_chat_completions_nonstream_as_stream)
        for fn in functions:
            for name in ('[官]glm-5.2', '[G]GLM-5.2', 'glm-5.2-local', 'vendor/GLM5.2-fast', 'claude-opus-4-6'):
                with self.subTest(transport=fn.__name__, model=name):
                    sent, logs, saved = [], [], []
                    payload = disabled_payload(name)
                    original = copy.deepcopy(payload)

                    def dispatch(request):
                        self.assertEqual(request.url.host, 'offline.invalid')
                        body = json.loads(request.content)
                        sent.append(body)
                        if body.get('stream'):
                            return httpx.Response(200, text=stream_reply(name), headers={'content-type': 'text/event-stream'})
                        return httpx.Response(200, json=json_reply(name))

                    def factory(*args, **kwargs):
                        return REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)

                    params = dict(url='http://offline.invalid/v1/chat/completions', request_data=payload,
                                  headers={}, timeout=httpx.Timeout(2), max_raw_sse_bytes=9999,
                                  deps=make_deps(logs, saved), model=name, messages=payload['messages'],
                                  trace_id='offline', enable_early_stop=False)
                    params = {k: v for k, v in params.items() if k in inspect.signature(fn).parameters}
                    with patch('httpx.AsyncClient', side_effect=factory):
                        result = fn(**params)
                        if inspect.isasyncgen(result):
                            chunks = [c async for c in result]
                            self.assertNotIn(b'UNEXPECTED_ERROR', b''.join(chunks))
                        else:
                            await result
                    self.assertEqual(len(sent), 1)
                    if is_glm52_model(name):
                        self.assert_official(sent[0])
                        self.assert_official(payload)  # Caller-side completion log sees real wire controls.
                        for record in saved:
                            self.assert_official(record['request_payload'])
                        self.assertTrue(any('glm52_official_thinking' in line for line in logs))
                    else:
                        expected = dict(original, stream=sent[0]['stream'])
                        self.assertEqual(sent[0], expected)
                        self.assertEqual(payload, original)
                        self.assertFalse(any('glm52_official_thinking' in line for line in logs))

    async def test_route_overrides_and_all_downstream_protocols_cannot_turn_family_off(self):
        scenarios = (('[官]glm-5.2', 'directory'), ('[G]GLM-5.2', 'directory'),
                     ('glm-5.2-local', 'directory'), ('free/GLM5.2-fast', 'free'),
                     ('vendor/GLM_5_2', 'passthrough'))
        for name, kind in scenarios:
            for protocol in ('chat', 'responses', 'anthropic'):
                for streaming in (False, True):
                    with self.subTest(model=name, route=kind, protocol=protocol, stream=streaming):
                        deps = copy.copy(app.chat_route_dependencies)
                        deps.active_stream_registry = type(app.active_stream_registry)(log=lambda _: None)
                        deps.release_active_stream_caller = deps.active_stream_registry.release
                        logs, saved, sent = [], [], []
                        deps.log = logs.append
                        deps.save_request_log = lambda *a, **kw: saved.append(copy.deepcopy(kw))
                        deps.build_chat_completions_upstream_deps = lambda: make_deps(logs, saved)
                        deps.replay_service = SimpleNamespace(prepare=lambda **kw: None)
                        route = {'name': 'pro-renamed-channel', 'base_url': 'http://offline.invalid/v1',
                                 'api_key': 'offline', 'enable_thinking': False, 'thinking': False,
                                 'reasoning_effort': 'none', 'effort': 'low',
                                 'drop_request_fields': ['thinking', 'reasoning_effort']}
                        deps.resolve_openai_compatible_route = AsyncMock(return_value=route if kind == 'directory' else None)
                        deps.resolve_model_name_passthrough_upstream = lambda _: route if kind == 'passthrough' else None
                        deps.get_claude_upstream_for_provider = lambda _: ('offline', 'http://offline.invalid/v1')
                        payload = disabled_payload(name)
                        payload['stream'] = streaming
                        if protocol == 'responses':
                            payload['input'] = payload.pop('messages')
                            payload['max_output_tokens'] = payload.pop('max_tokens')
                        original = copy.deepcopy(payload)
                        path = {'chat': '/v1/chat/completions', 'responses': '/v1/responses', 'anthropic': '/v1/messages'}[protocol]
                        request = Request({'type': 'http', 'method': 'POST', 'path': path,
                                           'headers': [], 'client': ('127.0.0.1', 4567)})
                        request._body = json.dumps(payload).encode()

                        def dispatch(req):
                            self.assertEqual(req.url.host, 'offline.invalid')
                            body = json.loads(req.content)
                            sent.append(body)
                            return httpx.Response(200, text=stream_reply(body['model']), headers={'content-type': 'text/event-stream'})

                        def factory(*args, **kwargs):
                            return REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)

                        with patch('httpx.AsyncClient', side_effect=factory):
                            result = (await chat_completions(request, deps) if protocol == 'chat' else
                                      await route_protocol_request(request, deps, protocol=protocol))
                            if hasattr(result, 'body_iterator'):
                                async for _ in result.body_iterator:
                                    pass
                        self.assertEqual(result.status_code, 200, (logs, getattr(result, 'body', b'')))
                        self.assertEqual(len(sent), 1, logs)
                        self.assert_official(sent[0])
                        self.assertEqual(sent[0]['model'], name.removeprefix('free/'))
                        self.assertEqual(sent[0]['max_tokens'], 987)
                        self.assertEqual(payload, original)
                        self.assertIn('Synthetic offline request.', json.dumps(sent[0]['messages']))


if __name__ == '__main__':
    unittest.main()
