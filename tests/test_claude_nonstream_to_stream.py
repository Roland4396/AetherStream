"""Only toggle the existing client-nonstream -> upstream-stream conversion.

All HTTP uses MockTransport. No provider requests, live credentials or GPU work.
"""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from starlette.requests import Request

from aetherstream.api import app
from aetherstream.api.chat_routes import chat_completions, _claude_nonstream_to_stream_enabled
from aetherstream.api.messages_routes import anthropic_messages
from aetherstream.api.protocol_gateway import route_protocol_request
from aetherstream.runtime.flags import RuntimeFlags
from aetherstream.upstreams.anthropic_messages import collectors
from aetherstream.upstreams.anthropic_messages.types import AnthropicMessagesDeps

REAL_CLIENT = httpx.AsyncClient
FLAG = ('claude', 'stream', 'nonstream_to_stream')


def native_body():
    return {'id': 'msg_offline', 'type': 'message', 'role': 'assistant', 'model': 'claude-opus-4-6',
            'content': [{'type': 'text', 'text': 'OK'}], 'stop_reason': 'end_turn',
            'stop_sequence': None, 'usage': {'input_tokens': 3, 'output_tokens': 2}}


def chat_body(model):
    return {'id': 'chatcmpl-offline', 'object': 'chat.completion', 'model': model,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}


def sse_body(native, model):
    if native:
        events = [
            {'type': 'message_start', 'message': native_body()},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'OK'}},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'},
             'usage': {'input_tokens': 3, 'output_tokens': 2}},
            {'type': 'message_stop'},
        ]
    else:
        events = [{'id': 'chatcmpl-offline', 'object': 'chat.completion.chunk', 'model': model,
                   'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'OK'},
                                'finish_reason': 'stop'}]},
                  {'object': 'chat.completion.chunk', 'choices': [], 'usage': chat_body(model)['usage']}]
    return ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + ('' if native else 'data: [DONE]\n\n')


def make_request(payload, path='/v1/chat/completions'):
    request = Request({'type': 'http', 'method': 'POST', 'path': path,
                       'headers': [], 'client': ('127.0.0.1', 1234)})
    request._body = json.dumps(payload).encode()
    return request


class ClaudeConversionRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, *, flag=None, stream=False, kind='directory', protocol='chat',
                       model=None, handler=None, updates=None):
        model = model or {'directory': '[m3]claude-opus-4-6',
                          'free': 'free/[m3]claude-opus-4-6',
                          'native': 'claude-opus-4-6',
                          'passthrough': 'vendor/claude-opus-4-6'}[kind]
        deps = copy.copy(app.chat_route_dependencies)
        deps.active_stream_registry = type(app.active_stream_registry)(log=lambda _: None)
        deps.release_active_stream_caller = deps.active_stream_registry.release
        logs, saved, captured = [], [], []
        deps.log = logs.append
        deps.save_request_log = lambda *a, **k: saved.append(k)
        deps.replay_service = SimpleNamespace(prepare=lambda **k: None)
        deps._runtime_lookup = lambda *keys: flag if keys == FLAG else None
        route = {'name': 'offline-claude', 'base_url': 'http://offline.invalid/v1', 'api_key': 'offline'}
        deps.resolve_openai_compatible_route = AsyncMock(return_value=route if kind == 'directory' else None)
        deps.resolve_model_name_passthrough_upstream = lambda _: route if kind == 'passthrough' else None
        deps.get_claude_upstream_for_provider = lambda _: ('offline', 'http://offline.invalid/v1')
        deps.get_free_openai_upstream = lambda: ('offline', 'http://offline.invalid/v1')
        deps.build_timed_claude_user_id = lambda **k: ('offline-user', 'offline-session', 'test', 60, 'test')
        deps.get_claude_prompt_caching_settings = lambda: {'enabled': False}
        deps.get_claude_cache_keepalive_settings = lambda: {'enabled': False}
        deps.apply_claude_client_compat_request = lambda p: (p, {})
        deps.apply_claude_output_settings = lambda p: p
        deps.run_exact_nonstream_once = AsyncMock(side_effect=lambda **k: None)
        # Keep existing Gemini dedupe observable, with no persistent test state.
        async def run_once(**kwargs):
            return await kwargs['runner'](), False
        deps.run_exact_nonstream_once.side_effect = run_once
        payload = {'model': model, 'stream': stream, 'max_tokens': 99,
                   'messages': [{'role': 'user', 'content': 'Offline synthetic request.'}],
                   **(updates or {})}
        if protocol == 'responses':
            payload['input'] = payload.pop('messages')
            payload['max_output_tokens'] = payload.pop('max_tokens')
        original = copy.deepcopy(payload)

        async def dispatch(request):
            self.assertEqual(request.url.host, 'offline.invalid')
            self.assertEqual(request.method, 'POST')
            outgoing = json.loads(request.content)
            captured.append(outgoing)
            if handler:
                return await handler(request)
            native = request.url.path.endswith('/messages')
            if outgoing['stream']:
                return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                      text=sse_body(native, outgoing['model']))
            return httpx.Response(200, json=native_body() if native else chat_body(outgoing['model']))

        with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
                transport=httpx.MockTransport(dispatch), **kw)):
            request = make_request(payload)
            response = (await chat_completions(request, deps) if protocol == 'chat'
                        else await route_protocol_request(request, deps, protocol=protocol))
            if hasattr(response, 'body_iterator'):
                parts = [part async for part in response.body_iterator]
                body = b''.join(part.encode() if isinstance(part, str) else part for part in parts)
            else:
                body = response.body
        self.assertEqual(payload, original)
        return response, body, captured, deps, logs, saved

    async def test_flag_matrix_all_existing_claude_routes_and_protocols(self):
        for kind in ('directory', 'free', 'native', 'passthrough'):
            for protocol in ('chat', 'anthropic', 'responses'):
                for streaming in (False, True):
                    for flag in (None, True, False):
                        with self.subTest(kind=kind, protocol=protocol, stream=streaming, flag=flag):
                            resp, body, sent, deps, logs, _ = await self.run_case(
                                flag=flag, stream=streaming, kind=kind, protocol=protocol)
                            self.assertEqual(resp.status_code, 200, (body, logs))
                            self.assertEqual(len(sent), 1)
                            self.assertIs(sent[0]['stream'], streaming or flag is not False)
                            self.assertIn(b'OK', body)
                            if not streaming:
                                self.assertIsInstance(json.loads(body), dict)
                                self.assertNotIn(b'data:', body)
                            if not streaming and (kind == 'native' or flag is False):
                                deps.run_exact_nonstream_once.assert_called_once()
                            else:
                                deps.run_exact_nonstream_once.assert_not_called()

    async def test_missing_flag_and_enabled_have_identical_upstream_payloads(self):
        for kind in ('directory', 'free', 'native', 'passthrough'):
            for stream in (False, True):
                old = await self.run_case(flag=None, stream=stream, kind=kind)
                enabled = await self.run_case(flag=True, stream=stream, kind=kind)
                self.assertEqual(old[2], enabled[2])

    async def test_disabling_changes_only_stream_field_of_upstream_request(self):
        for kind in ('directory', 'free', 'native', 'passthrough'):
            enabled = await self.run_case(flag=True, kind=kind)
            disabled = await self.run_case(flag=False, kind=kind)
            self.assertEqual(dict(enabled[2][0], stream=False), disabled[2][0])

    async def test_aliases_and_opus55_parameters_survive(self):
        for model in ('[m1]claude-opus-5-5', 'anthropic-claude-opus-5-5', 'agy-claude-opus-4-6'):
            resp, _, sent, *_ = await self.run_case(flag=False, model=model,
                                                  updates={'thinking': {'type': 'adaptive'}})
            self.assertEqual(resp.status_code, 200)
            self.assertIs(sent[0]['stream'], False)
            self.assertEqual(sent[0]['model'], model)
            if '5-5' in model:
                self.assertEqual(sent[0]['thinking'], {'type': 'adaptive'})

    async def test_channel_delimiter_aliases_obey_switch_and_log_actual_payload(self):
        aliases = ('「anti5」claude-opus-5-5', '【anti5】claude-opus-5-5',
                   '『anti5』claude-opus-5-5', '[AWSB]claude-opus-4-6',
                   '[kiro2]claude-opus-5-5', '[官逆]claude-opus-5-5')
        for alias in aliases:
            for kind in ('directory', 'free', 'passthrough'):
                model = 'free/' + alias if kind == 'free' else alias
                for protocol in ('chat', 'anthropic', 'responses'):
                    for flag in (None, True, False):
                        with self.subTest(model=model, kind=kind, protocol=protocol, flag=flag):
                            resp, body, sent, _, logs, saved = await self.run_case(
                                flag=flag, model=model, kind=kind, protocol=protocol,
                                updates={'max_tokens':50000, 'presence_penalty':0,
                                         'frequency_penalty':0, 'stop':['<end>']})
                            self.assertEqual(resp.status_code, 200, (body, logs))
                            self.assertEqual(len(sent), 1)
                            self.assertIs(sent[0]['stream'], flag is not False)
                            self.assertEqual(sent[0]['model'], alias if kind == 'free' else model)
                            self.assertEqual(saved[-1]['request_payload'], sent[0])
                            self.assertIs(saved[-1]['stream'], False)
                            self.assertTrue(any('claude_nonstream_to_stream ' in line for line in logs))
                            self.assertIn(b'OK', body)
                            self.assertNotIn(b'data:', body)

    async def test_channel_alias_client_streaming_stays_streaming(self):
        for protocol in ('chat', 'anthropic', 'responses'):
            resp, body, sent, *_ = await self.run_case(
                flag=False, stream=True, model='「anti5」claude-opus-5-5', protocol=protocol)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(sent), 1)
            self.assertIs(sent[0]['stream'], True)
            self.assertIn(b'data:', body)

    async def test_conversion_does_not_match_claude_substrings_inside_other_names(self):
        for model in ('notclaude-opus-5-5', 'custom_claude-opus-5-5'):
            on = await self.run_case(flag=True, model=model)
            off = await self.run_case(flag=False, model=model)
            self.assertEqual(on[2], off[2])
            self.assertEqual(off[0].status_code, 200)
            self.assertIs(off[2][0]['stream'], True)

    async def test_gemini_and_other_models_are_unchanged(self):
        for model in ('gemini-3.1-pro-preview', 'gpt-5.5', 'GLM-5.2'):
            for stream in (False, True):
                on = await self.run_case(flag=True, model=model, stream=stream)
                off = await self.run_case(flag=False, model=model, stream=stream)
                self.assertEqual(on[2], off[2])
                self.assertEqual(on[0].status_code, 200, on[1])
                self.assertEqual(off[0].status_code, 200, off[1])

    async def test_plain_openai_result_preserves_fields_and_tools(self):
        body = chat_body('[m3]claude-opus-4-6')
        body['choices'][0]['message']['tool_calls'] = [{'id': 'c1', 'type': 'function',
            'function': {'name': 'lookup', 'arguments': '{"x":1}'}}]
        body['choices'][0]['message']['reasoning_content'] = 'offline reasoning'
        body['choices'][0]['finish_reason'] = 'tool_calls'
        body['extra_provider_field'] = 'preserve'
        async def handler(_):
            return httpx.Response(200, json=body)
        resp, raw, sent, *_ = await self.run_case(flag=False, handler=handler)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(json.loads(raw), body)
        self.assertEqual(len(sent), 1)

    async def test_plain_error_does_not_retry_or_return_empty_success(self):
        for kind in ('directory', 'native'):
            for status, body in [(429, {'error': {'message': 'offline quota'}}),
                                 (200, {'error': {'message': 'offline error'}}),
                                 (200, []), (200, {'unexpected': True})]:
                async def handler(_, status=status, body=body):
                    return httpx.Response(status, json=body)
                with self.subTest(kind=kind, status=status, body=body):
                    resp, _, sent, *_ = await self.run_case(flag=False, kind=kind, handler=handler)
                    self.assertGreaterEqual(resp.status_code, 400)
                    self.assertEqual(len(sent), 1)
                    self.assertIs(sent[0]['stream'], False)

    async def test_off_cancellation_propagates_through_route(self):
        started, closed = asyncio.Event(), asyncio.Event()
        class PendingBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                yield b'{'
                await asyncio.Event().wait()
            async def aclose(self):
                closed.set()
        calls = []
        async def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, stream=PendingBody())
        task = asyncio.create_task(self.run_case(flag=False, handler=handler))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]['stream'], False)

    async def test_legacy_native_route_also_obeys_switch(self):
        for flag in (None, True, False):
            deps = copy.copy(app.messages_route_dependencies)
            deps._runtime_lookup = lambda *keys: flag if keys == FLAG else None
            deps.log = lambda _: None
            deps.save_request_log = lambda *a, **k: None
            deps.get_claude_upstream_for_provider = lambda _: ('offline', 'http://offline.invalid/v1')
            calls = []
            async def collect(**kwargs):
                calls.append(kwargs['upstream_stream'])
                return native_body(), ''
            deps.collect_anthropic_messages_response = collect
            with patch('httpx.AsyncClient.send', side_effect=AssertionError('External network forbidden')):
                resp = await anthropic_messages(make_request({
                    'model': 'claude-opus-4-6', 'stream': False,
                    'messages': [{'role': 'user', 'content': 'Offline.'}],
                }), deps)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(calls, [flag is not False])

    def test_default_and_hot_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'flags.json'
            path.write_text('{}')
            flags = RuntimeFlags(path=str(path), poll_sec=0, log=lambda _: None)
            deps = SimpleNamespace(_runtime_lookup=flags.lookup)
            self.assertTrue(_claude_nonstream_to_stream_enabled(deps))
            for value, expected in [(False, False), (True, True), ('false', False)]:
                path.write_text(json.dumps({'claude': {'stream': {'nonstream_to_stream': value}}}))
                flags._cache_mtime = None
                self.assertEqual(_claude_nonstream_to_stream_enabled(deps), expected)


class NativeCollectorSwitchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.deps = AnthropicMessagesDeps(
            log=lambda _: None, save_request_log=lambda *a, **k: None,
            build_openai_sse_error=lambda *a: b'', has_stop_tag=lambda _: False,
            find_stop_tag=lambda _: -1, fmt_ms=lambda *a: '0ms', release_caller=lambda *a: None)
        self.kwargs = dict(url='http://offline.invalid/v1/messages', request_data={
            'model': 'claude-opus-4-6', 'stream': False, 'messages': []}, headers={},
            model='claude-opus-4-6', trace_id='offline', timeout=httpx.Timeout(2),
            max_raw_sse_bytes=100000, deps=self.deps)

    async def test_default_and_off_both_native_collectors(self):
        for chat in (False, True):
            for options in ({}, {'upstream_stream': True}, {'upstream_stream': False}):
                sent = []
                async def handler(request):
                    payload = json.loads(request.content)
                    sent.append(payload)
                    return (httpx.Response(200, text=sse_body(True, payload['model']))
                            if payload['stream'] else httpx.Response(200, json=native_body()))
                with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
                        transport=httpx.MockTransport(handler), **kw)):
                    kwargs = dict(self.kwargs, **options)
                    if chat:
                        result = await collectors.collect_anthropic_messages_as_chat_completion(messages=[], **kwargs)
                        self.assertEqual(result[0], 'OK')
                        self.assertEqual(result[2]['total_tokens'], 5)
                    else:
                        result = await collectors.collect_anthropic_messages_response(**kwargs)
                        self.assertEqual(result[0]['content'], [{'type': 'text', 'text': 'OK'}])
                    self.assertEqual(len(sent), 1)
                    self.assertIs(sent[0]['stream'], options.get('upstream_stream', True))
        self.assertIs(self.kwargs['request_data']['stream'], False)

    async def test_native_plain_tools_and_usage(self):
        body = native_body()
        body['content'].append({'type': 'tool_use', 'id': 'toolu_offline', 'name': 'lookup', 'input': {'x': 1}})
        body['stop_reason'] = 'tool_use'
        async def handler(_):
            return httpx.Response(200, json=body)
        with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
                transport=httpx.MockTransport(handler), **kw)):
            result = await collectors.collect_anthropic_messages_as_chat_completion(
                messages=[], upstream_stream=False, **self.kwargs)
            self.assertEqual(result[3], 'tool_calls')
            self.assertEqual(json.loads(result[5][0]['function']['arguments']), {'x': 1})
            raw_result = await collectors.collect_anthropic_messages_response(upstream_stream=False, **self.kwargs)
            self.assertEqual(raw_result[0], body)

    async def test_off_cancellation_closes_response_without_retry(self):
        for chat in (False, True):
            started, closed = asyncio.Event(), asyncio.Event()
            calls = []
            class PendingBody(httpx.AsyncByteStream):
                async def __aiter__(self):
                    started.set()
                    yield b'{'
                    await asyncio.Event().wait()
                async def aclose(self):
                    closed.set()
            async def handler(request):
                calls.append(json.loads(request.content))
                return httpx.Response(200, stream=PendingBody())
            with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
                    transport=httpx.MockTransport(handler), **kw)):
                operation = (collectors.collect_anthropic_messages_as_chat_completion(
                    messages=[], upstream_stream=False, **self.kwargs) if chat else
                    collectors.collect_anthropic_messages_response(upstream_stream=False, **self.kwargs))
                task = asyncio.create_task(operation)
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(closed.is_set())
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0]['stream'], False)


if __name__ == '__main__':
    unittest.main()
