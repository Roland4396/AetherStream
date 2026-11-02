"""Managed pipeline acceptance; fake credentials and mock-only inference."""
import asyncio
import base64
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from starlette.requests import Request

from aetherstream.api import app
from aetherstream.api.channel_gateway import chat_completions, apply_request_policy
from aetherstream.routing.channel_policy import policy_scope, current_policy
from aetherstream.streaming.modes import upstream_stream_mode, collect_chat_stream
from aetherstream.upstreams import openai_chat_completions, openai_responses
from aetherstream.upstreams.anthropic_messages import transport
from aetherstream.streaming.upstream_close import close_upstream_stream

REAL_CLIENT = httpx.AsyncClient


def policy(**changes):
    value = {'streaming': {'nonstream_to_stream': False, 'force_upstream': 'auto', 'exact_dedupe': False},
             'early_stop': {'enabled': True, 'action': 'disconnect', 'tags': ['<CUT>'], 'case_sensitive': True},
             'injections': {}, 'filters': {}, 'request': {}, 'adapter': {}, 'runtime': {}}
    for key, change in changes.items():
        value[key].update(change)
    return value


def plan(model='free/test', transport='chat', settings=None):
    return {'model': model, 'channel': 'arbitrary-name', 'route': 'arbitrary-route',
            'revision': 'offline-v1', 'transport': transport, 'policy': settings or policy(),
            'base_url': 'http://acc.invalid:3200', 'api_key': 'service-test'}


def request(payload):
    result = Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions',
                      'headers': [], 'client': ('127.0.0.1', 1234)})
    result._body = json.dumps(payload).encode()
    return result


def chat_json(model):
    return {'id': 'fake', 'object': 'chat.completion', 'model': model,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK',
                         'tool_calls': [{'id': 'tool-1', 'type': 'function', 'function': {'name': 'sum', 'arguments': '{"a":1}'}}]},
                         'finish_reason': 'tool_calls'}],
            'usage': {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}}


def sse(data):
    return ('data: ' + json.dumps(data) + '\n\n').encode()


def events(protocol, *, stop=False):
    text = 'visible<CUT>not-forwarded' if stop else 'OK'
    if protocol == 'chat':
        out = [sse({'id': 'mock', 'model': 'wire-model', 'choices': [{'delta': {'reasoning_content': '<CUT> is only a thought'}, 'finish_reason': None}]}),
               sse({'id': 'mock', 'model': 'wire-model', 'choices': [{'delta': {'content': text}, 'finish_reason': None}]})]
        if not stop:
            out += [sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'total_tokens': 6}}), b'data: [DONE]\n\n']
        return out
    if protocol == 'messages':
        out = [sse({'type': 'message_start', 'message': {'id': 'mock', 'model': 'wire-model', 'usage': {'input_tokens': 4}}}),
               sse({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
               sse({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})]
        if not stop:
            out += [sse({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}}),
                    sse({'type': 'message_stop'})]
        return out
    out = [sse({'type': 'response.created', 'response': {'id': 'fake'}}),
           sse({'type': 'response.output_text.delta', 'delta': text})]
    if not stop:
        out.append(sse({'type': 'response.completed', 'response': {'status': 'completed', 'output': [],
                                                                'usage': {'input_tokens': 4, 'output_tokens': 2}}}))
    return out


class ControlledStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, wait_after=False):
        self.chunks = chunks
        self.wait_after = wait_after
        self.closed = False
        self.cancelled = False
        self.reads = 0
    async def __aiter__(self):
        try:
            for chunk in self.chunks:
                self.reads += 1
                yield chunk
                await asyncio.sleep(0)
            if self.wait_after:
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
    async def aclose(self):
        self.closed = True


class ChannelGatewayTests(unittest.IsolatedAsyncioTestCase):
    def deps(self, resolved):
        deps = copy.copy(app.chat_route_dependencies)
        deps.resolve_account_route = AsyncMock(return_value=resolved)
        deps.account_routing_enabled = lambda: True
        deps.active_stream_registry = type(app.active_stream_registry)(log=lambda _: None)
        deps.release_active_stream_caller = deps.active_stream_registry.release
        deps.replay_service = SimpleNamespace(prepare=lambda **_: None)
        deps.build_timed_claude_user_id = lambda **_: ('offline-user', 'offline-session', 'test', 60, 'offline-session-key')
        deps.get_claude_prompt_caching_settings = lambda: {'enabled': False}
        deps.get_claude_cache_keepalive_settings = lambda: {'enabled': False}
        deps.apply_claude_client_compat_request = lambda p: (p, {})
        deps.apply_claude_output_settings = lambda p: p
        deps.log = Mock()
        deps.save_request_log = Mock()
        async def once(**kwargs):
            return await kwargs['runner'](), False
        deps.run_exact_nonstream_once = AsyncMock(side_effect=once)
        return deps

    async def invoke(self, resolved, payload, handler, *, consume=True, downstream=None, asgi=False):
        deps = self.deps(resolved)
        sent = []
        def upstream(req):
            sent.append(req)
            return handler(req)
        def factory(*_, **__):
            return REAL_CLIENT(transport=httpx.MockTransport(upstream))
        with patch.object(httpx, 'AsyncClient', side_effect=factory):
            if downstream:
                from aetherstream.api.protocol_gateway import route_protocol_request
                response = await route_protocol_request(request(payload), deps, protocol=downstream)
            else:
                response = await chat_completions(request(payload), deps)
            if hasattr(response, 'body_iterator') and consume:
                if asgi:
                    messages = []
                    async def receive():
                        await asyncio.Event().wait()
                    async def send(message):
                        messages.append(message)
                    await asyncio.wait_for(response({'type': 'http', 'asgi': {'spec_version': '2.3'}}, receive, send), 2)
                    self.assertEqual(messages[0]['type'], 'http.response.start')
                    self.assertEqual(messages[-1], {'type': 'http.response.body', 'body': b'', 'more_body': False})
                    body = b''.join(m.get('body', b'') for m in messages)
                else:
                    body = b''.join([part async for part in response.body_iterator])
            else:
                body = getattr(response, 'body', b'')
        return response, body, sent, deps

    async def test_public_name_is_not_rewritten_or_routed_in_stream(self):
        response, body, sent, deps = await self.invoke(plan(), {'model': 'free/test', 'messages': [], 'stream': False},
                                                       lambda req: httpx.Response(200, json=chat_json('wire-model')))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(sent), 1)
        self.assertEqual(json.loads(sent[0].content)['model'], 'free/test')
        self.assertEqual(str(sent[0].url), 'http://acc.invalid:3200/v1/chat/completions')
        self.assertEqual(sent[0].headers['Authorization'], 'Bearer service-test')
        self.assertEqual(json.loads(body)['choices'][0]['message']['tool_calls'][0]['id'], 'tool-1')
        deps.resolve_account_route.assert_awaited_once_with('free/test')
        deps.run_exact_nonstream_once.assert_not_awaited()

    async def test_all_native_sse_protocols_really_close_without_fabricated_done(self):
        for protocol in ('chat', 'messages', 'responses'):
            with self.subTest(protocol=protocol):
                upstream = ControlledStream(events(protocol, stop=True), wait_after=True)
                response, body, sent, deps = await asyncio.wait_for(self.invoke(
                    plan(transport=protocol), {'model': 'free/test', 'messages': [{'role': 'user', 'content': 'Synthetic'}], 'stream': True},
                    lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=upstream),
                ), 2)
                self.assertTrue(upstream.closed)
                self.assertEqual(len(sent), 1)
                self.assertIn(b'visible', body)
                self.assertNotIn(b'not-forwarded', body)
                self.assertNotIn(b'[DONE]', body)
                self.assertNotIn(b'"finish_reason": "stop"', body)
                self.assertEqual(len(deps.active_stream_registry), 0)

    async def test_final_account_policy_overrides_the_requested_route(self):
        settings = {'enabled': False, 'action': 'disconnect', 'tags': ['<CUT>']}
        header = base64.urlsafe_b64encode(json.dumps(settings).encode()).decode()
        chunks = events('chat', stop=False)
        chunks[1] = sse({'choices': [{'delta': {'content': 'visible<CUT>kept'}, 'finish_reason': None}]})
        upstream = ControlledStream(chunks)
        _, body, _, _ = await self.invoke(plan(), {'model': 'free/test', 'messages': [], 'stream': True},
            lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream', 'x-acc-early-stop': header,
                                                 'x-account-pool-id': 'any-target-name'}, stream=upstream))
        self.assertIn(b'visible<CUT>kept', body)
        self.assertIn(b'[DONE]', body)
        self.assertTrue(upstream.closed)

    async def test_one_conversion_switch_for_every_stream_capable_protocol(self):
        for protocol in ('chat', 'messages', 'responses'):
            with self.subTest(protocol=protocol):
                upstream = ControlledStream(events(protocol))
                resolved = plan(transport=protocol, settings=policy(streaming={'force_upstream': 'stream'}))
                response, body, sent, _ = await self.invoke(resolved, {'model': 'free/test', 'messages': [], 'stream': False},
                    lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=upstream))
                self.assertEqual(response.status_code, 200)
                self.assertTrue(json.loads(sent[0].content)['stream'])
                self.assertEqual(json.loads(body)['choices'][0]['message']['content'], 'OK')
                self.assertTrue(upstream.closed)

    async def test_true_json_upstream_is_replayed_by_the_shared_converter(self):
        resolved = plan(settings=policy(streaming={'force_upstream': 'json'}))
        _, body, sent, _ = await self.invoke(resolved, {'model': 'free/test', 'messages': [], 'stream': True},
                                            lambda _: httpx.Response(200, json=chat_json('wire-model')))
        self.assertFalse(json.loads(sent[0].content)['stream'])
        self.assertIn(b'"tool_calls"', body)
        self.assertIn(b'"total_tokens": 6', body)
        self.assertIn(b'[DONE]', body)

    async def test_json_upstream_429_is_a_stream_error_not_an_asgi_exception(self):
        resolved = plan(settings=policy(streaming={'force_upstream': 'json'}))
        upstream_error = {'message': 'Resource has been exhausted (e.g. check quota).',
                          'type': 'upstream_error', 'param': '', 'code': 429}
        response, body, sent, deps = await self.invoke(
            resolved, {'model': 'free/test', 'messages': [], 'stream': True},
            lambda _: httpx.Response(429, json={'error': upstream_error}),
            asgi=True,
        )
        self.assertEqual(response.status_code, 200)  # SSE headers already sent.
        errors = [json.loads(line[6:])['error'] for line in body.splitlines()
                  if line.startswith(b'data: {') and 'error' in json.loads(line[6:])]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['status'], 429)
        self.assertEqual(errors[0]['code'], 429)
        self.assertEqual(errors[0]['message'], upstream_error['message'])
        self.assertEqual(len(sent), 1)
        self.assertFalse(json.loads(sent[0].content)['stream'])
        self.assertNotIn(b'[DONE]', body)
        self.assertNotIn(b'"finish_reason": "stop"', body)
        self.assertEqual(len(deps.active_stream_registry), 0)
        deps.save_request_log.assert_called_once()
        self.assertEqual(deps.save_request_log.call_args.kwargs['error_type'], 'channel_upstream_error')
        self.assertIn('429', deps.save_request_log.call_args.kwargs['raw_sse'])
        self.assertTrue(any('nonstream_replay_error status=429' in call.args[0] for call in deps.log.call_args_list))

    async def test_json_upstream_failure_survives_other_public_stream_protocols(self):
        for downstream in ('responses', 'anthropic'):
            with self.subTest(downstream=downstream):
                _, body, sent, deps = await self.invoke(
                    plan(settings=policy(streaming={'force_upstream': 'json'})),
                    {'model': 'free/test', 'messages': [], 'input': 'Synthetic', 'stream': True},
                    lambda _: httpx.Response(429, json={'error': {'message': 'Synthetic quota exhausted', 'code': 429}}),
                    downstream=downstream, asgi=True,
                )
                self.assertIn(b'event: error', body)
                self.assertIn(b'Synthetic quota exhausted', body)
                self.assertNotIn(b'response.completed', body)
                self.assertNotIn(b'"stop_reason": "end_turn"', body)
                self.assertEqual(len(sent), 1)
                self.assertEqual(len(deps.active_stream_registry), 0)

    async def test_nonstream_early_exit_is_marked_not_claimed_natural_completion(self):
        upstream = ControlledStream(events('chat', stop=True), wait_after=True)
        resolved = plan(settings=policy(streaming={'force_upstream': 'stream'}))
        response, body, _, _ = await asyncio.wait_for(self.invoke(resolved, {'model': 'free/test', 'messages': [], 'stream': False},
            lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=upstream)), 2)
        self.assertEqual(response.status_code, 200, body)
        self.assertEqual(json.loads(body)['choices'][0]['finish_reason'], 'early_stop')
        self.assertTrue(json.loads(body)['proxy_termination']['upstream_closed'])
        self.assertTrue(upstream.closed)

    async def test_public_protocols_do_not_fabricate_terminal_events_after_early_stop(self):
        for downstream, terminal in (('responses', b'response.completed'), ('anthropic', b'message_stop')):
            for transport in ('chat', 'messages', 'responses'):
                with self.subTest(downstream=downstream, transport=transport):
                    upstream = ControlledStream(events(transport, stop=True), wait_after=True)
                    _, body, sent, _ = await self.invoke(plan(transport=transport),
                        {'model': 'free/test', 'messages': [], 'input': 'Synthetic', 'stream': True},
                        lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=upstream),
                        downstream=downstream)
                    self.assertIn(b'visible', body)
                    self.assertNotIn(terminal, body)
                    self.assertTrue(upstream.closed)
                    self.assertEqual(len(sent), 1)

    async def test_public_protocol_json_partial_is_marked_incomplete(self):
        for downstream in ('responses', 'anthropic'):
            upstream = ControlledStream(events('chat', stop=True), wait_after=True)
            cfg = policy(streaming={'force_upstream': 'stream'})
            response, body, _, _ = await self.invoke(plan(settings=cfg),
                {'model': 'free/test', 'messages': [], 'input': 'Synthetic', 'stream': False},
                lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=upstream),
                downstream=downstream)
            self.assertEqual(response.status_code, 200, body)
            value = json.loads(body)
            self.assertTrue(value['proxy_termination']['upstream_closed'])
            if downstream == 'responses':
                self.assertEqual(value['status'], 'incomplete')
            else:
                self.assertIsNone(value['stop_reason'])

    async def test_policy_deadline_survives_public_protocol_adaptation(self):
        from aetherstream.api.timed_routes import TimedSSEResponse
        for downstream in ('responses', 'anthropic'):
            response, _, sent, _ = await self.invoke(plan(settings=policy(streaming={'deadline_seconds': 1})),
                {'model': 'free/test', 'messages': [], 'input': 'Synthetic', 'stream': True},
                lambda _: self.fail('a returned response must not pre-open inference'),
                downstream=downstream, consume=False)
            self.assertIsInstance(response, TimedSSEResponse)
            self.assertTrue(hasattr(response.response, 'body_iterator'))
            self.assertEqual(sent, [])

    def test_closure_implementation_is_the_same_object_in_every_adapter(self):
        for method in (openai_chat_completions._close_upstream_stream,
                       openai_responses._close_upstream_stream, transport._close_upstream_stream):
            self.assertIs(method, close_upstream_stream)

    def test_stream_mode_truth_table_and_context_isolation(self):
        for downstream in (False, True):
            for convert in (False, True):
                self.assertEqual(upstream_stream_mode(policy(streaming={'nonstream_to_stream': convert}), downstream), downstream)
            self.assertFalse(upstream_stream_mode(policy(streaming={'force_upstream': 'json'}), downstream))
            self.assertTrue(upstream_stream_mode(policy(streaming={'force_upstream': 'stream'}), downstream))
        with policy_scope(policy()):
            self.assertTrue(current_policy()['early_stop']['enabled'])
            with policy_scope(policy(early_stop={'enabled': False})):
                self.assertFalse(current_policy()['early_stop']['enabled'])
            self.assertTrue(current_policy()['early_stop']['enabled'])
        self.assertIsNone(current_policy())

    def test_injection_and_field_operations_do_not_depend_on_channel_names(self):
        deps = self.deps(plan())
        original = {'model': 'generic', 'messages': [{'role': 'user', 'content': 'Synthetic'}, {'role': 'assistant', 'content': 'prefill'}],
                    'top_p': 0.9, 'temperature': 0.5}
        cfg = policy(injections={'assistant_prefill': True}, request={'drop_request_fields': ['temperature'], 'reasoning_effort': 'low'})
        output = apply_request_policy(original, cfg, deps, trace_id='offline')
        self.assertEqual(output['messages'][-1]['role'], 'user')
        self.assertEqual(len(original['messages']), 2)
        self.assertNotIn('temperature', output)
        self.assertEqual(output['top_p'], 0.9)
        self.assertEqual(output['reasoning_effort'], 'low')


if __name__ == '__main__':
    unittest.main()
