"""Initial Anthropic text participates in output and real socket early-stop.

All streams are local MockTransport fixtures; hanging tails prove the adapter
closes instead of waiting for upstream completion or relying on test cancellation.
"""
import asyncio
import json
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import httpx

from aetherstream.routing.channel_policy import policy_scope
from aetherstream.observability.logging import ProxyLogger
from aetherstream.upstreams.anthropic_messages import (
    forward_anthropic_messages_as_chat_stream,
    forward_anthropic_messages_stream,
    collect_anthropic_messages_as_chat_completion,
    collect_anthropic_messages_response,
    collect_anthropic_chat_completion_from_raw_sse,
    replay_anthropic_chat_stream,
    AnthropicMessagesDeps,
)
from aetherstream.upstreams.anthropic_messages.protocol import _extract_anthropic_sse_delta
import test_channel_gateway as gateway
from test_channel_gateway import ControlledStream, plan, policy, sse

REAL_CLIENT = httpx.AsyncClient
TAG = '<CUT>'
MODES = ('chat_stream', 'raw_stream', 'chat_collect', 'raw_collect')


def message(content=None):
    value = {'id': 'initial-text', 'model': 'claude-offline', 'usage': {'input_tokens': 3}}
    if content is not None:
        value['content'] = content
    return {'type': 'message_start', 'message': value}


def start(text='', index=0):
    return {'type': 'content_block_start', 'index': index,
            'content_block': {'type': 'text', 'text': text}}


def delta(text, index=0):
    return {'type': 'content_block_delta', 'index': index,
            'delta': {'type': 'text_delta', 'text': text}}


def text_block(text):
    return {'type': 'text', 'text': text}


def finish():
    return [{'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'},
             'usage': {'output_tokens': 4}}, {'type': 'message_stop'}]


def visible(body):
    return ''.join(json.loads(line[6:])['choices'][0]['delta'].get('content', '')
                   for line in body.splitlines() if line.startswith(b'data: {'))


def deps_for(logs):
    return AnthropicMessagesDeps(
        log=logs.append, save_request_log=Mock(),
        build_openai_sse_error=lambda *args: b'fixture-error',
        has_stop_tag=lambda text: TAG in text, find_stop_tag=lambda text: text.find(TAG),
        fmt_ms=lambda begin, end=None: f'{((end or time.perf_counter()) - begin)*1000:.1f}ms',
        release_caller=lambda *args: None,
    )


class InitialTextProtocolTests(unittest.TestCase):
    def test_initial_message_and_block_text(self):
        self.assertEqual(_extract_anthropic_sse_delta(message([text_block('A'), text_block('B')])), ('AB', 'content'))
        self.assertEqual(_extract_anthropic_sse_delta(start('C', 9)), ('C', 'content'))
        self.assertEqual(_extract_anthropic_sse_delta(delta('D')), ('D', 'content'))

    def test_empty_malformed_and_nontext_initial_blocks_are_not_text(self):
        for event in [message(), message(None), message({'text': TAG}),
                      message([{'type': 'thinking', 'thinking': TAG}, {'type': 'tool_use', 'input': {'text': TAG}}]),
                      {'type': 'message_start', 'message': []},
                      {'type': 'content_block_start', 'content_block': 'bad'},
                      {'type': 'content_block_start', 'content_block': {'type': 'text', 'text': 123}},
                      {'type': 'content_block_start', 'content_block': {'type': 'thinking', 'thinking': TAG}},
                      {'type': 'content_block_delta', 'delta': {'type': 'input_json_delta', 'partial_json': TAG}}]:
            with self.subTest(event=event):
                self.assertEqual(_extract_anthropic_sse_delta(event)[0], '')

    def test_real_close_summary_is_visible_without_verbose_logging(self):
        logger = ProxyLogger(debug=False, log_dir=tempfile.gettempdir())
        logger.verbose_trace = False
        self.assertTrue(logger._should_emit('[TRACE offline] early_stop_upstream_close reason=stop_tag response_closed=true client_closed=true'))
        self.assertFalse(logger._should_emit('[TRACE offline] early_stop_policy enabled=true source=stream_policy'))


class InitialTextAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def run_adapter(self, mode, events, *, hanging=False, settings=None, request=None):
        upstream = ControlledStream([sse(event) for event in events], wait_after=hanging)
        clients, sent, logs = [], [], []
        deps = deps_for(logs)
        payload = request or {'model': 'claude-offline', 'messages': [{'role': 'user', 'content': 'Synthetic fixture'}], 'stream': True}
        config = settings or policy()
        def handler(req):
            sent.append(req)
            return httpx.Response(200, headers={'content-type': 'text/event-stream', 'x-account-pool-id': 'himodels_1'}, stream=upstream)
        def factory(*args, **kwargs):
            client = REAL_CLIENT(transport=httpx.MockTransport(handler))
            clients.append(client)
            return client
        common = dict(url='http://fixture.invalid/v1/messages', request_data=payload,
                      headers={}, timeout=httpx.Timeout(10), deps=deps, trace_id='initial-fixture')
        async def run():
            with policy_scope(config, payload):
                if mode == 'raw_stream':
                    return b''.join([part async for part in forward_anthropic_messages_stream(**common)])
                common.update(model='claude-offline', max_raw_sse_bytes=100000)
                if mode == 'raw_collect':
                    return await collect_anthropic_messages_response(**common)
                common['messages'] = payload['messages']
                if mode == 'chat_collect':
                    return await collect_anthropic_messages_as_chat_completion(**common)
                return b''.join([part async for part in forward_anthropic_messages_as_chat_stream(
                    **common, caller_key='', caller_desc='')])
        with patch.object(httpx, 'AsyncClient', side_effect=factory):
            task = asyncio.create_task(run())
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5 if hanging else 2)
                # These assertions happen before cancellation/fixture cleanup.
                self.assertIn(task, done, 'adapter waited on upstream after a stop match')
                result = await task
                self.assertTrue(upstream.closed, 'response still open after adapter returned')
                self.assertTrue(all(c.is_closed for c in clients), 'client still open')
                self.assertEqual(len(sent), 1, 'early-stop must never retry')
                if hanging:
                    self.assertTrue(any('early_stop_upstream_close ' in line and
                                        'response_closed=true client_closed=true' in line for line in logs), logs)
                return result, logs
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def test_all_adapters_close_for_initial_and_split_stop(self):
        cases = {
            'first_delta': [message(), start(), delta(TAG + 'hidden')],
            'split_deltas': [message(), start(), delta('<C'), delta('UT>hidden')],
            'block_start': [message(), start(TAG + 'hidden')],
            'split_start_delta': [message(), start('<C'), delta('UT>hidden')],
            'message_content': [message([text_block(TAG + 'hidden')])],
            'split_message_block': [message([text_block('<C')]), start('UT>hidden')],
            'later_block': [message(), start(''), delta(''), start(TAG + 'hidden', 7)],
        }
        for mode in MODES:
            for case, events in cases.items():
                with self.subTest(mode=mode, case=case):
                    result, _ = await self.run_adapter(mode, events, hanging=True)
                    if mode.endswith('stream'):
                        self.assertNotIn(b'hidden', result)
                        self.assertNotIn(b'[DONE]', result)
                        self.assertNotIn(b'"type": "message_stop"', result)
                    elif mode == 'chat_collect':
                        self.assertEqual(result[0], '')
                        self.assertEqual(result[3], 'stop')
                    else:
                        self.assertEqual(result[0]['content'], [text_block('')])
                        self.assertEqual(result[0]['stop_reason'], 'stop_sequence')

    async def test_visible_initial_content_is_not_dropped_or_double_appended(self):
        events = [message([text_block('A'), text_block('B')]), start('C'), delta('D'),
                  delta('D'), {'type': 'content_block_stop', 'index': 0},
                  start('E', 4), delta('F', 4), *finish()]
        for mode in MODES:
            with self.subTest(mode=mode):
                result, logs = await self.run_adapter(mode, events)
                self.assertFalse(any('early_stop_upstream_close ' in line for line in logs))
                if mode == 'chat_stream':
                    self.assertEqual(visible(result), 'ABCDDEF')
                    self.assertIn(b'[DONE]', result)
                elif mode == 'chat_collect':
                    self.assertEqual(result[0], 'ABCDDEF')
                    self.assertEqual(result[1], 'claude-offline')
                    self.assertEqual(result[2]['total_tokens'], 7)
                elif mode == 'raw_collect':
                    self.assertEqual(result[0]['content'], [text_block('ABCDDEF')])
                else:
                    decoded = [json.loads(line[6:]) for line in result.splitlines() if line.startswith(b'data: ')]
                    self.assertEqual(decoded, events)

    async def test_literal_repeated_delta_after_initial_text_is_not_deduplicated(self):
        # Delta means new text. Never infer overlap merely from equal strings;
        # that would silently discard legitimate repetition in model output.
        for first in (message([text_block('OK')]), start('OK')):
            for mode in ('chat_stream', 'chat_collect', 'raw_collect'):
                with self.subTest(mode=mode, initial=first['type']):
                    result, _ = await self.run_adapter(mode, [first, delta('OK'), *finish()])
                    if mode == 'chat_stream':
                        self.assertEqual(visible(result), 'OKOK')
                    elif mode == 'chat_collect':
                        self.assertEqual(result[0], 'OKOK')
                    else:
                        self.assertEqual(result[0]['content'], [text_block('OKOK')])

    async def test_start_event_keeps_only_the_prefix_before_stop(self):
        for initial in (message([text_block('safe' + TAG + 'hidden')]), start('safe' + TAG + 'hidden')):
            for mode in ('chat_stream', 'chat_collect', 'raw_collect'):
                with self.subTest(mode=mode, initial=initial['type']):
                    result, _ = await self.run_adapter(mode, [initial], hanging=True)
                    if mode == 'chat_stream':
                        self.assertEqual(visible(result), 'safe')
                    elif mode == 'chat_collect':
                        self.assertEqual(result[0], 'safe')
                    else:
                        self.assertEqual(result[0]['content'], [text_block('safe')])

    async def test_disabled_policy_and_account_override_are_respected(self):
        for enabled in (False, True):
            settings = policy(early_stop={'enabled': not enabled})
            settings['_local_policy'] = True
            settings['account_overrides'] = {'himodels_1': {'early_stop': {'enabled': enabled}}}
            for mode in MODES:
                with self.subTest(mode=mode, enabled=enabled):
                    events = [message(), start(TAG + 'visible-if-disabled')]
                    if not enabled:
                        events += finish()
                    result, logs = await self.run_adapter(mode, events, hanging=enabled, settings=settings)
                    self.assertEqual(any('early_stop_upstream_close ' in line for line in logs), enabled)
                    if not enabled and mode == 'chat_stream':
                        self.assertEqual(visible(result), TAG + 'visible-if-disabled')

    async def test_initial_content_guard_is_unchanged(self):
        request = {'model': 'claude-offline', 'messages': [{'role': 'user', 'content': 'Return <content>body</content>'}], 'stream': True}
        settings = policy(early_stop={'tags': ['<disclaimer>'], 'content_guard': True})
        for mode in MODES:
            with self.subTest(mode=mode):
                events = [message(), start('<disclaimer>preface'), delta('<content>body</content>'), *finish()]
                _, logs = await self.run_adapter(mode, events, request=request, settings=settings)
                self.assertFalse(any('early_stop_upstream_close ' in line for line in logs))
                self.assertTrue(any('early_stop_disclaimer_ignored' in line for line in logs))

    async def test_tools_and_initial_thinking_are_not_visible_text(self):
        events = [message([{'type': 'thinking', 'thinking': TAG}]),
                  {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': TAG}},
                  {'type': 'content_block_start', 'index': 2, 'content_block': {'type': 'tool_use', 'id': 'tool-1', 'name': 'lookup', 'input': {'query': TAG}}},
                  start('OK', 3), *finish()]
        for mode in MODES:
            with self.subTest(mode=mode):
                result, logs = await self.run_adapter(mode, events)
                self.assertFalse(any('early_stop_upstream_close ' in line for line in logs))
                if mode == 'chat_stream':
                    self.assertEqual(visible(result), 'OK')
                    self.assertIn(b'lookup', result)
                if mode == 'chat_collect':
                    self.assertEqual(result[0], 'OK')
                    self.assertEqual(result[5][0]['function']['name'], 'lookup')

    async def test_gateway_initial_stop_closes_and_releases_owner(self):
        harness = gateway.ChannelGatewayTests('runTest')
        upstream = ControlledStream([sse(message()), sse(start(TAG + 'hidden'))], wait_after=True)
        response, body, sent, deps = await asyncio.wait_for(harness.invoke(
            plan(model='free/claude-opus-5-5', transport='messages'),
            {'model': 'free/claude-opus-5-5', 'messages': [{'role': 'user', 'content': 'Synthetic'}], 'stream': True},
            lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=upstream),
        ), 1)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(upstream.closed)
        self.assertEqual(len(sent), 1)
        self.assertNotIn(b'hidden', body)
        self.assertNotIn(b'[DONE]', body)
        self.assertEqual(len(deps.active_stream_registry), 0)

    async def test_replay_paths_read_the_same_initial_text(self):
        for stopped in (False, True):
            logs = []
            events = [message([text_block('A')]), start('B'), delta(TAG + 'hidden' if stopped else 'C'), *finish()]
            common = dict(raw_sse_text=b''.join(sse(e) for e in events).decode(),
                          request_data={}, model='claude-offline', messages=[], trace_id='replay-fixture', deps=deps_for(logs))
            result = await collect_anthropic_chat_completion_from_raw_sse(**common)
            body = b''.join([part async for part in replay_anthropic_chat_stream(**common, caller_key='', caller_desc='')])
            self.assertEqual(result[0], 'AB' if stopped else 'ABC')
            self.assertEqual(visible(body), 'AB' if stopped else 'ABC')


if __name__ == '__main__':
    unittest.main()
