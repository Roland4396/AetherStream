"""Offline regression for one-way Claude refusal session retirement."""
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx

from aetherstream.runtime.shared import SharedRuntimeState
from aetherstream.upstreams.anthropic_messages.chat_stream import forward_anthropic_messages_as_chat_stream
from aetherstream.upstreams.anthropic_messages.collectors import collect_anthropic_messages_as_chat_completion
from aetherstream.upstreams.anthropic_messages.types import AnthropicMessagesDeps


REAL_CLIENT = httpx.AsyncClient
MODEL = 'claude-offline'
KEY = 'default:' + MODEL


def native_message(reason):
    return {
        'id': 'msg_offline', 'type': 'message', 'role': 'assistant', 'model': MODEL,
        'content': [{'type': 'text', 'text': 'offline'}], 'stop_reason': reason,
        'usage': {'input_tokens': 1, 'output_tokens': 1},
    }


def sse(reason, *, complete=True):
    events = [
        {'type': 'message_start', 'message': native_message(reason)},
        {'type': 'content_block_delta', 'index': 0,
         'delta': {'type': 'text_delta', 'text': 'offline'}},
        {'type': 'message_delta', 'delta': {'stop_reason': reason}},
    ]
    if complete:
        events.append({'type': 'message_stop'})
    return ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)


class ClaudeRefusalSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = SharedRuntimeState(self.tmp.name)
        self.logs = []
        self.calls = 0

    def deps(self, sid):
        return AnthropicMessagesDeps(
            log=self.logs.append, save_request_log=lambda *a, **k: None,
            build_openai_sse_error=lambda *a: b'error',
            has_stop_tag=lambda _: False, find_stop_tag=lambda _: -1,
            fmt_ms=lambda *a: '0ms', release_caller=lambda *a: None,
            shared_runtime_state=self.state, session_key=KEY, session_id=sid,
        )

    async def request(self, *, streaming_downstream, upstream_stream, reason,
                      complete=True, status=200):
        sid, mode = self.state.session(KEY, 1800)
        async def handler(request):
            self.calls += 1
            if status != 200:
                return httpx.Response(status, text='offline error')
            if upstream_stream:
                return httpx.Response(200, text=sse(reason, complete=complete))
            return httpx.Response(200, json=native_message(reason))

        with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
                transport=httpx.MockTransport(handler), **kw)):
            kwargs = dict(
                url='http://offline.invalid/v1/messages',
                request_data={'model': MODEL, 'messages': [], 'stream': upstream_stream},
                headers={}, model=MODEL, messages=[], trace_id='offline',
                timeout=httpx.Timeout(2), max_raw_sse_bytes=100000,
                deps=self.deps(sid),
            )
            if streaming_downstream:
                output = b''.join([part async for part in
                    forward_anthropic_messages_as_chat_stream(
                        **kwargs, caller_key='', caller_desc='', cache_keepalive=None)])
            else:
                output = await collect_anthropic_messages_as_chat_completion(
                    **kwargs, upstream_stream=upstream_stream)
        return sid, mode, output

    async def test_refusal_rotates_on_next_request_without_replay(self):
        for downstream, upstream in ((True, True), (False, True), (False, False)):
            with self.subTest(downstream=downstream, upstream=upstream):
                self.state.retire_session(KEY, self.state.session(KEY, 1800)[0])
                before = self.calls
                old, _, _ = await self.request(
                    streaming_downstream=downstream, upstream_stream=upstream,
                    reason='refusal')
                self.assertEqual(self.calls, before + 1)
                new, mode, _ = await self.request(
                    streaming_downstream=downstream, upstream_stream=upstream,
                    reason='end_turn')
                self.assertEqual(self.calls, before + 2)
                self.assertNotEqual(new, old)
                self.assertEqual(mode, 'new')
                self.assertEqual(self.state.session(KEY, 1800), (new, 'reused'))

    async def test_normal_or_incomplete_or_http_error_does_not_retire(self):
        for case in (
            dict(streaming_downstream=True, upstream_stream=True, reason='end_turn'),
            dict(streaming_downstream=False, upstream_stream=False, reason='end_turn'),
            dict(streaming_downstream=True, upstream_stream=True, reason='refusal', complete=False),
            dict(streaming_downstream=True, upstream_stream=True, reason='refusal', status=500),
        ):
            with self.subTest(case=case):
                sid, _, _ = await self.request(**case)
                self.assertEqual(self.state.session(KEY, 1800), (sid, 'reused'))

    def test_stale_refusal_and_other_session_are_untouched(self):
        old, _ = self.state.session(KEY, 1800)
        self.assertTrue(self.state.retire_session(KEY, old))
        current, _ = self.state.session(KEY, 1800)
        self.assertFalse(self.deps(old).retire_refused_session('refusal'))
        self.assertEqual(self.state.session(KEY, 1800), (current, 'reused'))
        self.assertFalse(self.deps(current).retire_refused_session('end_turn'))


if __name__ == '__main__':
    unittest.main()
