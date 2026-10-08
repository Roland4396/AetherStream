"""Offline diagnostics: upstream metadata only, no inference or behavior changes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from aetherstream.observability.logging import ProxyLogger
from aetherstream.observability.refusals import AnthropicRefusalDiagnostics
from aetherstream.upstreams.anthropic_messages.chat_stream import forward_anthropic_messages_as_chat_stream
from aetherstream.upstreams.anthropic_messages.collectors import (
    collect_anthropic_messages_as_chat_completion, collect_anthropic_messages_response,
)
from aetherstream.upstreams.anthropic_messages.messages import forward_anthropic_messages_stream
from aetherstream.upstreams.anthropic_messages.types import AnthropicMessagesDeps


REAL_CLIENT = httpx.AsyncClient
DETAILS = {'type': 'refusal', 'category': 'cyber', 'explanation': 'Upstream fixture explanation.'}


def message(details=DETAILS, reason='refusal'):
    return {'type': 'message', 'id': 'msg_fixture', 'model': 'claude-fixture', 'role': 'assistant',
            'content': [{'type': 'text', 'text': 'partial text'}], 'stop_reason': reason,
            'stop_details': copy.deepcopy(details), 'usage': {'input_tokens': 2, 'output_tokens': 3}}


def events(details=DETAILS, reason='refusal'):
    return [
        {'type': 'message_start', 'message': {**message(None, None), 'content': []}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'partial text'}},
        {'type': 'message_delta', 'delta': {'stop_reason': reason, 'stop_details': copy.deepcopy(details)}},
        {'type': 'message_stop'},
    ]


def sse(details=DETAILS, reason='refusal'):
    return ''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n'
                   for e in events(details, reason))


class RefusalDiagnosticsTests(unittest.TestCase):
    def test_preserves_known_and_future_categories_without_guessing(self):
        for category in ['cyber', 'bio', 'frontier_llm', 'reasoning_extraction', 'general_harms', 'future_category']:
            details = {**DETAILS, 'category': category}
            for raw in (sse(details), json.dumps(message(details))):
                with self.subTest(category=category, json=raw.startswith('{')):
                    d = AnthropicRefusalDiagnostics.from_raw(raw)
                    self.assertEqual(d.metadata()['category'], category)
                    self.assertEqual(d.metadata()['stop_details'], details)
                    self.assertEqual(d.metadata()['explanation'], DETAILS['explanation'])
                    self.assertIn('refusal_category="' + category + '"', d.log_suffix())

    def test_missing_details_explicitly_says_unspecified(self):
        for details in (None, {}, [], 'malformed', {'type': 'refusal'}):
            d = AnthropicRefusalDiagnostics.from_raw(sse(details))
            self.assertEqual(d.metadata()['category'], 'upstream_unspecified')
            self.assertEqual(d.metadata()['explanation'], '上游未提供具体原因')
        d = AnthropicRefusalDiagnostics.from_raw(sse({'type': 'refusal', 'category': 'cyber'}))
        self.assertEqual(d.metadata()['category'], 'cyber')
        self.assertEqual(d.metadata()['explanation'], '上游未提供具体原因')

    def test_no_diagnosis_from_user_text_reasoning_or_tools(self):
        d = AnthropicRefusalDiagnostics()
        for event in [None, [], 'bad', {'type': {}}, {'type': []}, {'type': 'content_block_delta', 'delta': {
            'type': 'text_delta', 'text': json.dumps(message()), 'stop_reason': 'refusal'}},
            {'type': 'content_block_start', 'content_block': {'type': 'tool_use', 'input': message()}},
            {'type': 'message', 'content': [{'type': 'text', 'text': 'refusal cyber'}], 'stop_reason': 'end_turn'},
        ]:
            d.observe(event)
        self.assertEqual(d.metadata(), {})
        self.assertEqual(d.log_suffix(), '')

    def test_null_followups_do_not_erase_details_and_input_is_not_mutated(self):
        stream = events()
        original = copy.deepcopy(stream)
        d = AnthropicRefusalDiagnostics()
        for event in stream:
            d.observe(event)
        d.observe({'type': 'message_delta', 'delta': {'stop_reason': 'refusal', 'stop_details': None}})
        self.assertEqual(d.metadata()['stop_details'], DETAILS)
        self.assertEqual(d.metadata()['source'], 'delta.stop_details')
        self.assertEqual(stream, original)
        d.metadata()['stop_details']['category'] = 'changed'
        self.assertEqual(d.metadata()['category'], 'cyber')

    def test_only_allowlisted_bounded_redacted_single_line_details(self):
        details = {**DETAILS, 'authorization': 'secret-header', 'token': 'secret-token',
                   'explanation': 'a\n\r\x1b[31m\u202e Bearer abcdef123456 api_key=abcdef123456 sk-secret123456',
                   'message': 'x' * 5000, 'code': {'secret': 'nested'}}
        d = AnthropicRefusalDiagnostics.from_raw(sse(details))
        info = d.metadata()
        self.assertNotIn('authorization', info['stop_details'])
        self.assertNotIn('token', info['stop_details'])
        self.assertNotIn('code', info['stop_details'])
        suffix = d.log_suffix()
        for secret in ['secret-header', 'secret-token', 'abcdef123456', 'sk-secret123456', '\n', '\r', '\x1b', '\u202e']:
            self.assertNotIn(secret, suffix)
        self.assertLess(len(info['stop_details']['message']), 4200)
        self.assertIn('[truncated]', info['stop_details']['message'])

    def test_tolerates_malformed_sse_and_message_start_metadata(self):
        raw = 'data: nope\ndata: []\ndata: [DONE]\n' + 'data:' + json.dumps(
            {'type': 'message_start', 'message': message()})
        d = AnthropicRefusalDiagnostics.from_raw(raw)
        self.assertEqual(d.metadata()['source'], 'message.stop_details')

    def test_output_and_json_logs_include_details_without_changing_raw_or_body(self):
        for raw in (sse(), json.dumps(message()), sse(None)):
            with tempfile.TemporaryDirectory() as tmp:
                logger = ProxyLogger(debug=False, log_dir=tmp)
                captured = []
                logger.log = captured.append
                logger.save_request_log(model='claude-fixture', messages=[], response='partial text',
                                        stream=True, raw_sse=raw, trace_id='fixture-id')
                log = json.loads((Path(tmp) / '01_input.json').read_text())
                output = (Path(tmp) / '01_output.txt').read_text()
                self.assertIn('upstream_refusal', log)
                self.assertIn('RefusalExplanation: ' + log['upstream_refusal']['explanation'], output)
                self.assertIn('TraceId: fixture-id', output)
                self.assertTrue(output.endswith('partial text'))
                self.assertTrue((Path(tmp) / '01_raw_sse.txt').read_text().endswith(raw))
                self.assertIn('refusal_category=', captured[-1])

    def test_success_does_not_inherit_previous_refusal_in_reused_log_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = ProxyLogger(debug=False, log_dir=tmp)
            for n in range(11):
                logger.save_request_log(model='fixture', messages=[], response='body', stream=True,
                                        raw_sse=sse() if n == 0 else sse(None, 'end_turn'))
            self.assertNotIn('upstream_refusal', json.loads((Path(tmp) / '01_input.json').read_text()))
            self.assertNotIn('RefusalCategory:', (Path(tmp) / '01_output.txt').read_text())


class RefusalTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_live_anthropic_paths_log_details_without_retries_or_body_changes(self):
        for mode in ('chat_stream', 'messages_stream', 'chat_collect_sse', 'chat_collect_json',
                     'messages_collect_sse', 'messages_collect_json'):
            for details in (DETAILS, None):
                with self.subTest(mode=mode, details=details):
                    logs, requests, saved = [], [], []
                    async def handler(request):
                        requests.append(json.loads(request.content))
                        return (httpx.Response(200, json=message(details)) if mode.endswith('json')
                                else httpx.Response(200, text=sse(details)))
                    deps = AnthropicMessagesDeps(
                        log=logs.append, save_request_log=lambda *a, **k: saved.append((a, k)),
                        build_openai_sse_error=lambda *a: b'error', has_stop_tag=lambda _: False,
                        find_stop_tag=lambda _: -1, fmt_ms=lambda *a: '0ms', release_caller=lambda *a: None,
                    )
                    request_data = {'model': 'claude-fixture', 'messages': [], 'stream': not mode.endswith('json')}
                    original = copy.deepcopy(request_data)
                    common = dict(url='http://offline.invalid/v1/messages', request_data=request_data,
                                  headers={}, deps=deps, timeout=httpx.Timeout(2), trace_id='fixture-trace')
                    collector = dict(model='claude-fixture', max_raw_sse_bytes=1)
                    # A tiny raw limit proves runtime diagnostics do not rely on a saved-body parser.
                    with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
                            transport=httpx.MockTransport(handler), **kw)):
                        if mode == 'chat_stream':
                            out = b''.join([part async for part in forward_anthropic_messages_as_chat_stream(
                                **common, **collector, messages=[], caller_key='', caller_desc='', cache_keepalive=None)])
                            self.assertIn(b'partial text', out)
                            self.assertIn(b'[DONE]', out)
                            self.assertNotIn(b'refusal_explanation', out)
                        elif mode == 'messages_stream':
                            out = b''.join([part async for part in forward_anthropic_messages_stream(**common)])
                            self.assertIn(b'partial text', out)
                            self.assertIn(b'"stop_reason": "refusal"', out)
                        elif mode.startswith('chat_collect'):
                            out = await collect_anthropic_messages_as_chat_completion(
                                **common, **collector, messages=[], upstream_stream=not mode.endswith('json'))
                            self.assertEqual(out[0], 'partial text')
                            self.assertEqual(out[3], 'stop')  # Existing downstream mapping is untouched.
                        else:
                            out, raw = await collect_anthropic_messages_response(
                                **common, **collector, upstream_stream=not mode.endswith('json'))
                            self.assertEqual(out['stop_reason'], 'refusal')
                            self.assertEqual(out['content'][0]['text'], 'partial text')
                            if mode.endswith('json'):
                                self.assertEqual(out, message(details))
                    self.assertEqual(requests, [original])
                    self.assertEqual(request_data, original)
                    done = [line for line in logs if '_done reason=' in line]
                    self.assertEqual(len(done), 1)
                    self.assertIn('[TRACE fixture-trace]', done[0])
                    self.assertIn('refusal_category="' + ('cyber' if details else 'upstream_unspecified') + '"', done[0])
                    self.assertIn(DETAILS['explanation'] if details else '上游未提供具体原因', done[0])
                    self.assertTrue(ProxyLogger(debug=False, log_dir=tempfile.gettempdir())._should_emit(done[0]))


if __name__ == '__main__':
    unittest.main()
