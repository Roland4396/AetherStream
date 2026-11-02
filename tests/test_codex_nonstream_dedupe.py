"""Codex Responses transport: exact JSON sharing with no real inference.

Exercise the registered Chat endpoint, real Responses SSE adapter, keepalive,
policy resolution and process-shared coalescer against synthetic transport only.
"""
import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from aetherstream.api import app as module
from aetherstream.api.chat_routes import register_routes
from aetherstream.routing.policy_config import dedupe_identity, enrich_selection
from aetherstream.runtime.flags import RuntimeFlags
from aetherstream.runtime.shared import SharedRuntimeState
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.upstreams.openai_responses import ResponsesUpstreamDeps


REAL_CLIENT = httpx.AsyncClient
MODEL = 'gpt-6-astra'


def flags(enabled=True):
    return {
        'streaming': {'exact_dedupe': False, 'json_keepalive': True},
        'channel_policies': {
            'schema': 1,
            'profiles': {'subscription-responses': {
                'early_stop': {'enabled': False},
            }},
            'channels': {'subscription': {'policy': {'streaming': {'force_upstream': 'stream'}}, 'routes': {
                'subscription-responses': {
                    'profiles': ['subscription-responses'],
                    'policy': {'streaming': {'exact_dedupe': enabled}},
                },
                'unrelated': {'policy': {}},
            }}},
        },
    }


def selected(model=MODEL, route='subscription-responses'):
    return {'model': model, 'upstream_model': model, 'channel': 'subscription',
            'route': route, 'transport': 'responses', 'revision': 'offline-only',
            'base_url': 'http://offline.invalid', 'api_key': 'synthetic-not-a-secret'}


def sse(data):
    return ('data: ' + json.dumps(data) + '\n\n').encode()


class SyntheticResponses(httpx.AsyncByteStream):
    def __init__(self, text='offline result', *, complete=True, delay=.08):
        self.text, self.complete, self.delay = text, complete, delay
        self.closed = False

    async def __aiter__(self):
        yield sse({'type': 'response.created', 'response': {'id': 'response-offline'}})
        await asyncio.sleep(self.delay)
        if self.text:
            yield sse({'type': 'response.output_text.delta', 'delta': self.text})
        if self.complete:
            yield sse({'type': 'response.completed', 'response': {
                'status': 'completed', 'output': [],
                'usage': {'input_tokens': 4, 'output_tokens': 2},
            }})

    async def aclose(self):
        self.closed = True


class CodexNonstreamDedupeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.shared = SharedRuntimeState(self.temp.name)
        self.logs, self.requests, self.streams = [], [], []
        self.config = flags()
        self.coalescer = ExactRequestCoalescer(ttl=180, log=self.logs.append,
                                              shared_state=self.shared)
        self.server = FastAPI()
        self.deps = register_routes(self.server, dict(vars(module), NONSTREAM_KEEPALIVE_INTERVAL=.01))
        d = self.deps
        d.account_routing_enabled = lambda: True
        d.resolve_account_route = AsyncMock(side_effect=self.resolve)
        d.build_account_dedupe = AsyncMock(side_effect=dedupe_identity)
        d.build_exact_request_key = self.coalescer.build_key
        d.run_exact_nonstream_once = self.coalescer.run
        d.replay_service = SimpleNamespace(prepare=lambda **_: None)
        d.log = self.logs.append
        d.save_request_log = lambda *_, **__: None
        d.build_responses_upstream_deps = lambda: ResponsesUpstreamDeps(
            log=self.logs.append, save_request_log=lambda *_, **__: None,
            build_openai_sse_error=module.build_openai_sse_error,
            has_stop_tag=lambda _: False, find_stop_tag=lambda _: -1,
            fmt_ms=module.fmt_ms, release_caller=lambda *_, **__: None,
        )
        self.payload = {'model': MODEL, 'stream': False, 'max_tokens': 100,
                        'messages': [{'role': 'user', 'content': 'Synthetic fixture only'}]}

    def resolve(self, model, **_):
        return enrich_selection(lambda k: self.config.get(k), selected(model))

    def transport(self, factory=None):
        def dispatch(request):
            self.assertEqual(request.url.host, 'offline.invalid')
            self.assertEqual(request.url.path, '/v1/responses')
            self.requests.append(json.loads(request.content))
            stream = factory(len(self.requests)) if factory else SyntheticResponses()
            self.streams.append(stream)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
        return patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
            transport=httpx.MockTransport(dispatch), **kw))

    async def request(self, payload=None, *, disconnect_on_heartbeat=False):
        body = json.dumps(payload or self.payload).encode()
        received, sent = False, []
        disconnect = asyncio.Event()

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            await disconnect.wait()
            return {'type': 'http.disconnect'}

        async def send(message):
            sent.append(message)
            if disconnect_on_heartbeat and message.get('more_body'):
                disconnect.set()

        path = '/v1/chat/completions'
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                 'method': 'POST', 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                 'root_path': '', 'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                 'client': ('127.0.0.1', 1234), 'server': ('offline.invalid', 80)}
        await self.server(scope, receive, send)
        bodies = [m for m in sent if m['type'] == 'http.response.body']
        beats = [m for m in bodies if m.get('more_body') and m.get('body', b'').isspace()]
        raw = b''.join(m.get('body', b'') for m in bodies)
        return beats, json.loads(raw) if raw.strip() else None

    async def test_four_identical_requests_share_one_responses_call_with_heartbeats(self):
        before = copy.deepcopy(self.payload)
        with self.transport():
            results = await asyncio.gather(*(self.request() for _ in range(4)))
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(self.requests[0]['stream'])
        self.assertEqual(self.payload, before)
        for beats, result in results:
            self.assertTrue(beats)
            self.assertEqual(result['choices'][0]['message']['content'], 'offline result')
            self.assertEqual(result['choices'][0]['finish_reason'], 'stop')
        self.assertTrue(all(result == results[0][1] for _, result in results))
        self.assertEqual(sum('nonstream_dedupe_wait ' in s for s in self.logs), 3)
        self.assertTrue(all(s.closed for s in self.streams))

    async def test_different_messages_or_parameters_are_never_merged(self):
        changed_message = copy.deepcopy(self.payload)
        changed_message['messages'][0]['content'] += ' different'
        with self.transport():
            await asyncio.gather(self.request(), self.request(changed_message),
                                 self.request(dict(self.payload, max_tokens=101)))
        self.assertEqual(len(self.requests), 3)

    async def test_success_reconnect_reuses_completed_result_without_new_inference(self):
        with self.transport():
            first = await self.request()
            second = await self.request()
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(first[1], second[1])
        self.assertTrue(any('nonstream_dedupe_recent_hit' in s for s in self.logs))

    async def test_empty_response_is_shared_but_next_manual_request_runs_fresh(self):
        with self.transport(lambda n: SyntheticResponses('' if n == 1 else 'fresh result')):
            a, b = await asyncio.gather(self.request(), self.request())
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(a[1]['choices'][0]['message']['content'], '')
            self.assertEqual(a[1], b[1])
            _, fresh = await self.request()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(fresh['choices'][0]['message']['content'], 'fresh result')

    async def test_incomplete_stream_is_shared_error_without_auto_retry_and_next_request_runs(self):
        with self.transport(lambda n: SyntheticResponses(complete=n > 1)):
            a, b = await asyncio.gather(self.request(), self.request())
            self.assertEqual(len(self.requests), 1)
            self.assertIn('error', a[1])
            self.assertEqual(a[1], b[1])
            _, fresh = await self.request()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(fresh['choices'][0]['message']['content'], 'offline result')

    async def test_one_disconnected_waiter_does_not_cancel_other_waiter(self):
        with self.transport():
            gone, remaining = await asyncio.gather(
                self.request(disconnect_on_heartbeat=True), self.request())
        self.assertEqual(len(self.requests), 1)
        self.assertIsNone(gone[1])
        self.assertEqual(remaining[1]['choices'][0]['message']['content'], 'offline result')
        self.assertEqual((await self.coalescer.state())['inflight'], 0)

    async def test_disabled_policy_reproduces_independent_upstream_requests(self):
        self.config = flags(False)
        with self.transport():
            await asyncio.gather(self.request(), self.request())
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(any('nonstream_dedupe_leader' in s for s in self.logs))


class CodexPolicyHotReloadTests(unittest.TestCase):
    def test_route_only_change_hot_reloads_without_mutating_existing_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'flags.json'
            path.write_text(json.dumps(flags(False)))
            store = RuntimeFlags(path=str(path), poll_sec=0, log=lambda _: None)
            old = enrich_selection(store.lookup, selected())
            inode = path.stat().st_ino
            old_mtime_ns = path.stat().st_mtime_ns
            # Single-file Docker bind mounts need the existing inode retained.
            with path.open('r+') as file:
                file.write(json.dumps(flags(True)))
                file.truncate()
            # Avoid same-tick mtime on coarse container filesystems.
            os.utime(path, ns=(old_mtime_ns + 1_000_000_000, old_mtime_ns + 1_000_000_000))
            self.assertEqual(path.stat().st_ino, inode)
            new = enrich_selection(store.lookup, selected())
            other = enrich_selection(store.lookup, selected(route='unrelated'))
            self.assertFalse(old['policy']['streaming']['exact_dedupe'])
            self.assertTrue(new['policy']['streaming']['exact_dedupe'])
            self.assertFalse(other['policy']['streaming']['exact_dedupe'])
            self.assertTrue(new['policy']['streaming']['json_keepalive'])
            self.assertEqual(new['policy']['streaming']['force_upstream'], 'stream')


if __name__ == '__main__':
    unittest.main()
