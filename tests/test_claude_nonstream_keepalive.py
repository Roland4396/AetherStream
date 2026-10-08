"""Registered JSON endpoints: downstream heartbeats and exact-request reuse.

All upstream traffic uses MockTransport with an offline-only hostname.
"""
import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from aetherstream.api import app as module
from aetherstream.api.chat_routes import register_routes
from aetherstream.api.messages_routes import register_routes as register_messages
from aetherstream.api.responses_routes import register_routes as register_responses
from aetherstream.runtime.shared import SharedRuntimeState
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.upstreams.anthropic_messages.types import AnthropicMessagesDeps


REAL_CLIENT = httpx.AsyncClient
MODEL = 'claude-opus-5-5'


class ClaudeNonstreamKeepaliveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.shared = SharedRuntimeState(self.temp.name)
        self.logs, self.requests = [], []
        self.server = FastAPI()
        self.deps = register_routes(self.server, dict(vars(module), NONSTREAM_KEEPALIVE_INTERVAL=0.01))
        d = self.deps
        d.log = self.logs.append
        d.save_request_log = lambda *a, **k: None
        d.replay_service = SimpleNamespace(prepare=lambda **k: None)
        d._runtime_lookup = lambda *keys: False if keys == ('claude', 'stream', 'nonstream_to_stream') else None
        d.resolve_openai_compatible_route = AsyncMock(return_value=None)
        d.resolve_model_name_passthrough_upstream = lambda _: None
        d.get_claude_upstream_for_provider = lambda _: ('offline', 'http://offline.invalid/v1')
        d.get_claude_prompt_caching_settings = lambda: {'enabled': False}
        d.get_claude_cache_keepalive_settings = lambda: {'enabled': False}
        d.apply_claude_client_compat_request = lambda p: (p, {})
        d.apply_claude_output_settings = lambda p: p
        d.build_timed_claude_user_id = self.session
        d.build_anthropic_messages_deps = lambda payload: AnthropicMessagesDeps(
            log=self.logs.append, save_request_log=lambda *a, **k: None,
            build_openai_sse_error=lambda *a: b'', has_stop_tag=lambda _: False,
            find_stop_tag=lambda _: -1, fmt_ms=lambda *a: '0ms',
            release_caller=lambda *a: None, shared_runtime_state=self.shared,
            retire_session=self.shared.retire_session,
        )
        self.coalescer = ExactRequestCoalescer(ttl=180, log=self.logs.append, shared_state=self.shared)
        d.run_exact_nonstream_once = self.coalescer.run
        d.build_exact_request_key = self.coalescer.build_key
        ctx = dict(vars(module), NONSTREAM_KEEPALIVE_INTERVAL=0.01)
        ctx.update({name: getattr(d, name) for name in d.names})
        register_messages(self.server, ctx)
        register_responses(self.server, ctx)
        self.payload = {'model': MODEL, 'stream': False, 'max_tokens': 99,
                        'messages': [{'role': 'user', 'content': 'offline synthetic request'}]}

    def session(self, *, model=None, provider=None):
        key = f'{provider or "default"}:{model}'
        sid, mode = self.shared.session(key, 1800)
        user = json.dumps({'account_uuid': 'offline-account', 'device_id': 'offline-device', 'session_id': sid})
        return user, sid, mode, 1800, key

    @staticmethod
    def result(reason='end_turn'):
        return {'id': 'msg_offline', 'type': 'message', 'role': 'assistant', 'model': MODEL,
                'content': [{'type': 'text', 'text': 'offline ok'}], 'stop_reason': reason,
                'usage': {'input_tokens': 1, 'output_tokens': 1}}

    def transport(self, handler):
        async def dispatch(request):
            self.assertEqual(request.url.host, 'offline.invalid')
            self.requests.append(json.loads(request.content))
            return await handler(request)
        return patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(
            transport=httpx.MockTransport(dispatch), **kw))

    async def request(self, payload=None, *, disconnect_on_heartbeat=False, path='/v1/chat/completions'):
        body = json.dumps(payload or self.payload).encode()
        received = False
        disconnect = asyncio.Event()
        sent = []
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
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                 'method': 'POST', 'scheme': 'http', 'path': path,
                 'raw_path': path.encode(), 'root_path': '', 'query_string': b'',
                 'headers': [(b'content-type', b'application/json')],
                 'client': ('127.0.0.1', 1234), 'server': ('offline.invalid', 80)}
        await self.server(scope, receive, send)
        bodies = [m for m in sent if m['type'] == 'http.response.body']
        heartbeats = [m for m in bodies if m.get('more_body') and m.get('body', b'').isspace()]
        final = b''.join(m.get('body', b'') for m in bodies)
        return heartbeats, json.loads(final) if final.strip() else None

    async def test_identical_concurrent_json_requests_share_one_upstream_and_emit_heartbeats(self):
        async def handler(_):
            await asyncio.sleep(0.08)
            return httpx.Response(200, json=self.result())
        with self.transport(handler):
            a, b = await asyncio.gather(self.request(), self.request())
        self.assertEqual(len(self.requests), 1)
        self.assertIs(self.requests[0]['stream'], False)
        for heartbeats, result in (a, b):
            self.assertGreaterEqual(len(heartbeats), 1)
            self.assertEqual(result['choices'][0]['message']['content'], 'offline ok')

    async def test_reconnect_joins_detached_nonstream_work_without_reposting(self):
        release = asyncio.Event()
        async def handler(_):
            await release.wait()
            return httpx.Response(200, json=self.result())
        with self.transport(handler):
            beats, body = await self.request(disconnect_on_heartbeat=True)
            self.assertTrue(beats)
            self.assertIsNone(body)
            self.assertEqual((await self.coalescer.state())['inflight'], 1)
            reconnect = asyncio.create_task(self.request())
            await asyncio.sleep(0.03)
            release.set()
            _, result = await reconnect
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(result['choices'][0]['message']['content'], 'offline ok')
        self.assertEqual((await self.coalescer.state())['inflight'], 0)

    async def test_changed_request_is_not_merged(self):
        async def handler(_):
            await asyncio.sleep(0.05)
            return httpx.Response(200, json=self.result())
        changed = dict(self.payload, max_tokens=100)
        with self.transport(handler):
            await asyncio.gather(self.request(), self.request(changed))
        self.assertEqual(len(self.requests), 2)

    async def test_messages_and_responses_json_endpoints_share_same_keepalive_and_dedupe(self):
        async def handler(_):
            await asyncio.sleep(0.05)
            return httpx.Response(200, json=self.result())
        with self.transport(handler):
            for index, path in enumerate(('/v1/messages', '/v1/responses')):
                payload = dict(self.payload, max_tokens=101 + index)
                if path.endswith('/responses'):
                    payload['input'] = payload.pop('messages')
                    payload['max_output_tokens'] = payload.pop('max_tokens')
                before = len(self.requests)
                results = await asyncio.gather(self.request(payload, path=path), self.request(payload, path=path))
                self.assertEqual(len(self.requests), before + 1)
                for beats, result in results:
                    self.assertTrue(beats)
                    self.assertFalse(result.get('error'))
                    self.assertIn('offline ok', json.dumps(result))

    async def test_refusal_rotates_session_and_next_turn_does_not_reuse_cached_refusal(self):
        async def handler(_):
            return httpx.Response(200, json=self.result('refusal' if len(self.requests) == 1 else 'end_turn'))
        with self.transport(handler):
            await self.request()
            await self.request()
        self.assertEqual(len(self.requests), 2)
        identities = [json.loads(p['metadata']['user_id']) for p in self.requests]
        self.assertNotEqual(identities[0]['session_id'], identities[1]['session_id'])
        self.assertEqual(identities[0]['device_id'], identities[1]['device_id'])
        self.assertEqual(identities[0]['account_uuid'], identities[1]['account_uuid'])

    async def test_upstream_failure_does_not_become_a_cached_success(self):
        async def handler(_):
            return httpx.Response(500, text='offline server failure')
        with self.transport(handler):
            for _ in range(2):
                _, result = await self.request()
                self.assertIn('error', result)
        self.assertEqual(len(self.requests), 2)


if __name__ == '__main__':
    unittest.main()
