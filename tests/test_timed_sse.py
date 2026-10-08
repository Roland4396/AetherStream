import asyncio
import time
import unittest
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from aetherstream.api.timed_routes import TimedSSEResponse, register_routes
from aetherstream.streaming.responses import DisconnectSafeStreamingResponse
from aetherstream.streaming.state import ActiveStreamRegistry


def dependencies():
    logs = []
    return SimpleNamespace(log=logs.append, active_stream_registry=ActiveStreamRegistry(log=logs.append), logs=logs)


class TimedResponseTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, *, finish=False, disconnect=False, stall_send=False, silent=False):
        deps = dependencies()
        deps.active_stream_registry.register('test', trace_id='timed', model='mock', msg_count=1)
        closed, cancelled, first = asyncio.Event(), asyncio.Event(), asyncio.Event()
        queue = asyncio.Queue()
        sent = []

        async def body():
            try:
                if silent:
                    await asyncio.Event().wait()
                yield b'data: first\n\n'
                if finish:
                    yield b'data: [DONE]\n\n'
                else:
                    await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                closed.set()

        async def send(message):
            sent.append(message)
            if message['type'] == 'http.response.body' and message.get('more_body'):
                first.set()
                if stall_send:
                    await asyncio.Event().wait()

        response = TimedSSEResponse(
            DisconnectSafeStreamingResponse(body(), media_type='text/event-stream'),
            deadline=time.monotonic() + (5 if disconnect else .06), seconds=.06, trace_id='timed', deps=deps,
        )
        scope = {'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions/timed', 'headers': [], 'asgi': {'spec_version': '2.3'}}
        task = asyncio.create_task(response(scope, queue.get, send))
        if disconnect:
            await asyncio.wait_for(first.wait(), 1)
            await queue.put({'type': 'http.disconnect'})
        await asyncio.wait_for(task, 1)
        self.assertTrue(closed.is_set())
        self.assertEqual(cancelled.is_set(), not finish)
        self.assertEqual(len(deps.active_stream_registry), 0)
        payload = b''.join(m.get('body', b'') for m in sent)
        self.assertEqual(b'[DONE]' in payload, finish)
        if not disconnect:
            self.assertFalse(sent[-1].get('more_body', False))

    async def test_deadline_cancels_upstream_wait(self):
        await self.exercise()

    async def test_deadline_before_first_chunk(self):
        await self.exercise(silent=True)

    async def test_deadline_during_downstream_send(self):
        await self.exercise(stall_send=True)

    async def test_client_disconnect_cleans_up_before_deadline(self):
        await self.exercise(disconnect=True)

    async def test_normal_completion_is_not_truncated(self):
        await self.exercise(finish=True)


class TimedRouteTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, handler, payload, query='', raw=None):
        app = FastAPI()
        deps = dependencies()
        register_routes(app, handler=handler, deps=deps)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            kwargs = {'content': raw} if raw is not None else {'json': payload}
            response = await client.post('/v1/chat/completions/timed' + query, headers={'authorization': 'Bearer mock'}, **kwargs)
        return response, deps

    async def test_same_handler_receives_auth_and_unchanged_payload(self):
        payload = {'model': 'any-provider-model', 'stream': True, 'messages': [], 'max_tokens': 100}
        seen = []

        async def handler(request, deps):
            seen.append((await request.json(), request.headers['authorization']))
            async def body():
                yield b'data: [DONE]\n\n'
            return DisconnectSafeStreamingResponse(body(), media_type='text/event-stream')

        response, _ = await self.request(handler, payload)
        self.assertEqual(seen, [(payload, 'Bearer mock')])
        self.assertEqual(float(response.headers['x-sse-time-limit']), 290)
        self.assertEqual(response.content, b'data: [DONE]\n\n')

    async def test_bad_duration_and_nonstream_do_not_call_handler(self):
        async def handler(*args):
            self.fail('invalid input must not call upstream')
        for value in ('0', '-1', 'nan', 'inf', '601', 'abc'):
            r, _ = await self.request(handler, {'stream': True}, '?duration_seconds=' + value)
            self.assertEqual(r.status_code, 422)
        for payload in ({'stream': False}, {}, []):
            r, _ = await self.request(handler, payload)
            self.assertEqual(r.status_code, 400)
        r, _ = await self.request(handler, None, raw=b'not-json')
        self.assertEqual(r.status_code, 400)

    async def test_timer_includes_handler_wait_before_headers(self):
        closed = asyncio.Event()
        async def handler(request, deps):
            deps.active_stream_registry.register('test', trace_id=request.state.trace_id, model='mock', msg_count=1)
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        r, deps = await self.request(handler, {'stream': True}, '?duration_seconds=0.05')
        self.assertEqual(r.status_code, 504)
        self.assertTrue(closed.is_set())
        self.assertEqual(len(deps.active_stream_registry), 0)

    async def test_upstream_error_preserved(self):
        async def handler(request, deps):
            return JSONResponse({'error': 'upstream rejected'}, status_code=429)
        r, _ = await self.request(handler, {'stream': True})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json(), {'error': 'upstream rejected'})

    async def test_deadline_closes_real_upstream_http_connection(self):
        disconnected = asyncio.Event()

        async def upstream(reader, writer):
            try:
                await reader.readuntil(b'\r\n\r\n')
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: first\n\n')
                await writer.drain()
                await reader.read()
            finally:
                disconnected.set()
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(upstream, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        async def handler(request, deps):
            async def body():
                async with httpx.AsyncClient(trust_env=False) as client:
                    async with client.stream('GET', f'http://127.0.0.1:{port}/') as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
            return DisconnectSafeStreamingResponse(body(), media_type='text/event-stream')
        try:
            r, _ = await self.request(handler, {'stream': True}, '?duration_seconds=0.3')
            self.assertEqual(r.content, b'data: first\n\n')
            await asyncio.wait_for(disconnected.wait(), 1)
        finally:
            server.close()
            await server.wait_closed()


if __name__ == '__main__':
    unittest.main()
