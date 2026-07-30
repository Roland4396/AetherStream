import asyncio
import json
import unittest

from starlette.responses import JSONResponse

from aetherstream.streaming.json_keepalive import JSONKeepaliveResponse


class JSONKeepaliveResponseTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _scope():
        return {
            'type': 'http',
            'asgi': {'version': '3.0', 'spec_version': '2.4'},
            'http_version': '1.1',
            'method': 'POST',
            'scheme': 'http',
            'path': '/v1/chat/completions',
            'raw_path': b'/v1/chat/completions',
            'query_string': b'',
            'headers': [],
            'client': ('test', 1),
            'server': ('test', 80),
            'root_path': '',
        }

    async def test_emits_json_safe_heartbeats_then_valid_json(self):
        release = asyncio.Event()
        messages = []

        async def operation():
            await release.wait()
            return JSONResponse({'answer': '完成'})

        async def receive():
            await asyncio.Future()

        async def send(message):
            messages.append(message)
            heartbeat_count = sum(
                item.get('type') == 'http.response.body' and item.get('more_body', False)
                for item in messages
            )
            if heartbeat_count >= 2:
                release.set()

        response = JSONKeepaliveResponse(operation, interval=0.01)
        await response(self._scope(), receive, send)

        self.assertEqual(messages[0]['type'], 'http.response.start')
        self.assertEqual(messages[0]['status'], 200)
        heartbeat_bodies = [
            item['body']
            for item in messages
            if item.get('type') == 'http.response.body' and item.get('more_body', False)
        ]
        self.assertGreaterEqual(len(heartbeat_bodies), 2)
        self.assertTrue(all(body.isspace() for body in heartbeat_bodies))
        complete_body = b''.join(
            item.get('body', b'')
            for item in messages
            if item.get('type') == 'http.response.body'
        )
        self.assertEqual(json.loads(complete_body), {'answer': '完成'})

    async def test_fast_response_preserves_error_status_and_has_no_heartbeat(self):
        messages = []

        async def operation():
            return JSONResponse({'error': 'bad request'}, status_code=422)

        async def receive():
            await asyncio.Future()

        async def send(message):
            messages.append(message)

        response = JSONKeepaliveResponse(operation, interval=1)
        await response(self._scope(), receive, send)

        self.assertEqual(messages[0]['status'], 422)
        body_messages = [item for item in messages if item.get('type') == 'http.response.body']
        self.assertEqual(len(body_messages), 1)
        self.assertFalse(body_messages[0].get('more_body', False))
        self.assertEqual(json.loads(body_messages[0]['body']), {'error': 'bad request'})

    async def test_disconnect_cancels_the_waiting_operation(self):
        receive_queue = asyncio.Queue()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        messages = []

        async def operation():
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def receive():
            return await receive_queue.get()

        async def send(message):
            messages.append(message)

        response = JSONKeepaliveResponse(operation, interval=1)
        response_task = asyncio.create_task(response(self._scope(), receive, send))
        await started.wait()
        await receive_queue.put({'type': 'http.disconnect'})
        await response_task

        self.assertTrue(cancelled.is_set())
        self.assertEqual(messages, [])


if __name__ == '__main__':
    unittest.main()
