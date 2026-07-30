import asyncio
import unittest

from aetherstream.streaming.responses import DisconnectSafeStreamingResponse
from starlette.requests import ClientDisconnect


class DisconnectSafeStreamingResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_during_send_closes_body_iterator(self):
        iterator_closed = asyncio.Event()
        iterator_cancelled = asyncio.Event()
        first_chunk_sent = asyncio.Event()
        receive_queue = asyncio.Queue()

        async def body_iterator():
            try:
                yield b"first"
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                iterator_cancelled.set()
                raise
            finally:
                iterator_closed.set()

        async def receive():
            return await receive_queue.get()

        async def send(message):
            if message["type"] == "http.response.body" and message.get("more_body"):
                first_chunk_sent.set()
                await asyncio.Event().wait()

        response = DisconnectSafeStreamingResponse(body_iterator())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "asgi": {"spec_version": "2.3"},
        }

        response_task = asyncio.create_task(response(scope, receive, send))
        await asyncio.wait_for(first_chunk_sent.wait(), timeout=1)
        await receive_queue.put({"type": "http.disconnect"})
        await asyncio.wait_for(response_task, timeout=1)

        self.assertTrue(iterator_closed.is_set())
        self.assertTrue(iterator_cancelled.is_set())

    async def test_normal_completion_closes_body_iterator(self):
        iterator_closed = asyncio.Event()
        iterator_cancelled = asyncio.Event()
        sent = []

        async def body_iterator():
            try:
                yield b"complete"
            except asyncio.CancelledError:
                iterator_cancelled.set()
                raise
            finally:
                iterator_closed.set()

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        response = DisconnectSafeStreamingResponse(body_iterator())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "asgi": {"spec_version": "2.3"},
        }

        await asyncio.wait_for(response(scope, receive, send), timeout=1)

        self.assertTrue(iterator_closed.is_set())
        self.assertFalse(iterator_cancelled.is_set())
        self.assertEqual(sent[-1], {"type": "http.response.body", "body": b"", "more_body": False})

    async def test_send_error_is_not_suppressed_for_iterator_without_aclose(self):
        class BodyIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                return b"chunk"

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("downstream closed")

        response = DisconnectSafeStreamingResponse(BodyIterator())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "asgi": {"spec_version": "2.4"},
        }

        with self.assertRaises(ClientDisconnect):
            await asyncio.wait_for(response(scope, receive, send), timeout=1)


if __name__ == "__main__":
    unittest.main()
