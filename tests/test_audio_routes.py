import asyncio
import json
import unittest
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aetherstream.api import audio_routes


class AudioRoutesTest(unittest.TestCase):
    def make_client(self, handler):
        logs = []

        def build_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        app = FastAPI()
        audio_routes.register_routes(
            app,
            {
                "TTS_UPSTREAM_URL": "http://tts.internal",
                "TTS_MAX_REQUEST_BYTES": 65536,
                "build_tts_http_client": build_client,
                "log": logs.append,
            },
        )
        return TestClient(app), logs

    def test_streams_audio_without_forwarding_browser_authorization(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["headers"] = dict(request.headers)
            captured["body"] = request.content
            return httpx.Response(
                200,
                headers={"content-type": "audio/mpeg", "x-tts-voice": "firefly"},
                stream=httpx.ByteStream(b"mp3-bytes"),
            )

        client, logs = self.make_client(handler)
        response = client.post(
            "/v1/audio/speech",
            headers={
                "Authorization": "Bearer channel-specific-key",
                "Content-Type": "application/json",
                "X-Client-Request-Id": "audio-test",
            },
            json={"model": "tts-1", "voice": "firefly", "input": "hello"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"mp3-bytes")
        self.assertEqual(response.headers["x-tts-voice"], "firefly")
        self.assertNotIn("authorization", captured["headers"])
        self.assertEqual(captured["headers"]["x-client-request-id"], "audio-test")
        self.assertEqual(json.loads(captured["body"])["voice"], "firefly")
        self.assertTrue(any("authorization=present" in line for line in logs))

    def test_preserves_upstream_validation_error(self):
        def handler(_request: httpx.Request):
            return httpx.Response(
                400,
                headers={"content-type": "application/json"},
                json={"error": {"code": "invalid_voice"}},
            )

        client, _ = self.make_client(handler)
        response = client.post(
            "/v1/audio/speech",
            headers={"Content-Type": "application/json"},
            json={"model": "tts-1", "voice": "missing", "input": "hello"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_voice")

    def test_rejects_non_json_and_large_bodies_before_upstream(self):
        def handler(_request: httpx.Request):
            self.fail("upstream should not be called")

        client, _ = self.make_client(handler)
        response = client.post("/v1/audio/speech", content=b"not json")
        self.assertEqual(response.status_code, 415)

    def test_cancels_upstream_while_waiting_for_headers_after_disconnect(self):
        upstream_cancelled = asyncio.Event()
        upstream_started = asyncio.Event()
        logs = []

        async def handler(_request: httpx.Request):
            upstream_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                upstream_cancelled.set()
                raise
            return httpx.Response(200, content=b"too late")

        async def receive():
            await upstream_started.wait()
            return {"type": "http.disconnect"}

        async def run():
            request = httpx.Request("POST", "http://unused")
            downstream = audio_routes.Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/audio/speech",
                    "headers": [],
                    "client": ("127.0.0.1", 1),
                    "server": ("test", 80),
                    "scheme": "http",
                    "query_string": b"",
                    "http_version": "1.1",
                },
                receive,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            deps = SimpleNamespace(
                TTS_UPSTREAM_URL="http://tts.internal",
                build_tts_http_client=lambda: client,
                log=logs.append,
            )
            response = await audio_routes._send_upstream(
                request=downstream,
                deps=deps,
                method="POST",
                path="/v1/audio/speech",
                body=request.content,
            )
            self.assertEqual(response.status_code, 499)
            self.assertTrue(upstream_cancelled.is_set())
            self.assertTrue(any("downstream_cancelled" in line for line in logs))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
