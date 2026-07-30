import asyncio
import json
import time
import unittest
from unittest.mock import patch

import httpx

from aetherstream.upstreams.anthropic_messages.chat_stream import (
    forward_anthropic_messages_as_chat_stream,
)
from aetherstream.upstreams.anthropic_messages.types import AnthropicMessagesDeps


class _FakeResponse:
    status_code = 200

    def __init__(self, pause_seconds: float, header_delay_seconds: float = 0.0):
        self.pause_seconds = pause_seconds
        self.header_delay_seconds = header_delay_seconds
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def aread(self):
        return b""

    async def aclose(self):
        self.closed = True

    async def aiter_lines(self):
        yield "event: message_start"
        yield 'data: {"type":"message_start","message":{"id":"msg_test","model":"claude-test"}}'
        yield ""
        await asyncio.sleep(self.pause_seconds)
        yield "event: message_delta"
        yield 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}'
        yield "event: message_stop"
        yield 'data: {"type":"message_stop"}'


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse, *args, **kwargs):
        self.response = response
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def build_request(self, method, url, **kwargs):
        return (method, url, kwargs)

    async def send(self, request, *, stream=False):
        await asyncio.sleep(self.response.header_delay_seconds)
        return self.response

    async def aclose(self):
        self.closed = True


def _build_deps(
    *,
    enabled: bool,
    seconds: float,
    logs: list[str],
    saved: list[dict],
    header_keepalive_enabled: bool = False,
    header_keepalive_seconds: float = 3.0,
):
    def save_request_log(*args, **kwargs):
        saved.append(kwargs)

    def build_error(status: int, message: str, error_type: str) -> bytes:
        return (
            "data: "
            + json.dumps({"error": {"status": status, "message": message, "type": error_type}})
            + "\n\n"
        ).encode()

    return AnthropicMessagesDeps(
        log=logs.append,
        save_request_log=save_request_log,
        build_openai_sse_error=build_error,
        has_stop_tag=lambda text: False,
        find_stop_tag=lambda text: -1,
        fmt_ms=lambda start, end=None: f"{((end or time.perf_counter()) - start) * 1000:.1f}ms",
        release_caller=lambda caller, trace: None,
        header_keepalive_enabled=header_keepalive_enabled,
        header_keepalive_interval_sec=header_keepalive_seconds,
        stream_idle_timeout_enabled=enabled,
        stream_idle_timeout_sec=seconds,
    )


async def _collect_stream(response: _FakeResponse, deps: AnthropicMessagesDeps):
    client = _FakeAsyncClient(response)

    def client_factory(*args, **kwargs):
        return client

    with patch(
        "aetherstream.upstreams.anthropic_messages.chat_stream.httpx.AsyncClient",
        side_effect=client_factory,
    ):
        chunks = [
            chunk
            async for chunk in forward_anthropic_messages_as_chat_stream(
                url="https://example.invalid/v1/messages",
                request_data={"model": "claude-test", "messages": [], "stream": True},
                headers={},
                model="claude-test",
                messages=[],
                trace_id="idle-test",
                caller_key="",
                caller_desc="",
                timeout=httpx.Timeout(30),
                max_raw_sse_bytes=1024 * 1024,
                deps=deps,
            )
        ]
    return b"".join(chunks), client


class AnthropicStreamIdleTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_keepalive_while_waiting_for_upstream_headers(self):
        logs: list[str] = []
        saved: list[dict] = []
        response = _FakeResponse(pause_seconds=0.0, header_delay_seconds=0.25)
        deps = _build_deps(
            enabled=False,
            seconds=0.05,
            logs=logs,
            saved=saved,
            header_keepalive_enabled=True,
            header_keepalive_seconds=0.05,
        )

        body, _client = await _collect_stream(response, deps)

        self.assertGreaterEqual(body.count(b'"choices"'), 3)
        self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
        header_keepalives = [line for line in logs if "claude_downstream_header_keepalive" in line]
        self.assertGreaterEqual(len(header_keepalives), 2)
        self.assertTrue(any("reason=message_stop:end_turn" in line for line in logs))

    async def test_closes_upstream_after_data_stream_stalls(self):
        logs: list[str] = []
        saved: list[dict] = []
        response = _FakeResponse(pause_seconds=1.0)
        deps = _build_deps(enabled=True, seconds=0.05, logs=logs, saved=saved)

        started = time.perf_counter()
        body, client = await _collect_stream(response, deps)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.5)
        self.assertTrue(response.closed)
        self.assertTrue(client.closed)
        self.assertIn(b'"type": "upstream_stream_idle_timeout"', body)
        self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
        self.assertTrue(any("claude_upstream_data_idle_timeout" in line for line in logs))
        self.assertEqual(saved[-1]["error_type"], "claude_upstream_data_idle_timeout")

    async def test_disabled_switch_allows_same_pause_to_finish(self):
        logs: list[str] = []
        saved: list[dict] = []
        response = _FakeResponse(pause_seconds=0.1)
        deps = _build_deps(enabled=False, seconds=0.05, logs=logs, saved=saved)

        body, _client = await _collect_stream(response, deps)

        self.assertNotIn(b"upstream_stream_idle_timeout", body)
        self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
        self.assertTrue(any("reason=message_stop:end_turn" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
