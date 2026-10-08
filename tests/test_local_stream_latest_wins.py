import asyncio
import unittest
from types import SimpleNamespace

import httpx

from aetherstream.api.chat_routes import LATEST_WINS_LOCAL_MODELS
from aetherstream.upstreams.openai_chat_completions import (
    forward_chat_completions_stream,
)


class LocalStreamLatestWinsTests(unittest.IsolatedAsyncioTestCase):
    def test_latest_wins_scope_is_limited_to_the_two_local_text_models(self):
        self.assertEqual(
            LATEST_WINS_LOCAL_MODELS,
            {"glm-5.2-local", "deepseek-v4-flash-local"},
        )

    async def test_superseded_stream_never_opens_upstream(self):
        logs = []
        releases = []
        event = asyncio.Event()
        event.set()
        deps = SimpleNamespace(
            log=logs.append,
            release_caller=lambda caller, trace: releases.append((caller, trace)),
            fmt_ms=lambda _start, _end=None: "0.0ms",
        )

        chunks = [
            chunk
            async for chunk in forward_chat_completions_stream(
                url="http://127.0.0.1:1/v1/chat/completions",
                request_data={"model": "glm-5.2-local", "stream": True},
                headers={},
                timeout=httpx.Timeout(None),
                max_raw_sse_bytes=1024,
                deps=deps,
                model="glm-5.2-local",
                messages=None,
                trace_id="old-trace",
                caller_key="caller-a",
                supersede_event=event,
            )
        ]

        self.assertEqual(chunks, [])
        self.assertEqual(releases, [("caller-a", "old-trace")])
        self.assertTrue(any("local_stream_superseded" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
