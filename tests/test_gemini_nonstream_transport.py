import asyncio
import ast
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import httpx
from fastapi.responses import JSONResponse

from aetherstream.streaming.responses import DisconnectSafeStreamingResponse
from aetherstream.upstreams import gemini_generate_content as gemini


REAL_CLIENT = httpx.AsyncClient
ROOT = Path(gemini.__file__).parents[1]


def response_body(text="OK", finish="STOP", parts=None):
    return {
        "candidates": [{"content": {"parts": parts or [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 1, "totalTokenCount": 7},
    }


def parse_events(chunks):
    return [json.loads(line[6:]) for chunk in chunks for line in chunk.decode().splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


class GeminiNonstreamTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.logs = []
        self.saved = []
        self.config = gemini.GeminiGenerateContentConfig(
            base_url="https://gemini.invalid", api_key="offline-test", include_thoughts=False,
            heartbeat_interval=0.01, max_retries=0, retry_delay=0, timeout=httpx.Timeout(3),
        )
        self.deps = gemini.GeminiGenerateContentDeps(
            log=self.logs.append, save_request_log=lambda *a, **kw: self.saved.append((a, kw)),
            build_openai_sse_error=lambda status, message, error_type: (
                "data: " + json.dumps({"error": {"code": status, "message": message, "type": error_type}}) + "\n\n"
            ).encode(),
        )
        self.payload = {"model": "gemini-3.1-pro-preview", "stream": True,
                        "messages": [{"role": "user", "content": "Reply OK."}]}

    def transport(self, handler):
        async def dispatch(request):
            self.requests.append(request)
            self.assertEqual(request.method, "POST")
            self.assertTrue(request.url.path.endswith(":generateContent"))
            self.assertNotIn("streamGenerateContent", str(request.url))
            self.assertFalse(request.url.query)
            self.assertEqual(request.headers["accept"], "application/json")
            self.assertNotIn("stream", json.loads(request.content))
            return await handler(request)
        return patch.object(gemini.httpx, "AsyncClient",
                            side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(dispatch), **kw))

    def stream(self, model="gemini-3.1-pro-preview"):
        return gemini.forward_gemini_generate_content_stream(
            model=model, openai_request=self.payload, config=self.config, deps=self.deps,
            messages=self.payload["messages"], trace_id="offline-native",
        )

    async def collect(self):
        return await gemini.collect_gemini_generate_content(
            model="gemini-3.1-pro-preview", openai_request=self.payload,
            config=self.config, deps=self.deps, trace_id="offline-native",
        )

    async def test_native_model_matrix_always_uses_nonstream(self):
        async def handler(_):
            return httpx.Response(200, json=response_body())
        for model in ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash"]:
            with self.subTest(model=model), self.transport(handler):
                chunks = [c async for c in self.stream(model)]
                events = parse_events(chunks)
                self.assertEqual("".join(e["choices"][0]["delta"].get("content", "") for e in events), "OK")
                self.assertEqual(events[-1]["usage"]["total_tokens"], 7)
                self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
                self.assertEqual(chunks.count(b"data: [DONE]\n\n"), 1)
                self.assertIn(model + ":generateContent", self.requests[-1].url.path)
        self.assertEqual(len(self.requests), 3)

    async def test_nonstream_collector_uses_same_json_endpoint(self):
        async def handler(_):
            return httpx.Response(200, json=response_body())
        with self.transport(handler):
            content, usage, finish = await self.collect()
        self.assertEqual((content, usage["total_tokens"], finish), ("OK", 7, "stop"))
        self.assertEqual(len(self.requests), 1)

    async def test_no_content_is_emitted_before_full_upstream_body(self):
        body_complete = asyncio.Event()
        release = asyncio.Event()
        class DelayedBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                raw = json.dumps(response_body()).encode()
                yield raw[:15]
                await release.wait()
                body_complete.set()
                yield raw[15:]
        async def handler(_):
            return httpx.Response(200, headers={"content-type": "application/json"}, stream=DelayedBody())
        with self.transport(handler):
            stream = self.stream()
            self.assertNotIn('"content"', (await anext(stream)).decode())
            heartbeat = await asyncio.wait_for(anext(stream), 1)
            self.assertFalse(body_complete.is_set())
            self.assertNotIn('"content"', heartbeat.decode())
            release.set()
            chunks = [c async for c in stream]
        self.assertTrue(body_complete.is_set())
        self.assertIn('"content": "OK"', b"".join(chunks).decode())

    async def test_downstream_close_cancels_pending_upstream(self):
        cancelled = asyncio.Event()
        async def handler(_):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        with self.transport(handler):
            stream = self.stream()
            await anext(stream)
            await anext(stream)  # request started, then heartbeat
            await stream.aclose()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(self.requests), 1)

    async def test_upstream_error_is_not_an_empty_success(self):
        async def handler(_):
            return httpx.Response(429, json={"error": {"message": "quota test"}})
        with self.transport(handler):
            chunks = [c async for c in self.stream()]
        events = parse_events(chunks)
        self.assertEqual(events[-1]["error"]["code"], 429)
        self.assertEqual(chunks.count(b"data: [DONE]\n\n"), 1)
        self.assertEqual(len(self.requests), 1)

    async def test_invalid_json_surfaces_error(self):
        async def handler(_):
            return httpx.Response(200, content=b"not json")
        with self.transport(handler):
            events = parse_events([c async for c in self.stream()])
        self.assertEqual(events[-1]["error"]["code"], 502)

    async def test_transport_timeout_is_not_retried(self):
        async def handler(request):
            raise httpx.ReadTimeout("offline timeout", request=request)
        with self.transport(handler):
            events = parse_events([c async for c in self.stream()])
        self.assertEqual(events[-1]["error"]["code"], 502)
        self.assertEqual(len(self.requests), 1)

    async def test_tool_calls_survive_replay(self):
        parts = [{"text": "result"}, {"functionCall": {"name": "lookup", "args": {"x": 1}}}]
        async def handler(_):
            return httpx.Response(200, json=response_body(parts=parts))
        with self.transport(handler):
            events = parse_events([c async for c in self.stream()])
        tools = [t for e in events for t in e["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual(tools[0]["function"], {"name": "lookup", "arguments": '{"x":1}'})
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")

    async def test_include_thoughts_setting_is_preserved(self):
        self.config.include_thoughts = True
        async def handler(_):
            return httpx.Response(200, json=response_body(parts=[{"text": "visible"}, {"text": "thought", "thought": True}]))
        with self.transport(handler):
            content, _, _ = await self.collect()
        self.assertEqual(content, "thought")

    async def test_long_response_replay_and_length_finish(self):
        async def handler(_):
            return httpx.Response(200, json=response_body(text="x" * 2601, finish="MAX_TOKENS"))
        with self.transport(handler):
            events = parse_events([c async for c in self.stream()])
        parts = [e["choices"][0]["delta"]["content"] for e in events if "content" in e["choices"][0]["delta"]]
        self.assertEqual(list(map(len, parts)), [1200, 1200, 201])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "length")

    async def test_actual_native_route_both_client_modes(self):
        # Execute the deployed route's native Gemini branch, not a copied routing implementation.
        path = ROOT / "api/chat_routes.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "chat_completions")
        branch = None
        for node in ast.walk(function):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "deps.model_policy.is_gemini_model(model)":
                if "GEMINI_ENABLED" in ast.unparse(node):
                    branch = node
                    break
        self.assertIsNotNone(branch)
        wrapper = ast.parse("async def route(data, model, stream, deps):\n trace_id='offline-route'\n trace_prefix='[TRACE offline-route]'\n route_t0=time.perf_counter()\n pass\n").body[0]
        wrapper.body[-1:] = [copy.deepcopy(branch)]
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        ns = dict(copy=copy, time=time, uuid=uuid, JSONResponse=JSONResponse, StreamingResponse=DisconnectSafeStreamingResponse)
        exec(compile(module, str(path), "exec"), ns)
        async def dedupe(**kw):
            return await kw["runner"](), False
        deps = SimpleNamespace(
            model_policy=SimpleNamespace(is_gemini_model=lambda _: True), GEMINI_ENABLED=True, GEMINI_API_KEY="offline",
            log=self.logs.append, fmt_ms=lambda *_: "0ms", apply_drawing_context_filter=lambda _: {},
            build_gemini_generate_content_config=lambda: self.config, build_gemini_generate_content_deps=lambda: self.deps,
            forward_gemini_generate_content_stream=gemini.forward_gemini_generate_content_stream,
            collect_gemini_generate_content=gemini.collect_gemini_generate_content,
            build_exact_request_key=lambda _: "offline", run_exact_nonstream_once=dedupe,
            extract_openai_chat_payload_content=lambda p: p["choices"][0]["message"]["content"],
            save_request_log=lambda *a, **kw: None,
        )
        async def handler(_):
            return httpx.Response(200, json=response_body())
        for mode in (True, False):
            with self.subTest(client_stream=mode), self.transport(handler):
                response = await ns["route"](copy.deepcopy(self.payload), "gemini-3.1-pro-preview", mode, deps)
                if mode:
                    chunks = [chunk async for chunk in response.body_iterator]
                    self.assertEqual(chunks[-1], b"data: [DONE]\n\n")
                else:
                    self.assertEqual(json.loads(response.body)["choices"][0]["message"]["content"], "OK")
        self.assertEqual(len(self.requests), 2)


if __name__ == "__main__":
    unittest.main()
