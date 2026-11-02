"""Synthetic SSE only: no provider calls, live credentials, or real reasoning text."""
import asyncio
import copy
import json
import time
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from aetherstream.upstreams import openai_chat_completions as chat


def event(delta=None, finish=None, **extra):
    return {
        "id": "offline-tail", "model": "claude-offline", "created": 123,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        **extra,
    }


REASONING = event({"reasoning_content": "synthetic reasoning fixture"})
STOP = event(finish="stop")
TAG = "<offline-stop>"


class ScriptBody(httpx.AsyncByteStream):
    def __init__(self, script, hang):
        self.script = script
        self.hang = hang
        self.closed = False

    async def __aiter__(self):
        for item in self.script:
            if isinstance(item, float):
                await asyncio.sleep(item)
            elif isinstance(item, Exception):
                raise item
            else:
                payload = json.dumps(item) if isinstance(item, dict) else item
                yield f"data: {payload}\n\n".encode()
                await asyncio.sleep(0)
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def decode(chunks):
    rows = []
    for line in b"".join(chunks).decode().splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            rows.append(payload if payload == "[DONE]" else json.loads(payload))
    return rows


def finishes(rows):
    return [row for row in rows if isinstance(row, dict)
            and any(c.get("finish_reason") for c in row.get("choices", []))]


def body_text(rows):
    return "".join(c.get("delta", {}).get("content", "")
                   for row in rows if isinstance(row, dict)
                   for c in row.get("choices", []))


@asynccontextmanager
async def fixture(script, *, grace=0.08, hang=False, supersede=None):
    body = ScriptBody(script, hang)
    logs, saved, requests, released = [], [], [], []
    deferred = asyncio.Event()
    payload = {
        "model": "claude-offline", "stream": True, "max_tokens": 30000,
        "stop": ["<end>"], "messages": [{"role": "user", "content": "Keep this exact text. 原文不变。"}],
    }
    original = copy.deepcopy(payload)

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"x-account-pool-id": "anti5_opus55_1"}, stream=body)

    def log(message):
        logs.append(message)
        if "empty_reasoning_finish_deferred " in message:
            deferred.set()

    def error(status, message, error_type):
        return ("data: " + json.dumps({"error": {"status": status, "message": message,
                                                 "type": error_type}}) + "\n\n").encode()

    deps = chat.ChatCompletionsUpstreamDeps(
        log=log, save_request_log=lambda *a, **kw: saved.append((a, kw)),
        build_openai_sse_error=error, has_stop_tag=lambda text: TAG in text,
        find_stop_tag=lambda text: text.find(TAG),
        fmt_ms=lambda start, end=None: f"{((end or time.perf_counter()) - start) * 1000:.1f}ms",
        release_caller=lambda *a: released.append(a),
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    stream = chat.forward_chat_completions_stream(
        url="https://fixture.invalid/v1/chat/completions", request_data=payload,
        headers={}, timeout=httpx.Timeout(2), max_raw_sse_bytes=1024 * 1024,
        deps=deps, model=payload["model"], messages=payload["messages"],
        trace_id="offline-tail", caller_key="offline-caller", supersede_event=supersede,
    )
    with patch.object(chat.httpx, "AsyncClient", return_value=client), \
            patch.object(chat, "REASONING_ONLY_FINISH_GRACE_SEC", grace), \
            patch.object(chat, "observe_natural_finish") as natural, \
            patch.object(chat, "disconnect_on_early_stop", return_value=False):
        try:
            yield SimpleNamespace(stream=stream, body=body, logs=logs, saved=saved,
                                  requests=requests, released=released, payload=payload,
                                  original=original, client=client, natural=natural, deferred=deferred)
        finally:
            await stream.aclose()
            await client.aclose()


class ReasoningOnlyFinishTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, run, timeout=2):
        async with asyncio.timeout(timeout):
            rows = decode([chunk async for chunk in run.stream])
        self.assertTrue(run.body.closed)
        self.assertTrue(run.client.is_closed)
        self.assertEqual(run.requests, [run.original], "exactly one unmodified upstream request")
        self.assertEqual(run.payload, run.original)
        self.assertEqual(run.released, [("offline-caller", "offline-tail")])
        self.assertEqual(len(run.saved), 1)
        return rows

    def assert_terminal_once(self, rows, reason="stop"):
        self.assertEqual(rows.count("[DONE]"), 1)
        self.assertEqual([c["finish_reason"] for row in finishes(rows) for c in row["choices"]], [reason])

    async def test_body_after_preliminary_stop_is_preserved_without_retry_or_request_change(self):
        async with fixture([REASONING, STOP, 0.01, event({"content": "continued body"}), STOP, "[DONE]"]) as run:
            rows = await self.collect(run)
            self.assertEqual(body_text(rows), "continued body")
            self.assert_terminal_once(rows)
            run.natural.assert_called_once_with("stop")
            self.assertTrue(any("empty_reasoning_finish_resumed" in s for s in run.logs))
            self.assertIn("continued body", run.saved[0][1]["raw_sse"])
            self.assertEqual(run.saved[0][1]["raw_sse"].count('"finish_reason": "stop"'), 2)
            self.assertFalse(any("synthetic reasoning fixture" in s for s in run.logs))

    async def test_resumed_generation_outlives_old_grace_deadline(self):
        async with fixture([REASONING, STOP, 0.01, event({"content": "one"}),
                            0.12, event({"content": "two"}), STOP], grace=0.05) as run:
            rows = await self.collect(run)
            self.assertEqual(body_text(rows), "onetwo")
            self.assert_terminal_once(rows)
            run.natural.assert_called_once_with("stop")

    async def test_resumed_reasoning_also_cancels_old_deadline(self):
        async with fixture([REASONING, STOP, 0.01, REASONING, 0.12,
                            event({"content": "answer"}), STOP], grace=0.05) as run:
            rows = await self.collect(run)
            self.assertEqual(body_text(rows), "answer")
            self.assert_terminal_once(rows)

    async def test_actual_done_finalizes_without_waiting_for_grace(self):
        async with fixture([REASONING, STOP, "[DONE]"], grace=10, hang=True) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assert_terminal_once(rows)
            run.natural.assert_called_once_with("stop")
            self.assertIn("[EMPTY_REASONING_FINISH: done]", run.saved[0][1]["raw_sse"])

    async def test_clean_eof_finalizes_without_waiting_for_grace(self):
        async with fixture([REASONING, STOP], grace=10) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assert_terminal_once(rows)
            run.natural.assert_called_once_with("stop")
            self.assertIn("[EMPTY_REASONING_FINISH: eof]", run.saved[0][1]["raw_sse"])

    async def test_hanging_stop_is_bounded(self):
        async with fixture([REASONING, STOP], grace=0.04, hang=True) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assert_terminal_once(rows)
            self.assertIn("[EMPTY_REASONING_FINISH: grace_expired]", run.saved[0][1]["raw_sse"])

    async def test_usage_only_tail_is_retained(self):
        usage = {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
        async with fixture([REASONING, STOP, {"choices": [], "usage": usage}, "[DONE]"]) as run:
            rows = await self.collect(run)
            self.assert_terminal_once(rows)
            self.assertEqual(finishes(rows)[0]["usage"], usage)
            self.assertFalse(any("finish_resumed" in s for s in run.logs))

    async def test_duplicate_stops_and_metadata_do_not_restart_deadline(self):
        script = [REASONING, STOP]
        for _ in range(40):
            script += [0.01, STOP, event({"role": "assistant"}), {"choices": [], "usage": {}}]
        async with fixture(script, grace=0.04, hang=True) as run:
            rows = await self.collect(run, timeout=0.25)
            self.assert_terminal_once(rows)
            self.assertEqual(sum("finish_deferred " in s for s in run.logs), 1)
            self.assertTrue(any("finish_duplicate " in s for s in run.logs))
            self.assertIn("[EMPTY_REASONING_FINISH: grace_expired]", run.saved[0][1]["raw_sse"])

    async def test_already_produced_body_keeps_immediate_finish(self):
        async with fixture([REASONING, event({"content": "done"}), STOP], grace=10, hang=True) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assert_terminal_once(rows)
            self.assertFalse(run.deferred.is_set())

    async def test_no_reasoning_keeps_immediate_empty_finish(self):
        async with fixture([event({"role": "assistant"}), STOP], grace=10, hang=True) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assert_terminal_once(rows)
            self.assertFalse(run.deferred.is_set())

    async def test_length_filter_and_tool_finish_are_not_deferred(self):
        for reason in ("length", "content_filter", "tool_calls"):
            with self.subTest(reason=reason):
                async with fixture([REASONING, event(finish=reason)], grace=10, hang=True) as run:
                    rows = await self.collect(run, timeout=0.5)
                    self.assert_terminal_once(rows, reason)
                    self.assertFalse(run.deferred.is_set())

    async def test_explicit_refusal_delta_and_stop_details_are_not_deferred(self):
        for refusal in (event({"refusal": "fixture"}), event(stop_details={"type": "refusal"}),
                        {"choices": [{"delta": {}, "stop_details": {"type": "refusal"}}]}):
            with self.subTest(refusal=refusal):
                async with fixture([REASONING, refusal, STOP], grace=10, hang=True) as run:
                    rows = await self.collect(run, timeout=0.5)
                    self.assert_terminal_once(rows)
                    self.assertFalse(run.deferred.is_set())

    async def test_nonterminal_tool_data_prevents_defer(self):
        tool = event({"tool_calls": [{"index": 0, "id": "tool-offline", "type": "function",
                                      "function": {"name": "ordinary_tool", "arguments": "{}"}}]})
        async with fixture([REASONING, tool, STOP], grace=10, hang=True) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assert_terminal_once(rows)
            self.assertFalse(run.deferred.is_set())

    async def test_reasoning_delta_on_stop_frame_is_not_lost(self):
        async with fixture([event({"reasoning_content": "synthetic last delta"}, finish="stop"), "[DONE]"]) as run:
            rows = await self.collect(run)
            self.assert_terminal_once(rows)
            reasoning = [c["delta"]["reasoning_content"] for row in rows if isinstance(row, dict)
                         for c in row.get("choices", []) if "reasoning_content" in c.get("delta", {})]
            self.assertEqual(reasoning, ["synthetic last delta"])

    async def test_stop_detection_still_closes_resumed_body_immediately(self):
        async with fixture([REASONING, STOP, event({"content": "prefix" + TAG + "hidden"})], grace=10, hang=True) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assertEqual(body_text(rows), "prefix")
            self.assert_terminal_once(rows)
            run.natural.assert_not_called()
            self.assertIn("[EARLY_STOP by proxy]", run.saved[0][1]["raw_sse"])

    async def test_terminal_tool_still_closes_pending_stream(self):
        tool = event({"tool_calls": [{"index": 0, "id": "terminal", "type": "function",
                                      "function": {"name": chat.TERMINAL_TOOL_NAME, "arguments": "{}"}}]})
        async with fixture([REASONING, STOP, tool], grace=10, hang=True) as run:
            rows = await self.collect(run, timeout=0.5)
            self.assert_terminal_once(rows)
            run.natural.assert_not_called()
            self.assertIn("[TERMINAL_TOOL by proxy]", run.saved[0][1]["raw_sse"])

    async def test_cancel_while_pending_cleans_up_without_counting_completion(self):
        async with fixture([REASONING, STOP], grace=10, hang=True) as run:
            async def consume():
                return [chunk async for chunk in run.stream]
            task = asyncio.create_task(consume())
            await asyncio.wait_for(run.deferred.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(run.body.closed)
            self.assertTrue(run.client.is_closed)
            self.assertEqual(run.released, [("offline-caller", "offline-tail")])
            run.natural.assert_not_called()
            self.assertEqual(run.requests, [run.original])

    async def test_supersede_while_pending_cleans_up(self):
        supersede = asyncio.Event()
        async with fixture([REASONING, STOP, 0.02, event({"role": "assistant"})],
                           grace=10, hang=True, supersede=supersede) as run:
            task = asyncio.create_task(self.collect(run))
            await asyncio.wait_for(run.deferred.wait(), 1)
            supersede.set()
            rows = await task
            self.assertFalse(finishes(rows))
            self.assertNotIn("[DONE]", rows)
            run.natural.assert_not_called()

    async def test_transport_failure_remains_an_error_without_retry(self):
        async with fixture([REASONING, STOP, httpx.ReadError("synthetic broken connection")]) as run:
            rows = await self.collect(run)
            self.assertFalse(finishes(rows))
            self.assertEqual([row["error"]["type"] for row in rows if isinstance(row, dict) and "error" in row],
                             ["proxy_stream_error"])
            run.natural.assert_not_called()

    async def test_explicit_upstream_error_remains_an_error(self):
        async with fixture([REASONING, STOP, {"error": {"message": "synthetic failure", "type": "upstream_error"}}]) as run:
            rows = await self.collect(run)
            self.assertFalse(finishes(rows))
            self.assertTrue(any(isinstance(row, dict) and "error" in row for row in rows))
            run.natural.assert_not_called()


if __name__ == "__main__":
    unittest.main()
