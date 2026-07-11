import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from aetherstream.api import app as app_module
from aetherstream.features.claude_replay import ReplayStore
from aetherstream.features.replay import ReplayService, parse_replay_record


def _envelope(body: str, model: str = "test-model") -> str:
    return (
        "Time: 20260711_000000\n"
        f"Model: {model}\n"
        "TraceId: test-trace\n"
        + "=" * 50
        + "\n"
        + body
    )


def _openai_chunk(content: str = "", finish_reason=None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


class ReplayParserTests(unittest.TestCase):
    def test_openai_incomplete_stream_is_a_usable_snapshot(self):
        raw = _envelope(
            _openai_chunk("partial ")
            + _openai_chunk("answer")
            + "[CANCELLED: downstream client disconnected]\n"
        )
        record = parse_replay_record(raw, fallback_model="requested")

        self.assertEqual(record.source_format, "openai_sse")
        self.assertEqual(record.text, "partial answer")
        self.assertFalse(record.source_complete)
        self.assertIn("CANCELLED", record.source_error)

    def test_plain_openai_json_preserves_reasoning(self):
        payload = {
            "id": "chatcmpl-json",
            "object": "chat.completion",
            "model": "json-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "brief reasoning",
                    "content": "final answer",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
        record = parse_replay_record(
            _envelope(json.dumps(payload) + "\n[NONSTREAM_REPLAY_FINISH_REASON: stop]"),
            fallback_model="requested",
        )

        self.assertEqual(record.source_format, "openai_json")
        self.assertEqual(record.text, "final answer")
        self.assertEqual(record.reasoning_content, "brief reasoning")
        self.assertTrue(record.source_complete)
        self.assertEqual(record.usage["total_tokens"], 5)

    def test_anthropic_sse_is_normalized(self):
        events = [
            {"type": "message_start", "message": {"model": "claude-test", "usage": {"input_tokens": 4}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            {"type": "message_stop"},
        ]
        raw = _envelope("".join(f"data: {json.dumps(event)}\n\n" for event in events))
        record = parse_replay_record(raw, fallback_model="requested")

        self.assertEqual(record.source_format, "anthropic_sse")
        self.assertEqual(record.model, "claude-test")
        self.assertEqual(record.text, "hello")
        self.assertTrue(record.source_complete)

    def test_responses_sse_preserves_summary_and_output(self):
        events = [
            {"type": "response.created", "response": {"model": "gpt-test"}},
            {"type": "response.reasoning_summary_text.delta", "delta": "summary"},
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed", "response": {"model": "gpt-test", "usage": {"input_tokens": 7, "output_tokens": 3}}},
        ]
        raw = _envelope("".join(f"data: {json.dumps(event)}\n\n" for event in events))
        record = parse_replay_record(raw, fallback_model="requested")

        self.assertEqual(record.source_format, "responses_sse")
        self.assertEqual(record.reasoning_content, "summary")
        self.assertEqual(record.text, "answer")
        self.assertTrue(record.source_complete)
        self.assertEqual(record.usage["total_tokens"], 10)

    def test_gemini_sse_is_normalized(self):
        event = {
            "candidates": [{
                "content": {"parts": [{"text": "thought", "thought": True}, {"text": "answer"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        }
        record = parse_replay_record(
            _envelope(f"data: {json.dumps(event)}\n\n"),
            fallback_model="gemini-test",
        )

        self.assertEqual(record.source_format, "gemini_sse")
        self.assertEqual(record.reasoning_content, "thought")
        self.assertEqual(record.text, "answer")
        self.assertTrue(record.source_complete)


class ReplayStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp.name)
        self.control_path = self.log_dir / "replay.json"
        self.store = ReplayStore(
            log_dir=str(self.log_dir),
            control_file=str(self.control_path),
            log=lambda _message: None,
        )
        self.messages = [{"role": "user", "content": "hello"}]
        (self.log_dir / "01_raw_sse.txt").write_text(
            _envelope(_openai_chunk("saved") + _openai_chunk(finish_reason="stop")),
            encoding="utf-8",
        )
        (self.log_dir / "01_input.json").write_text(
            json.dumps({"model": "free/claude-opus-4-6", "messages": self.messages}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_lookup_matches_original_request(self):
        self.store.write_control({
            "enabled": True,
            "mode": "always",
            "match_request": True,
            "raw_sse_path": "01_raw_sse.txt",
        })
        result = self.store.lookup(model="free/claude-opus-4-6", messages=self.messages)
        mismatch = self.store.lookup(model="different", messages=self.messages)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(mismatch["status"], "mismatch")

        entries = self.store.list_entries()
        self.assertEqual(entries[0]["replay_format"], "openai_sse")
        self.assertTrue(entries[0]["replay_usable"])
        self.assertTrue(entries[0]["replay_source_complete"])

    def test_once_does_not_consume_a_newer_selection(self):
        (self.log_dir / "02_raw_sse.txt").write_text(
            _envelope(_openai_chunk("new") + _openai_chunk(finish_reason="stop")),
            encoding="utf-8",
        )
        self.store.write_control({
            "enabled": True,
            "mode": "once",
            "match_request": False,
            "raw_sse_path": "01_raw_sse.txt",
        })
        old_spec = self.store.lookup(model="any", messages=[])["spec"]
        self.store.write_control({
            "enabled": True,
            "mode": "once",
            "match_request": False,
            "raw_sse_path": "02_raw_sse.txt",
        })

        consumed = self.store.consume_if_needed(old_spec)

        self.assertFalse(consumed)
        self.assertTrue(self.store.read_control()["enabled"])
        self.assertEqual(self.store.read_control()["raw_sse_path"], "02_raw_sse.txt")

    def test_once_claim_is_atomic(self):
        self.store.write_control({
            "enabled": True,
            "mode": "once",
            "match_request": False,
            "raw_sse_path": "01_raw_sse.txt",
        })
        spec = self.store.lookup(model="any", messages=[])["spec"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda _index: self.store.consume_if_needed(spec), range(2)))

        self.assertEqual(sorted(claims), [False, True])
        self.assertFalse(self.store.read_control()["enabled"])

    def test_malformed_control_is_invalid_instead_of_disabled(self):
        self.control_path.write_text('{not-json', encoding='utf-8')

        result = self.store.lookup(model="any", messages=[])
        state = self.store.build_state()

        self.assertEqual(result["status"], "invalid")
        self.assertIn("failed to read replay control", result["reason"])
        self.assertIn("failed to read replay control", state["control_error"])


class ReplayRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp.name)
        self.control_path = self.log_dir / "replay.json"
        self.store = ReplayStore(
            log_dir=str(self.log_dir),
            control_file=str(self.control_path),
            log=lambda _message: None,
        )
        (self.log_dir / "01_raw_sse.txt").write_text(
            _envelope(
                _openai_chunk("route-independent replay")
                + "[CANCELLED: downstream client disconnected]\n"
            ),
            encoding="utf-8",
        )
        self.store.write_control({
            "enabled": True,
            "mode": "once",
            "match_request": False,
            "raw_sse_path": "01_raw_sse.txt",
        })
        self.saved = []
        self.service = ReplayService(
            store=self.store,
            log=lambda _message: None,
            save_request_log=lambda *args, **kwargs: self.saved.append((args, kwargs)),
            release_caller=app_module.release_active_stream_caller,
        )
        self.client = TestClient(app_module.app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_free_provider_stream_is_intercepted_before_routing(self):
        request = {
            "model": "free/claude-opus-4-6",
            "stream": True,
            "messages": [{"role": "user", "content": "must not reach upstream"}],
        }
        with patch.object(app_module.chat_route_dependencies, "replay_service", self.service):
            response = self.client.post("/v1/chat/completions", json=request)

        self.assertEqual(response.status_code, 200)
        self.assertIn("route-independent replay", response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertFalse(self.store.read_control()["enabled"])
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(len(app_module.active_stream_registry), 0)

    def test_nonstream_replay_uses_the_same_interceptor(self):
        self.store.write_control({
            "enabled": True,
            "mode": "always",
            "match_request": False,
            "raw_sse_path": "01_raw_sse.txt",
        })
        request = {
            "model": "unknown-future-provider/model",
            "stream": False,
            "messages": [{"role": "user", "content": "must not route"}],
        }
        with patch.object(app_module.chat_route_dependencies, "replay_service", self.service):
            response = self.client.post("/v1/chat/completions", json=request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "route-independent replay")

    def test_invalid_enabled_replay_fails_closed_before_routing(self):
        self.control_path.write_text('{not-json', encoding='utf-8')
        request = {
            "model": "unknown-future-provider/model",
            "stream": False,
            "messages": [{"role": "user", "content": "must not route"}],
        }

        with patch.object(app_module.chat_route_dependencies, "replay_service", self.service):
            response = self.client.post("/v1/chat/completions", json=request)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["type"], "replay_invalid")


if __name__ == "__main__":
    unittest.main()
