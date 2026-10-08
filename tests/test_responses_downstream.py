from __future__ import annotations

import json
import unittest

from aetherstream.transforms.downstream_protocols import (
    chat_stream_to_responses,
    responses_to_chat_request,
)


async def chunks(*values: bytes):
    for value in values:
        yield value


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.decode().strip().split("\n\n"):
        lines = block.splitlines()
        if not lines or lines[0].startswith(":"):
            continue
        event_line = next(line for line in lines if line.startswith("event: "))
        data_line = next(line for line in lines if line.startswith("data: "))
        events.append((event_line[7:], json.loads(data_line[6:])))
    return events


class ResponsesDownstreamTests(unittest.IsolatedAsyncioTestCase):
    def test_ds4_removes_empty_developer_string(self) -> None:
        converted = responses_to_chat_request({
            "model": "deepseek-v4-flash-local",
            "input": [
                {"role": "developer", "content": "  \n"},
                {"role": "user", "content": "hello"},
            ],
        })

        self.assertEqual(converted["messages"], [{"role": "user", "content": "hello"}])

    def test_ds4_removes_empty_developer_array(self) -> None:
        for content in ([], [{"type": "input_text", "text": "  "}]):
            with self.subTest(content=content):
                converted = responses_to_chat_request({
                    "model": "deepseek-v4-flash-local",
                    "input": [{"role": "developer", "content": content}],
                })

                self.assertEqual(converted["messages"], [])

    def test_ds4_converts_nonempty_developer_string_to_system(self) -> None:
        converted = responses_to_chat_request({
            "model": "deepseek-v4-flash-local",
            "input": [{"role": "developer", "content": "instructions"}],
        })

        self.assertEqual(
            converted["messages"],
            [{"role": "system", "content": "instructions"}],
        )

    def test_ds4_converts_nonempty_developer_array_to_system(self) -> None:
        converted = responses_to_chat_request({
            "model": "deepseek-v4-flash-local",
            "input": [{
                "role": "developer",
                "content": [{"type": "input_text", "text": "instructions"}],
            }],
        })

        self.assertEqual(
            converted["messages"],
            [{"role": "system", "content": "instructions"}],
        )

    def test_other_model_preserves_nonempty_developer(self) -> None:
        converted = responses_to_chat_request({
            "model": "another-model",
            "input": [{"role": "developer", "content": "instructions"}],
        })

        self.assertEqual(
            converted["messages"],
            [{"role": "developer", "content": "instructions"}],
        )

    def test_other_model_removes_empty_developer(self) -> None:
        converted = responses_to_chat_request({
            "model": "another-model",
            "input": [{"role": "developer", "content": ""}],
        })

        self.assertEqual(converted["messages"], [])

    async def test_text_stream_emits_complete_codex_event_sequence(self) -> None:
        upstream = chunks(
            b'data: {"id":"chatcmpl-1","model":"deepseek-v4-flash-local","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-1","model":"deepseek-v4-flash-local","choices":[{"delta":{"content":"DS4_CODEX_OK"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-1","model":"deepseek-v4-flash-local","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n\n',
            b'data: [DONE]\n\n',
        )
        raw = b"".join([
            chunk async for chunk in chat_stream_to_responses(
                upstream,
                model="deepseek-v4-flash-local",
            )
        ])
        events = parse_sse(raw)
        event_types = [event_type for event_type, _ in events]
        self.assertEqual(
            event_types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [payload["sequence_number"] for _, payload in events],
            list(range(len(events))),
        )
        self.assertEqual(events[4][1]["delta"], "DS4_CODEX_OK")
        completed = events[-1][1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "DS4_CODEX_OK")


if __name__ == "__main__":
    unittest.main()
