import unittest

from aetherstream.upstreams.anthropic_messages.collectors import _ToolCallAccumulator


class ToolCallAccumulatorTests(unittest.TestCase):
    def test_collects_parallel_tool_calls_and_incremental_arguments(self):
        calls = _ToolCallAccumulator()
        calls.start(2, {"type": "tool_use", "id": "toolu_write", "name": "Write", "input": {}})
        calls.append_delta(2, {"type": "input_json_delta", "partial_json": '{"file_'})
        calls.append_delta(2, {"type": "input_json_delta", "partial_json": 'path":"/tmp/a"}'})
        calls.start(3, {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {}})

        self.assertEqual(
            calls.as_openai(),
            [
                {
                    "id": "toolu_write",
                    "type": "function",
                    "function": {"name": "Write", "arguments": '{"file_path":"/tmp/a"}'},
                },
                {
                    "id": "toolu_read",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
