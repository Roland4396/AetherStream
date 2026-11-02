import copy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from aetherstream.features.kunlun_linux import (
    DEFAULT_TOOL_PREFIX,
    KunlunStreamDecoder,
    SOURCE_SHA256,
    prepare_kunlun_linux,
    selected_settings,
    unwrap_completion,
)
from aetherstream.features.output_tool import OutputToolStreamDecoder, record_marker, record_settings
from aetherstream.routing.policy_config import enrich_selection
from aetherstream.api.channel_gateway import prepare_adapter


CONFIG = {
    "enabled": True,
    "match": {
        "channels": ["free"],
        "public_models": ["free/gemini-3.8-flash-high"],
    },
    "transports": ["chat"],
    "source_sha256": SOURCE_SHA256,
}


class ZeroShuffle:
    @staticmethod
    def shuffle(values):
        # JavaScript source uses Fisher-Yates. This is Math.random() === 0.
        for index in range(len(values) - 1, 0, -1):
            values[index], values[0] = values[0], values[index]


def prepared(payload, *, rng=None):
    logs = []
    body = copy.deepcopy(payload)
    with patch("aetherstream.features.kunlun_linux.secrets.token_hex", return_value="0001020304050607"):
        response = prepare_kunlun_linux(body, CONFIG, log=logs.append, rng=rng)
    return body, response, logs


class KunlunLinuxTests(unittest.TestCase):
    def test_policy_selects_kunlun_instead_of_stacking_existing_output_tool(self):
        output_tool = {
            "enabled": True,
            "match": {"regex": r"gemini-3[.]8-flash(?:$|[-_/])"},
            "transports": ["chat"],
            "tool": {"type": "function", "function": {
                "name": "actual_content", "description": "existing",
                "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
            }},
            "field": "content", "control_prompt": "existing", "anchors": [],
        }
        flags = {
            "channel_policies": {"schema": 1, "defaults": {}, "profiles": {}, "model_rules": [], "channels": {}},
            "kunlun_linux": CONFIG,
            "output_tool": output_tool,
        }
        target = {"channel": "free", "route": "r", "model": "free/gemini-3.8-flash-high",
                  "upstream_model": "gemini-3.8-flash-high", "transport": "chat"}
        policy = enrich_selection(lambda key: flags.get(key), target)["policy"]["runtime"]
        self.assertIn("kunlun_linux", policy)
        self.assertNotIn("output_tool", policy)
        other = enrich_selection(lambda key: flags.get(key), {**target, "channel": "anti"})["policy"]["runtime"]
        self.assertNotIn("kunlun_linux", other)
        self.assertIn("output_tool", other)

    def test_adapter_keeps_original_identity_while_wire_payload_uses_kunlun(self):
        data = {"model": "free/gemini-3.8-flash-high", "messages": [{"role": "user", "content": "正文任务"}], "stream": True}
        plan = {
            "transport": "chat", "base_url": "http://mock.invalid/v1", "api_key": "mock",
            "policy": {"runtime": {"kunlun_linux": CONFIG}},
        }
        deps = SimpleNamespace(log=lambda _message: None, build_chat_completions_upstream_deps=lambda: object())
        with patch("aetherstream.features.kunlun_linux.secrets.token_hex", return_value="0001020304050607"):
            result = prepare_adapter(data, plan, deps, trace_id="test", upstream_stream=False)
        self.assertEqual(result.identity_payload, {**data, "stream": False})
        self.assertEqual(result.identity_plan["policy"]["runtime"], {})
        self.assertEqual(result.payload["messages"][-1]["role"], "user")
        self.assertEqual(result.payload["tools"][-1]["function"]["name"], DEFAULT_TOOL_PREFIX + "0001020304050607")
        self.assertEqual(result.output_tool["kind"], "kunlun_linux")

    def test_selection_is_exact_public_model_channel_and_transport(self):
        base = {"channel": "free", "model": "free/gemini-3.8-flash-high", "transport": "chat"}
        self.assertEqual(selected_settings(CONFIG, base), CONFIG)
        for change in (
            {"channel": "anti"},
            {"model": "gemini-3.8-flash-high"},
            {"model": "free/gemini-3.8-flash"},
            {"transport": "gemini"},
        ):
            with self.subTest(change=change):
                self.assertIsNone(selected_settings(CONFIG, {**base, **change}))

    def test_request_port_preserves_caller_fields_tools_and_builds_reviewed_envelope(self):
        original = {
            "model": "free/gemini-3.8-flash-high",
            "stream": False,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": "【资料配置】\n插入数量 = 0\n<角色表>\n甲\n乙\n</角色表>"},
                {"role": "system", "content": "【ToolCall引导】工具说明\n交付完整正文。\n【ToolCall引导】字段说明\n正文。"},
                {"role": "user", "content": "【HISTORICAL_USER_INPUT】旧问题"},
                {"role": "assistant", "content": "旧回答"},
                {"role": "user", "content": "【LAST_USER_INPUT】新问题"},
            ],
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
        }
        body, response, logs = prepared(original)
        name = DEFAULT_TOOL_PREFIX + "0001020304050607"
        self.assertEqual(body["model"], original["model"])
        self.assertIs(body["stream"], False)
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["tools"][:-1], original["tools"])
        self.assertEqual(body["tools"][-1]["function"]["name"], name)
        self.assertEqual(body["tools"][-1]["function"]["parameters"]["required"], ["display_stream"])
        description = body["tools"][-1]["function"]["description"]
        self.assertIn("交付完整正文。", description)
        self.assertIn("【POSIX 存储节点·历史章节文件连续读取规范】", description)
        self.assertIn("旧回答", description)
        self.assertIn("新问题", description)
        self.assertNotIn("时空界标", description)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual([item["role"] for item in body["messages"]], ["system", "user", "assistant", "user"])
        self.assertTrue(all(name in item["content"] for item in body["messages"]))
        self.assertEqual(response["kind"], "kunlun_linux")
        self.assertEqual(response["client_tool_names"], ["weather"])
        self.assertEqual(len(logs), 1)
        self.assertIn("kunlun_linux injected", logs[0])

    def test_two_historical_users_activate_exact_count_and_runtime_scene_text(self):
        payload = {
            "model": "free/gemini-3.8-flash-high",
            "messages": [
                {"role": "system", "content": "【资料配置】\n插入数量 = 3\n<角色表>\n甲\n乙\n</角色表>"},
                {"role": "user", "content": "一"},
                {"role": "assistant", "content": "答一"},
                {"role": "user", "content": "二"},
                {"role": "assistant", "content": "答二"},
                {"role": "user", "content": "三"},
            ],
        }
        body, _, _ = prepared(payload, rng=ZeroShuffle())
        description = body["tools"][-1]["function"]["description"]
        self.assertEqual(description.count("〔时空界标：大千微尘折跃 · 现实停驻一瞬〕"), 3)
        self.assertEqual(description.count("〔时空归位：现世锚点复原〕"), 3)
        self.assertIn("甲的心智与年轮无声蜕变", description)
        self.assertIn("scene_06_preview.txt", description)

    def test_caller_tool_control_bypasses_without_mutation(self):
        for choice in ("none", "required", {"type": "function", "function": {"name": "x"}}):
            payload = {"model": "free/gemini-3.8-flash-high", "messages": [{"role": "user", "content": "x"}], "tool_choice": choice}
            with self.subTest(choice=choice):
                body, response, logs = prepared(payload)
                self.assertEqual(body, payload)
                self.assertIsNone(response)
                self.assertIn("skipped=", logs[0])

    def test_nonstream_unwrap_matches_own_and_foreign_but_keeps_client_tool(self):
        payload = {
            "model": "free/gemini-3.8-flash-high",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "dispatch_stream_payload_client", "parameters": {"type": "object"}}}],
        }
        _, config, _ = prepared(payload)
        own = config["tool"]["function"]["name"]
        data = {"choices": [{"message": {"content": "短", "tool_calls": [
            {"function": {"name": own, "arguments": json.dumps({"display_stream": "完整正文"}, ensure_ascii=False)}},
            {"function": {"name": "render_document_chunk_deadbeef", "arguments": json.dumps({"content": "更长的恢复正文"}, ensure_ascii=False)}},
            {"function": {"name": "dispatch_stream_payload_client", "arguments": "{}"}},
        ]}, "finish_reason": "tool_calls"}]}
        result = unwrap_completion(data, config)
        self.assertEqual(result["choices"][0]["message"]["content"], "更长的恢复正文")
        self.assertEqual([c["function"]["name"] for c in result["choices"][0]["message"]["tool_calls"]],
                         ["dispatch_stream_payload_client"])
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")

    def test_stream_decoder_restores_split_json_string_and_normalizes_finish(self):
        _, config, _ = prepared({"messages": [{"role": "user", "content": "x"}]})
        name = config["tool"]["function"]["name"]
        decoder = KunlunStreamDecoder(config)
        first = decoder.feed({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": name, "arguments": '{"display_stream":"你'}}]}, "finish_reason": None}]})
        second = decoder.feed({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '好\\n世界"}'}}]}, "finish_reason": "tool_calls"}]})
        self.assertEqual(first["choices"][0]["delta"]["content"], "你")
        self.assertEqual(second["choices"][0]["delta"]["content"], "好\n世界")
        self.assertEqual(second["choices"][0]["finish_reason"], "stop")

    def test_output_tool_dispatch_and_replay_marker_round_trip(self):
        _, config, _ = prepared({"messages": [{"role": "user", "content": "x"}]})
        marker = record_marker(config).strip()
        restored = record_settings([marker])
        self.assertEqual(restored, config)
        decoder = OutputToolStreamDecoder(restored)
        name = config["tool"]["function"]["name"]
        event = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": name, "arguments": '{"display_stream":"正文"}'}}]}, "finish_reason": "tool_calls"}]}
        out = decoder.feed(event)
        self.assertEqual(out["choices"][0]["delta"]["content"], "正文")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
