import copy
import unittest

from aetherstream.features.terminal_tool import (
    TERMINAL_TOOL_DESCRIPTION,
    TERMINAL_TOOL_HISTORY_EXAMPLE_ID,
    TERMINAL_TOOL_SYSTEM_PROMPT,
    anthropic_terminal_tool,
    gemini_terminal_declaration,
    inject_anthropic_terminal_tool,
    inject_openai_chat_terminal_tool,
    openai_chat_terminal_tool,
    openai_responses_terminal_tool,
    prepend_anthropic_terminal_prompt,
    terminal_tool_parameters,
    terminal_tool_enabled_for_model,
)


class TerminalToolDefinitionTests(unittest.TestCase):
    def test_terminal_tool_is_disabled_globally(self):
        for model in (
            "claude-fable-5",
            "gemini-3.1-pro",
            "TRANSSION/99",
            "deepseek-v4-flash-local",
            "GLM-5.2-LOCAL",
            "gpt-5.5",
            "",
            None,
        ):
            with self.subTest(model=model):
                self.assertFalse(terminal_tool_enabled_for_model(model))

    def test_openai_local_payload_is_not_modified(self):
        payload = {
            "model": "deepseek-v4-flash-local",
            "messages": [{"role": "user", "content": "plan this scene"}],
        }
        original = copy.deepcopy(payload)

        self.assertFalse(inject_openai_chat_terminal_tool(payload))
        self.assertEqual(payload, original)

    def test_openai_gemini_payload_is_not_modified(self):
        payload = {
            "model": "agy-gemini-3.1-pro-low",
            "messages": [
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "current question"},
            ],
            "tools": [{"type": "function", "function": {"name": "ordinary"}}],
        }
        original = copy.deepcopy(payload)

        self.assertFalse(inject_openai_chat_terminal_tool(payload))
        self.assertEqual(payload, original)

    def test_anthropic_gemini_payload_is_not_modified(self):
        payload = {
            "model": "free/gemini-test",
            "system": [{"type": "text", "text": "existing"}],
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "ordinary", "input_schema": {"type": "object"}}],
        }
        original = copy.deepcopy(payload)

        self.assertFalse(inject_anthropic_terminal_tool(payload))
        self.assertEqual(payload, original)

    def test_description_covers_with_and_without_other_tools(self):
        self.assertIn("Call it exactly once", TERMINAL_TOOL_DESCRIPTION)
        self.assertIn("final action", TERMINAL_TOOL_DESCRIPTION)
        self.assertIn("non-empty user-visible response", TERMINAL_TOOL_DESCRIPTION)
        self.assertIn("Never call it as the first or only content block", TERMINAL_TOOL_DESCRIPTION)
        self.assertIn("After receiving results from ordinary tools", TERMINAL_TOOL_DESCRIPTION)
        self.assertIn("If other tools are needed", TERMINAL_TOOL_DESCRIPTION)
        self.assertIn("without waiting for tool results", TERMINAL_TOOL_DESCRIPTION)
        self.assertIn("Generic filler is invalid", TERMINAL_TOOL_DESCRIPTION)

    def test_all_protocols_share_the_mandatory_description(self):
        descriptions = (
            openai_chat_terminal_tool()["function"]["description"],
            openai_responses_terminal_tool()["description"],
            anthropic_terminal_tool()["description"],
            gemini_terminal_declaration()["description"],
        )
        self.assertTrue(all(value == TERMINAL_TOOL_DESCRIPTION for value in descriptions))

    def test_string_parameter_minimum_lengths_remain_zero(self):
        properties = terminal_tool_parameters()["properties"]
        self.assertTrue(properties)
        strings = [spec for spec in properties.values() if spec["type"] == "string"]
        self.assertTrue(strings)
        self.assertTrue(all(spec["minLength"] == 0 for spec in strings))

    def test_schema_uses_five_fields_and_typed_score(self):
        parameters = terminal_tool_parameters()
        self.assertEqual(
            parameters["required"],
            ["answer_summary", "diagnosis", "improvement_plan", "overall_score", "score_rationale"],
        )
        score = parameters["properties"]["overall_score"]
        self.assertEqual(score["type"], "integer")
        self.assertEqual((score["minimum"], score["maximum"]), (0, 100))

    def test_string_fields_define_language_and_examples(self):
        properties = terminal_tool_parameters()["properties"]
        for spec in properties.values():
            if spec["type"] != "string":
                continue
            self.assertIn("BAD:", spec["description"])
            self.assertIn("GOOD:", spec["description"])
            self.assertIn("Write in Simplified Chinese", spec["description"])

    def test_anthropic_injection_is_disabled(self):
        payload = {"tools": [{"name": "ordinary", "input_schema": {"type": "object"}}]}
        original = copy.deepcopy(payload)

        self.assertFalse(inject_anthropic_terminal_tool(payload))

        self.assertEqual(payload, original)

    def test_neutral_prompt_uses_exact_tool_name(self):
        self.assertEqual(
            TERMINAL_TOOL_SYSTEM_PROMPT,
            "请在本次回复中调用 submit_response_self_evaluation 工具，使用它完成对本次回复的自我评价。",
        )

    def test_openai_injection_is_disabled(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        original = copy.deepcopy(payload)

        self.assertFalse(inject_openai_chat_terminal_tool(payload))

        self.assertEqual(payload, original)

    def test_anthropic_prompt_is_first_and_deduplicated(self):
        payload = {"system": [{"type": "text", "text": "existing"}]}

        prepend_anthropic_terminal_prompt(payload)
        prepend_anthropic_terminal_prompt(payload)

        self.assertEqual(payload["system"][0]["text"], TERMINAL_TOOL_SYSTEM_PROMPT)
        self.assertEqual(payload["system"][1]["text"], "existing")
        self.assertEqual(len(payload["system"]), 2)

    def test_openai_history_is_not_modified(self):
        payload = {
            "messages": [
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "current question"},
            ]
        }

        original = copy.deepcopy(payload)

        self.assertFalse(inject_openai_chat_terminal_tool(payload))

        self.assertEqual(payload, original)

    def test_anthropic_history_is_not_modified(self):
        payload = {
            "messages": [
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
                {"role": "user", "content": "current question"},
            ],
            "tools": [],
        }

        original = copy.deepcopy(payload)

        self.assertFalse(inject_anthropic_terminal_tool(payload))

        self.assertEqual(payload, original)

if __name__ == "__main__":
    unittest.main()
