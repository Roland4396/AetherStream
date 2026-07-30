import unittest

import anthropic_upstream
import codex_upstream
import gemini_upstream
import openai_upstream
from aetherstream.upstreams import (
    anthropic_messages,
    gemini_generate_content,
    openai_chat_completions,
    openai_responses,
)


class ProtocolModuleNamesTests(unittest.TestCase):
    def test_canonical_modules_export_protocol_named_contracts(self):
        expected = {
            openai_chat_completions: (
                "ChatCompletionsUpstreamDeps",
                "forward_chat_completions_stream",
                "collect_chat_completions_stream",
            ),
            openai_responses: (
                "ResponsesUpstreamDeps",
                "forward_responses_as_chat_stream",
                "collect_responses_as_chat_completion",
            ),
            gemini_generate_content: (
                "GeminiGenerateContentConfig",
                "GeminiGenerateContentDeps",
                "forward_gemini_generate_content_stream",
            ),
            anthropic_messages: (
                "AnthropicMessagesDeps",
                "forward_anthropic_messages_stream",
                "collect_anthropic_messages_response",
            ),
        }

        for module, names in expected.items():
            for name in names:
                self.assertTrue(callable(getattr(module, name)), f"{module.__name__}.{name}")

    def test_legacy_root_wrappers_keep_old_import_contracts(self):
        expected = {
            openai_upstream: ("OpenAIUpstreamDeps", "forward_stream", "collect_stream"),
            codex_upstream: ("CodexUpstreamDeps", "forward_codex_chat_stream"),
            gemini_upstream: ("GeminiUpstreamConfig", "forward_gemini_stream"),
            anthropic_upstream: ("AnthropicUpstreamDeps", "forward_anthropic_stream"),
        }

        for module, names in expected.items():
            for name in names:
                self.assertTrue(callable(getattr(module, name)), f"{module.__name__}.{name}")


if __name__ == "__main__":
    unittest.main()
