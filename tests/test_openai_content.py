import unittest

from aetherstream.upstreams.openai_chat_completions import (
    _coerce_openai_text_content,
)


class OpenAIContentTests(unittest.TestCase):
    def test_null_content_is_an_empty_fragment(self):
        self.assertEqual(_coerce_openai_text_content(None), "")

    def test_string_and_text_part_content_are_preserved(self):
        self.assertEqual(_coerce_openai_text_content("plain"), "plain")
        self.assertEqual(
            _coerce_openai_text_content([
                {"type": "text", "text": "first"},
                " second",
                {"type": "tool_call", "name": "ignored"},
            ]),
            "first second",
        )


if __name__ == "__main__":
    unittest.main()
