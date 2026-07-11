import unittest

from aetherstream.features.opus_notes import (
    PRO_OPUS_LAST_USER_APPEND_MARKER,
    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
)
from aetherstream.features.request_injections import (
    ASSISTANT_PREFILL_CONTINUATION_TEXT,
    append_assistant_prefill_continuation,
    apply_direct_opus_note,
    apply_pioneer_opus_note,
)


class RequestInjectionTests(unittest.TestCase):
    def test_assistant_prefill_continuation_is_shared_and_conditional(self):
        logs = []
        request = {"messages": [{"role": "assistant", "content": "prefill"}]}

        changed = append_assistant_prefill_continuation(
            request,
            selected_model="claude-test",
            trace_prefix="[TRACE test]",
            route_label="route",
            log=logs.append,
        )

        self.assertTrue(changed)
        self.assertEqual(request["messages"][-1]["role"], "user")
        self.assertEqual(request["messages"][-1]["content"], ASSISTANT_PREFILL_CONTINUATION_TEXT)
        self.assertEqual(len(logs), 1)

    def test_pioneer_runtime_switch_prevents_all_opus_injection(self):
        request = {"messages": [{"role": "user", "content": "text"}]}
        apply_pioneer_opus_note(
            request,
            selected_model="claude-opus-4-6",
            trace_prefix="[TRACE test]",
            route_label="free_openai",
            runtime_lookup=lambda *keys: False,
            is_opus_model=lambda _model: True,
            log=lambda _message: None,
        )

        self.assertEqual(request["messages"][0]["content"], "text")

    def test_direct_opus_injection_is_idempotent(self):
        request = {
            "messages": [{
                "role": "user",
                "content": "<latest_human_message>hello</latest_human_message>",
            }]
        }
        kwargs = {
            "selected_model": "claude-opus-4-6",
            "trace_prefix": "[TRACE test]",
            "route_label": "claude_code",
            "is_opus_model": lambda _model: True,
            "log": lambda _message: None,
        }

        apply_direct_opus_note(request, **kwargs)
        apply_direct_opus_note(request, **kwargs)

        content = request["messages"][0]["content"]
        self.assertEqual(content.count(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER), 3)
        self.assertEqual(content.count(PRO_OPUS_LAST_USER_APPEND_MARKER), 1)


if __name__ == "__main__":
    unittest.main()
