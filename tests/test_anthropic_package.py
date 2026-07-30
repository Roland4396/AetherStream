import inspect
import unittest

from aetherstream.upstreams import anthropic_messages


class AnthropicPackageTests(unittest.TestCase):
    def test_public_upstream_contract_is_reexported(self):
        expected = {
            'AnthropicMessagesDeps',
            'collect_anthropic_messages_as_chat_completion',
            'collect_anthropic_chat_completion_from_raw_sse',
            'collect_anthropic_messages_response',
            'forward_anthropic_messages_as_chat_stream',
            'forward_anthropic_messages_stream',
            'replay_anthropic_chat_stream',
        }

        self.assertEqual(set(anthropic_messages.__all__), expected)
        for name in expected:
            self.assertTrue(callable(getattr(anthropic_messages, name)), name)

    def test_primary_call_signatures_remain_keyword_compatible(self):
        signatures = {
            'forward_anthropic_messages_stream': {
                'url', 'request_data', 'headers', 'timeout', 'deps',
                'enable_early_stop', 'trace_id',
            },
            'forward_anthropic_messages_as_chat_stream': {
                'url', 'request_data', 'headers', 'model', 'messages',
                'trace_id', 'caller_key', 'caller_desc', 'timeout',
                'max_raw_sse_bytes', 'deps', 'cache_keepalive',
            },
            'collect_anthropic_messages_as_chat_completion': {
                'url', 'request_data', 'headers', 'model', 'messages',
                'trace_id', 'timeout', 'max_raw_sse_bytes', 'deps',
            },
            'collect_anthropic_messages_response': {
                'url', 'request_data', 'headers', 'model', 'timeout',
                'max_raw_sse_bytes', 'deps', 'trace_id',
            },
        }

        for name, expected_parameters in signatures.items():
            parameters = set(inspect.signature(getattr(anthropic_messages, name)).parameters)
            self.assertEqual(parameters, expected_parameters, name)
