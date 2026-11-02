import ast
import inspect
import unittest

from aetherstream.api import audio_routes, admin_routes, chat_routes, messages_routes, system_routes, channel_gateway, request_entry
from aetherstream.api.dependencies import build_route_dependencies


class RouteDependenciesTests(unittest.TestCase):
    def test_missing_dependencies_fail_during_route_registration(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing route dependencies: logger'):
            build_route_dependencies({'service': object()}, ('service', 'logger'))

    def test_dependency_can_be_replaced_for_a_focused_test(self):
        original = object()
        replacement = object()
        dependencies = build_route_dependencies({'service': original}, ('service',))

        self.assertIs(dependencies.service, original)
        dependencies.service = replacement
        self.assertIs(dependencies.service, replacement)

    def test_route_dependency_declarations_match_handler_usage(self):
        cases = (
            ((chat_routes.chat_completions,
              chat_routes._plain_claude_when_conversion_disabled,
              chat_routes._claude_nonstream_to_stream_enabled,
              channel_gateway.chat_completions, channel_gateway.apply_request_policy,
              channel_gateway.prepare_adapter, channel_gateway.forward_prepared,
              channel_gateway.collect_prepared, channel_gateway._owned_stream,
              request_entry.protocol_entry), chat_routes.CHAT_DEPENDENCY_NAMES),
            ((messages_routes.anthropic_messages, chat_routes._claude_nonstream_to_stream_enabled),
             messages_routes.MESSAGES_DEPENDENCY_NAMES),
            (
                (audio_routes._send_upstream, audio_routes.create_speech, audio_routes.voices),
                audio_routes.AUDIO_DEPENDENCY_NAMES,
            ),
            (
                (
                    admin_routes.admin_claude_replay_state,
                    admin_routes.admin_claude_replay_update,
                    admin_routes.admin_claude_replay_log_output,
                ),
                admin_routes.ADMIN_DEPENDENCY_NAMES,
            ),
            ((system_routes.models, system_routes.health), system_routes.SYSTEM_DEPENDENCY_NAMES),
        )

        for handlers, declared in cases:
            if not isinstance(handlers, tuple):
                handlers = (handlers,)
            used = set()
            for handler in handlers:
                tree = ast.parse(inspect.getsource(handler))
                used.update(
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == 'deps'
                )
            # Optional compatibility dependencies use getattr so old focused
            # fixtures can omit the managed registry without reading config.
            for handler in handlers:
                tree = ast.parse(inspect.getsource(handler))
                used.update(node.args[1].value for node in ast.walk(tree)
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == 'getattr' and len(node.args) >= 2
                            and isinstance(node.args[0], ast.Name) and node.args[0].id == 'deps'
                            and isinstance(node.args[1], ast.Constant))
            self.assertEqual(used, set(declared), handlers)


if __name__ == '__main__':
    unittest.main()
