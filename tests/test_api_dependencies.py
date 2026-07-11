import ast
import inspect
import unittest

from aetherstream.api import admin_routes, chat_routes, messages_routes, system_routes
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
            (chat_routes.chat_completions, chat_routes.CHAT_DEPENDENCY_NAMES),
            (messages_routes.anthropic_messages, messages_routes.MESSAGES_DEPENDENCY_NAMES),
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
            self.assertEqual(used, set(declared), handlers)


if __name__ == '__main__':
    unittest.main()
