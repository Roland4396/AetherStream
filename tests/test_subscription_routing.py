import ast
from pathlib import Path
import unittest
from unittest.mock import AsyncMock

from aetherstream.routing.model_policy import ModelPolicy, OPENAI_SUBSCRIPTION_MODELS


APP_PATH = Path(__file__).resolve().parents[1] / 'aetherstream/api/app.py'


def routing_functions():
    # Exercise the actual routing functions without starting services or reading
    # production credentials during an offline unit test.
    tree = ast.parse(APP_PATH.read_text())
    names = {'_upstream_allows_model', 'resolve_openai_compatible_route'}
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    ast.fix_missing_locations(module)
    policy = ModelPolicy(frozenset(), OPENAI_SUBSCRIPTION_MODELS,
                         frozenset(), frozenset(), frozenset())
    namespace = {'model_policy': policy, 'log': lambda _: None,
                 '_requires_native_claude_route': lambda *_: False}
    exec(compile(module, str(APP_PATH), 'exec'), namespace)
    return namespace


class SubscriptionRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = routing_functions()

    def test_only_verified_models_are_exposed(self):
        self.assertEqual(OPENAI_SUBSCRIPTION_MODELS, {
            'gpt-6-astra', 'gpt-6.1-sol', 'gpt-6-sol', 'gpt-6-luna', 'gpt-5.6-sol',
            'gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-5.5', 'codex-auto-review',
        })

    def test_third_party_catalog_cannot_reintroduce_gpt_routes(self):
        allowed = self.ns['_upstream_allows_model']
        for model in (*OPENAI_SUBSCRIPTION_MODELS, 'gpt-legacy-retired'):
            for upstream in ({}, {'include_model_families': ['gpt']}):
                with self.subTest(model=model, upstream=upstream):
                    self.assertFalse(allowed(upstream, model))

    async def test_gpt_ignores_stale_legacy_directory_route(self):
        refresh = AsyncMock(return_value={'routes': {
            model: {'name': 'old-paid', 'base_url': 'https://old.invalid/v1'}
            for model in OPENAI_SUBSCRIPTION_MODELS
        }})
        self.ns['refresh_model_directory'] = refresh
        for model in (*OPENAI_SUBSCRIPTION_MODELS, 'gpt-legacy-retired'):
            self.assertIsNone(await self.ns['resolve_openai_compatible_route'](model))
        refresh.assert_not_awaited()

    async def test_other_model_routes_unchanged(self):
        allowed = self.ns['_upstream_allows_model']
        for model in ('grok-4.5', 'claude-opus-4-6', 'gemini-3.1-pro-preview', 'kimi-k3'):
            with self.subTest(model=model):
                self.assertTrue(allowed({}, model))
                route = {'name': 'existing-route', 'base_url': 'https://kept.invalid/v1'}
                self.ns['refresh_model_directory'] = AsyncMock(return_value={'routes': {model: route}})
                self.assertEqual(await self.ns['resolve_openai_compatible_route'](model), route)

    def test_family_filters_still_apply(self):
        allowed = self.ns['_upstream_allows_model']
        self.assertTrue(allowed({'include_model_families': ['claude']}, 'claude-opus-4-6'))
        self.assertFalse(allowed({'include_model_families': ['claude']}, 'grok-4.5'))


if __name__ == '__main__':
    unittest.main()
