"""The compatibility entry must not infer discovery from an empty model list."""
import unittest
from unittest.mock import patch

import httpx

from aetherstream.api import app


REAL_CLIENT = httpx.AsyncClient


class ExplicitDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def directory(self, upstreams):
        calls = []

        def request(req):
            calls.append(req)
            self.assertEqual(req.url.host, 'mio.invalid')
            return httpx.Response(200, json={'data': [{'id': 'mio-model'}]})

        def lookup(*keys):
            return upstreams if keys == ('openai_compatible_upstreams',) else None

        with patch.object(app, '_runtime_lookup', lookup), \
             patch.object(app, 'log', lambda *_: None), \
             patch.object(app, '_model_directory_cache', {}), \
             patch.object(httpx, 'AsyncClient',
                 lambda *a, **k: REAL_CLIENT(transport=httpx.MockTransport(request))):
            result = await app.refresh_model_directory(force=True, request_authorization='Bearer test')
        return result, calls

    async def test_missing_or_empty_static_list_does_not_enable_discovery(self):
        result, calls = await self.directory([
            {'name': 'missing', 'base_url': 'http://missing.invalid/v1'},
            {'name': 'empty', 'base_url': 'http://empty.invalid/v1', 'include_models': []},
        ])
        self.assertEqual(calls, [])
        self.assertEqual(result['routes'], {})

    async def test_only_explicit_mio_source_is_requested(self):
        result, calls = await self.directory([
            {'name': 'static', 'base_url': 'http://static.invalid/v1', 'include_models': ['fixed-model']},
            {'name': 'empty', 'base_url': 'http://empty.invalid/v1'},
            {'name': 'mio', 'base_url': 'http://mio.invalid/v1', 'discover_models': True},
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].url.path, '/v1/models')
        self.assertEqual(set(result['routes']), {'fixed-model', 'mio-model'})


if __name__ == '__main__':
    unittest.main()
