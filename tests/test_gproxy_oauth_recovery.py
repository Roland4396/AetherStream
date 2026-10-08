import asyncio
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import httpx

from aetherstream.features.gproxy_oauth import GproxyOAuthRecovery
from aetherstream.runtime.shared import SharedRuntimeState
from aetherstream.upstreams.anthropic_messages.transport import (
    _send_with_oauth_recovery, _stream_with_oauth_recovery,
)


URL = 'http://account-pool-proxy:3200/v1/messages?beta=true'
ACCOUNT = 'gproxy_claude_subscription_1'
ERROR = {'type': 'error', 'error': {'type': 'authentication_error',
         'message': 'OAuth access token has expired. Re-authenticate to continue.'}}


def rejected(status=401, account=ACCOUNT, body=ERROR):
    return httpx.Response(status, headers={'x-account-pool-id': account}, json=body)


class FakeAdmin:
    def __init__(self):
        self.providers = [{'id': 1, 'name': 'claude', 'channel': 'claudecode', 'enabled': True}]
        self.credentials = [{'id': 30, 'auth_kind': 'oauth', 'enabled': True, 'status': 'active'}]
        self.expiry = None
        self.calls = []
        self.refreshes = 0
        self.failure = None
        self.entered = asyncio.Event()
        self.pause_refresh = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def api(self, path, data=None):
        self.calls.append((path, data))
        if path == 'providers':
            return self.providers
        if path == 'providers/1/credentials':
            return self.credentials
        if path == 'credentials/v3-credentials-30':
            return {'expiresAtMs': self.expiry, 'status': 'active'}
        if path == 'credentials/v3-credentials-30/refresh?force=true':
            self.entered.set()
            if self.pause_refresh:
                await asyncio.sleep(100)
            await asyncio.sleep(.025)
            if self.failure:
                raise self.failure
            self.refreshes += 1
            self.expiry = int(time.time() * 1000) + 3600000
            return {'expiresAtMs': self.expiry, 'status': 'active', 'version': 2}
        raise AssertionError('Unexpected API path: ' + path)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.shared = SharedRuntimeState(self.temp.name)
        self.admin = FakeAdmin()
        self.logs = []
        self.recovery = self.make_recovery(self.shared)

    def make_recovery(self, shared):
        return GproxyOAuthRecovery(base_url='http://gproxy:8787', credentials_file='unused',
            trusted_url='http://account-pool-proxy:3200', shared_state=shared,
            log=self.logs.append, client_factory=lambda: self.admin)

    async def test_missing_imported_expiry_uses_existing_native_refresh_only(self):
        self.assertTrue(await self.recovery.recover(rejected(), URL))
        self.assertEqual(self.admin.refreshes, 1)
        self.assertEqual(self.admin.calls[-1], ('credentials/v3-credentials-30/refresh?force=true', {}))
        self.assertFalse(any('generate' in p or 'quota' in p or 'reveal' in p for p, _ in self.admin.calls))

    async def test_fresh_peer_material_is_not_rotated_again(self):
        self.admin.expiry = int(time.time() * 1000) + 3600000
        self.assertTrue(await self.recovery.recover(rejected(), URL))
        self.assertEqual(self.admin.refreshes, 0)

    async def test_independent_release_instances_share_refresh_lock(self):
        peer = self.make_recovery(SharedRuntimeState(self.temp.name))
        result = await asyncio.gather(self.recovery.recover(rejected(), URL),
                                      peer.recover(rejected(), URL))
        self.assertEqual(result, [True, True])
        self.assertEqual(self.admin.refreshes, 1)

    async def test_unrelated_failures_routes_and_origins_do_not_call_admin(self):
        cases = [
            (rejected(status=403), URL), (rejected(status=429), URL),
            (rejected(status=503), URL), (rejected(status=200), URL),
            (rejected(account='gproxy_1'), URL), (rejected(account='paid_fallback'), URL),
            (rejected(), 'https://account-pool-proxy:3200/v1/messages'),
            (rejected(), 'http://account-pool-proxy:9999/v1/messages'),
            (rejected(), 'https://other.invalid/v1/messages'),
            (rejected(body={'error': {'type': 'authentication_error', 'message': 'invalid key'}}), URL),
            (rejected(body={'error': {'type': 'permission_error', 'message': ERROR['error']['message']}}), URL),
            (httpx.Response(401, headers={'x-account-pool-id': ACCOUNT}, content=b'not-json'), URL),
        ]
        for response, url in cases:
            with self.subTest(status=response.status_code, url=url):
                self.assertFalse(await self.recovery.recover(response, url))
        self.assertEqual(self.admin.calls, [])

    async def test_no_shared_coordinator_fails_closed(self):
        self.assertFalse(await self.make_recovery(None).recover(rejected(), URL))
        self.assertEqual(self.admin.calls, [])

    async def test_multiple_or_disabled_credentials_are_not_guessed(self):
        self.admin.credentials.append({**self.admin.credentials[0], 'id': 31})
        self.assertFalse(await self.recovery.recover(rejected(), URL))
        self.assertEqual(self.admin.refreshes, 0)

    async def test_disabled_provider_is_not_restored(self):
        self.admin.providers[0]['enabled'] = False
        self.assertFalse(await self.recovery.recover(rejected(), URL))
        self.assertEqual(self.admin.refreshes, 0)

    async def test_refresh_error_never_logs_secret_or_changes_pause_state(self):
        secret = 'PRIVATE-TOKEN-MUST-NOT-LEAK'
        self.admin.failure = httpx.HTTPStatusError(secret, request=httpx.Request('POST', 'http://gproxy/'),
                                                 response=httpx.Response(400, text=secret))
        self.assertFalse(await self.recovery.recover(rejected(), URL))
        self.assertNotIn(secret, '\n'.join(self.logs))
        self.assertIn('http_status=400', '\n'.join(self.logs))
        self.assertEqual(self.admin.credentials[0]['status'], 'active')

    async def test_cancellation_releases_lock_without_detached_refresh(self):
        self.admin.pause_refresh = True
        operation = asyncio.create_task(self.recovery.recover(rejected(), URL))
        await self.admin.entered.wait()
        operation.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await operation
        self.admin.pause_refresh = False
        self.assertTrue(await asyncio.wait_for(self.recovery.recover(rejected(), URL), 2))
        self.assertEqual(self.admin.refreshes, 1)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def run_send(self, first=401, final=200, recovery_result=True):
        requests = []
        recoveries = []
        closed = []
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield json.dumps(ERROR).encode()
            async def aclose(self):
                closed.append(True)
        async def handler(request):
            requests.append((request.method, str(request.url), dict(request.headers), await request.aread()))
            if len(requests) == 1:
                return httpx.Response(first, headers={'x-account-pool-id': ACCOUNT}, stream=Body())
            return httpx.Response(final, content=b'final')
        async def recover(response, url, prefix):
            recoveries.append(response.status_code)
            await response.aread()
            return recovery_result
        deps = SimpleNamespace(recover_oauth=recover, log=lambda _: None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_with_oauth_recovery(client=client, url=URL,
                request_data={'model': 'claude-opus-5-5', 'messages': [{'role': 'user', 'content': 'fake'}]},
                headers={'x-api-key': 'fake-key'}, deps=deps)
            await response.aclose()
        return response, requests, recoveries, closed

    async def test_one_retry_has_identical_payload_headers_and_closes_rejection(self):
        response, requests, recoveries, closed = await self.run_send()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(recoveries, [401])
        self.assertTrue(closed)

    async def test_second_expired_response_is_terminal(self):
        response, requests, recoveries, _ = await self.run_send(final=401)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(len(requests), 2)
        self.assertEqual(recoveries, [401])

    async def test_no_replay_when_recovery_fails(self):
        response, requests, _, _ = await self.run_send(recovery_result=False)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(len(requests), 1)

    async def test_success_and_non_auth_failures_are_never_replayed(self):
        for status in [200, 403, 429, 500, 503]:
            response, requests, recoveries, _ = await self.run_send(first=status)
            self.assertEqual(response.status_code, status)
            self.assertEqual(len(requests), 1)
            self.assertEqual(recoveries, [])

    async def test_cancellation_during_recovery_closes_response(self):
        entered = asyncio.Event()
        closed = []
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'error'
            async def aclose(self):
                closed.append(True)
        async def handler(request):
            return httpx.Response(401, stream=Body())
        async def recover(*args):
            entered.set()
            await asyncio.sleep(100)
        deps = SimpleNamespace(recover_oauth=recover, log=lambda _: None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(_send_with_oauth_recovery(client=client, url=URL,
                request_data={}, headers={}, deps=deps))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(closed)

    async def test_stream_context_uses_recovery_and_closes_final_response(self):
        calls = []
        async def handler(request):
            calls.append(await request.aread())
            return rejected() if len(calls) == 1 else httpx.Response(200, content=b'ok')
        async def recover(*args):
            return True
        deps = SimpleNamespace(recover_oauth=recover, log=lambda _: None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with _stream_with_oauth_recovery(client=client, url=URL,
                    request_data={'stream': False}, headers={}, deps=deps) as response:
                self.assertEqual(response.status_code, 200)
            self.assertTrue(response.is_closed)
        self.assertEqual(len(calls), 2)
