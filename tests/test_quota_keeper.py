import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from aetherstream.api.quota_keeper_routes import register_routes
from aetherstream.features.quota_keeper.client import GproxyClient, GENERATION_URL, MODEL, ProbeError
from aetherstream.features.quota_keeper.policy import account_groups, decision, windows
from aetherstream.features.quota_keeper.service import QuotaKeeper, handle_account, quota_failure
from aetherstream.features.quota_keeper.state import StateError, StateStore

NOW = 2_000_000_000
MEMBERS = [{'id': 13, 'provider_id': 4, 'enabled': True,
            'label': 'private@example.test pro', 'health': 'alive'}]


def snapshot(short_used=0, week_used=0, short_end=None, week_end=None, disabled=False):
    def entry(name, used, deadline, label=None):
        return {'id': name, 'source_id': 'subscription', 'label': label,
                'value': {'kind': 'window', 'used_percent': None if used is None else str(used),
                          'period_end': deadline}}
    return {'sources': [{'capability': {'id': 'subscription'},
                          'observed_at_ms': NOW * 1000, 'error': None}],
            'entries': [entry('3p-5h', short_used, NOW + 18000 if short_end is None else short_end,
                              'antigravity_disabled' if disabled else None),
                        entry('3p-weekly', week_used, NOW + 604800 if week_end is None else week_end),
                        entry('gemini-5h', 0, NOW - 500)]}


class FakeApi:
    def __init__(self, cached=None, fresh=None):
        self.cached_value = cached or snapshot()
        self.fresh = fresh or self.cached_value
        self.cached_calls = 0
        self.probes = []
        self.generations = []
        self.api_calls = []
        self.result = {'http_status': 200, 'ok': True, 'usage': {'totalTokenCount': 16}}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def api(self, name):
        self.api_calls.append(name)
        return [{'id': 4, 'channel': 'antigravity', 'enabled': True}] if name == 'providers' else MEMBERS

    async def cached(self, cid):
        self.cached_calls += 1
        return self.cached_value

    async def probe(self, cid):
        self.probes.append(cid)
        return self.fresh

    async def generate(self, cid):
        self.generations.append(cid)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class PolicyTests(unittest.TestCase):
    def test_running_clock_with_remaining_quota_waits(self):
        for used in (0.0032, 1, 20, 99, 100):
            self.assertEqual(decision(windows(snapshot(short_used=used)), NOW), ('wait', NOW + 18030))

    def test_unused_sliding_deadline_is_not_a_running_clock(self):
        self.assertEqual(decision(windows(snapshot()), NOW), ('unused', NOW))

    def test_weekly_exhaustion_waits_for_week(self):
        self.assertEqual(decision(windows(snapshot(short_used=None, week_used=100,
                         short_end=NOW - 1, disabled=True)), NOW), ('wait', NOW + 604830))

    def test_expired_boundary_needs_fresh_probe_not_generation(self):
        self.assertEqual(decision(windows(snapshot(short_end=NOW - 31)), NOW), ('due', NOW))
        self.assertEqual(decision(windows(snapshot(week_used=100, week_end=NOW-31)), NOW), ('due', NOW))

    def test_missing_quota_is_not_zero(self):
        self.assertEqual(decision({}, NOW), ('unknown', NOW))
        self.assertEqual(decision(windows(snapshot(short_used=None)), NOW), ('unknown', NOW))

    def test_gemini_ignored_and_duplicate_email_deduplicated(self):
        self.assertEqual(set(windows(snapshot())), {'3p-5h', '3p-weekly'})
        groups = account_groups(MEMBERS + [dict(MEMBERS[0], id=24),
                    dict(MEMBERS[0], id=4, enabled=False), dict(MEMBERS[0], id=28, provider_id=6)], {4})
        self.assertEqual([r['id'] for r in next(iter(groups.values()))], [13, 24])


class AccountTests(unittest.IsolatedAsyncioTestCase):
    async def run_account(self, api, row, now=NOW, persist=lambda: None, may_generate=lambda: True):
        await handle_account(api, MEMBERS, row, persist, lambda *a, **k: None,
                             clock=lambda: now, may_generate=may_generate)

    async def test_active_window_never_probes_or_generates(self):
        api = FakeApi(snapshot(short_used=20))
        row = {}
        await self.run_account(api, row)
        self.assertEqual(api.probes, [])
        self.assertEqual(api.generations, [])
        self.assertEqual(row['next_at'], NOW + 18030)

    async def test_normal_user_use_before_fresh_probe_skips_wakeup(self):
        api = FakeApi(snapshot(), snapshot(short_used=2))
        await self.run_account(api, {})
        self.assertEqual(api.probes, [13])
        self.assertEqual(api.generations, [])

    async def test_guard_persisted_before_post_and_one_request_only(self):
        api = FakeApi()
        row = {}
        persisted = []
        original = api.generate
        async def generate(cid):
            self.assertTrue(persisted[-1]['pending_verify'])
            self.assertEqual(persisted[-1]['triggers'], 1)
            self.assertEqual(persisted[-1]['guard_until'], NOW + 18030)
            return await original(cid)
        api.generate = generate
        await self.run_account(api, row, persist=lambda: persisted.append(dict(row)))
        await self.run_account(api, row, NOW + 1)
        self.assertEqual(api.generations, [13])

    async def test_confirmation_does_not_generate_again(self):
        api = FakeApi()
        row = {}
        await self.run_account(api, row)
        api.fresh = snapshot(short_used=0.0032)
        await self.run_account(api, row, NOW+61)
        self.assertEqual(api.generations, [13])
        self.assertTrue(row['cycle_confirmed'])
        self.assertFalse(row['pending_verify'])

    async def test_timeout_uncertain_delivery_verifies_without_resubmission(self):
        api = FakeApi()
        api.result = httpx.ReadTimeout('secret response must not be logged')
        row = {}
        await self.run_account(api, row)
        api.fresh = snapshot(short_used=0.0032)
        await self.run_account(api, row, NOW+61)
        self.assertTrue(row['cycle_confirmed'])
        self.assertEqual(api.generations, [13])

    async def test_timeout_unconfirmed_pauses(self):
        api = FakeApi()
        api.result = httpx.ReadTimeout('timeout')
        row = {}
        await self.run_account(api, row)
        await self.run_account(api, row, NOW+61)
        await self.run_account(api, row, NOW+20000)
        self.assertTrue(row['paused'])
        self.assertEqual(api.generations, [13])

    async def test_crash_guard_never_blindly_resubmits(self):
        api = FakeApi()
        row = {'pending_verify': True, 'guard_until': NOW+18030, 'next_at': NOW}
        await self.run_account(api, row)
        self.assertEqual(api.generations, [])
        self.assertTrue(row['paused'])

    async def test_generation_503_stays_paused(self):
        api = FakeApi()
        api.result = {'http_status': 503, 'ok': False}
        row = {}
        await self.run_account(api, row)
        await self.run_account(api, row, NOW+604800)
        self.assertEqual(api.generations, [13])
        self.assertEqual(row['status'], 'paused_generation_error')

    async def test_unknown_and_stale_quota_do_not_generate(self):
        for data in (snapshot(short_used=None), snapshot(short_end=NOW-100)):
            api = FakeApi(data)
            with self.assertRaises(ProbeError):
                await self.run_account(api, {})
            self.assertEqual(api.generations, [])

    async def test_quota_failures_back_off_then_pause(self):
        row = {}
        for index, delay in enumerate((300, 900, None)):
            quota_failure(row, NOW, ProbeError(), lambda: None, lambda *a, **k: None)
            self.assertEqual(row['failures'], index+1)
            if delay:
                self.assertEqual(row['next_at'], NOW+delay)
        self.assertTrue(row['paused'])

    async def test_http_failure_logs_safe_status_and_operation_only(self):
        request = httpx.Request('POST', 'http://private:8787/admin/api/credentials/9/quota-probe?token=PRIVATE',
                                headers={'Authorization': 'Bearer PRIVATE'})
        response = httpx.Response(403, request=request, text='PRIVATE response body')
        error = httpx.HTTPStatusError('PRIVATE exception message', request=request, response=response)
        fields = []
        row = {'credential_id': 9, 'guard_until': NOW + 18030, 'triggers': 3}
        quota_failure(row, NOW, error, lambda: None, lambda event, **data: fields.append((event, data)))
        event, data = fields[0]
        self.assertEqual(event, 'quota_failed')
        self.assertEqual((data['http_status'], data['method'], data['operation']), (403, 'POST', 'quota_probe'))
        self.assertNotIn('PRIVATE', json.dumps(data))
        self.assertNotIn('private:8787', json.dumps(data))
        self.assertEqual((row['guard_until'], row['triggers']), (NOW + 18030, 3))

    async def test_disable_or_stop_during_probe_prevents_generation(self):
        api = FakeApi()
        await self.run_account(api, {}, may_generate=lambda: False)
        self.assertEqual(api.probes, [13])
        self.assertEqual(api.generations, [])

    async def test_cancel_after_guard_preserves_indeterminate_request(self):
        api = FakeApi()
        entered = asyncio.Event()
        async def generate(cid):
            entered.set()
            await asyncio.Event().wait()
        api.generate = generate
        row = {}
        task = asyncio.create_task(self.run_account(api, row))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(row['pending_verify'])
        self.assertEqual(row['guard_until'], NOW+18030)


class NativeFeatureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.flags = {'enabled': True, 'check_interval_sec': 60}
        self.logs = []
        self.api = FakeApi()
        self.keeper = QuotaKeeper(
            lookup=lambda root, key: self.flags.get(key), state_dir=self.tmp.name,
            base_url='http://unused', credentials_file='unused', log=self.logs.append,
            client_factory=lambda: self.api, clock=lambda: NOW,
        )

    async def test_disabled_has_no_network_or_state_writes(self):
        self.flags['enabled'] = False
        await self.keeper.start()
        await asyncio.sleep(0.01)
        await self.keeper.sweep()
        await self.keeper.stop()
        self.assertEqual(self.api.api_calls, [])
        self.assertFalse(self.keeper.store.path.exists())

    async def test_native_sweep_uses_same_policy_and_state(self):
        await self.keeper.sweep()
        row = next(iter(self.keeper.store.load()['accounts'].values()))
        self.assertEqual(row['triggers'], 1)
        self.assertTrue(row['pending_verify'])
        await self.keeper.sweep()
        self.assertEqual(self.api.generations, [13])

    async def test_legacy_state_is_preserved_on_first_sweep(self):
        legacy = {'accounts': {'private@example.test': {
            'credential_ids': [13], 'credential_id': 13, 'active': True,
            'next_at': NOW+1000, 'guard_until': NOW+1000, 'paused': True,
            'triggers': 9, 'status': 'paused_generation_error',
        }}, 'last_run_at': NOW-100}
        self.keeper.store.save(legacy)
        await self.keeper.sweep()
        self.assertEqual(self.keeper.store.load()['accounts'], legacy['accounts'])
        self.assertEqual(self.api.generations, [])

    async def test_second_process_lock_blocks_all_network(self):
        other = StateStore(self.tmp.name)
        with other.locked() as acquired:
            self.assertTrue(acquired)
            await self.keeper.sweep()
        self.assertEqual(self.api.api_calls, [])
        self.assertEqual(self.keeper.status()['status'], 'standby')

    async def test_corrupt_state_fails_closed_without_network(self):
        self.keeper.store.path.write_text('{broken')
        with self.assertRaises(StateError):
            await self.keeper.sweep()
        self.assertEqual(self.api.api_calls, [])
        self.assertEqual(self.keeper.status()['status'], 'degraded')
        self.assertEqual(self.keeper.store.path.read_text(), '{broken')

    async def test_status_is_read_only_and_sanitized(self):
        await self.keeper.sweep()
        before = self.keeper.store.path.read_bytes()
        calls = len(self.api.api_calls), len(self.api.probes), len(self.api.generations)
        result = self.keeper.status()
        self.assertNotIn('private@example.test', json.dumps(result))
        self.assertEqual(result['unique_active_accounts'], 1)
        self.assertEqual(self.keeper.store.path.read_bytes(), before)
        self.assertEqual((len(self.api.api_calls), len(self.api.probes), len(self.api.generations)), calls)

    async def test_resume_preserves_guard_and_sends_no_request(self):
        self.keeper.store.save({'accounts': {'x': {'credential_ids': [13], 'paused': True,
                                'guard_until': NOW+100, 'triggers': 3}}})
        self.assertEqual(await self.keeper.resume(13), 'scheduled')
        row = self.keeper.store.load()['accounts']['x']
        self.assertFalse(row['paused'])
        self.assertEqual(row['guard_until'], NOW+100)
        self.assertEqual(row['triggers'], 3)
        self.assertEqual(self.api.api_calls, [])
        self.assertEqual(await self.keeper.resume(99), 'not_found')

    async def test_async_request_does_not_block_status_or_loop(self):
        started = asyncio.Event()
        release = asyncio.Event()
        async def generate(cid):
            started.set()
            await release.wait()
            return self.api.result
        self.api.generate = generate
        task = asyncio.create_task(self.keeper.sweep())
        await started.wait()
        status = self.keeper.status()
        self.assertTrue(status['in_sweep'])
        self.assertEqual(await self.keeper.resume(13), 'busy')
        release.set()
        await task

    async def test_lifecycle_is_singleton_and_gracefully_stops(self):
        self.api.cached_value = snapshot(short_used=20)
        await self.keeper.start()
        task = self.keeper._task
        await self.keeper.start()
        self.assertIs(self.keeper._task, task)
        await asyncio.sleep(0.03)
        await self.keeper.stop()
        self.assertTrue(task.done())
        self.assertEqual(self.api.generations, [])

    async def test_lifecycle_error_is_isolated_and_redacted(self):
        self.keeper.client_factory = lambda: (_ for _ in ()).throw(RuntimeError('token=SECRET'))
        await self.keeper.start()
        await asyncio.sleep(0.03)
        status = self.keeper.status()
        await self.keeper.stop()
        self.assertEqual(status['status'], 'degraded')
        self.assertEqual(status['last_error'], 'RuntimeError')
        self.assertNotIn('SECRET', ''.join(self.logs))

    async def test_shutdown_timeout_preserves_guard_and_releases_file_lock(self):
        started = asyncio.Event()
        async def generate(cid):
            started.set()
            await asyncio.Event().wait()
        self.api.generate = generate
        self.keeper.shutdown_timeout = 0.01
        await self.keeper.start()
        await started.wait()
        await self.keeper.stop()
        row = next(iter(self.keeper.store.load()['accounts'].values()))
        self.assertTrue(row['pending_verify'])
        self.assertEqual(row['guard_until'], NOW+18030)
        with self.keeper.store.locked() as acquired:
            self.assertTrue(acquired)

    async def test_state_file_private_and_invalid_deadlines_rejected(self):
        self.keeper.store.save({'accounts': {}})
        self.assertEqual(self.keeper.store.path.stat().st_mode & 0o777, 0o600)
        self.keeper.store.path.write_text('{"accounts":{"x":{"guard_until":"bad"}}}')
        with self.assertRaises(StateError):
            self.keeper.store.load()

    async def test_interval_cannot_be_accelerated(self):
        for value in (0, -5, 0.01, 'bad'):
            self.flags['check_interval_sec'] = value
            self.assertEqual(self.keeper.interval(), 60)

    async def test_authenticated_routes_and_health_have_no_upstream_side_effects(self):
        token_file = Path(self.tmp.name, 'admin.token')
        token_file.write_text('x'*48)
        app = FastAPI()
        register_routes(app, keeper=self.keeper, admin_token_file=str(token_file))
        @app.get('/health')
        async def health():
            return {'status': 'ok'}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            for headers in ({}, {'Authorization': 'Bearer wrong'}):
                self.assertEqual((await client.get('/admin/quota-keeper', headers=headers)).status_code, 401)
            result = await client.get('/admin/quota-keeper', headers={'Authorization': 'Bearer '+'x'*48})
            self.assertEqual(result.status_code, 200)
            self.assertEqual((await client.get('/health')).status_code, 200)
            self.assertEqual((await client.post('/admin/quota-keeper/credentials/13/resume')).status_code, 401)
        self.assertEqual(self.api.api_calls, [])
        self.assertFalse(self.keeper.store.path.exists())


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_cookie_management_posts_carry_exact_origin_and_gets_do_not(self):
        cases = ['http://gproxy:8787', 'http://gproxy:80/prefix',
                 'https://GPROXY:443/prefix', 'http://[::1]:8787/prefix']
        for base in cases:
            with self.subTest(base=base), tempfile.TemporaryDirectory() as directory:
                credentials = Path(directory, 'fake.env')
                credentials.write_text('GPROXY_ADMIN_USER=fake\nGPROXY_ADMIN_PASSWORD=fake-password\n')
                requests = []
                def handler(request):
                    requests.append(request)
                    expected = request.url.scheme + '://' + request.headers['host']
                    if request.method == 'POST':
                        self.assertEqual(request.headers.get('origin'), expected)
                    else:
                        self.assertNotIn('origin', request.headers)
                    if request.url.path.endswith('/login'):
                        return httpx.Response(200, json={'ok': True}, headers={'Set-Cookie': 'gproxy_session=fake; Path=/'})
                    self.assertIn('gproxy_session=fake', request.headers.get('cookie', ''))
                    if request.url.path.endswith('/quota-probe'):
                        self.assertEqual(request.url.query, b'force=true')
                        return httpx.Response(200, json={'raw': ['synthetic-evidence'], 'snapshot': snapshot()})
                    if request.url.path.endswith('/reveal'):
                        return httpx.Response(200, json={'secret': {'project_id': 'fake'}})
                    return httpx.Response(200, json=snapshot())
                session = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
                with patch('aetherstream.features.quota_keeper.client.httpx.AsyncClient', return_value=session):
                    async with GproxyClient(base, str(credentials)) as api:
                        await api.cached(13)
                        with patch('aetherstream.features.quota_keeper.client.time.time', return_value=NOW):
                            await api.probe(13)
                        await api.api('credentials/13/reveal', {})
                self.assertEqual(len(requests), 4)
                self.assertEqual([r.method for r in requests], ['POST', 'GET', 'POST', 'POST'])
                self.assertTrue(session.is_closed)
                self.assertTrue(all(r.url.host == httpx.URL(base).host for r in requests))

    async def test_invalid_management_base_rejected(self):
        for base in ('file:///private', '/missing-host', 'ftp://gproxy'):
            with self.subTest(base=base), self.assertRaises(ValueError):
                GproxyClient(base, 'unused')

    async def test_stale_probe_rejected(self):
        api = GproxyClient('http://test', 'unused')
        api.api = AsyncMock(return_value={'raw': '', 'snapshot': snapshot()})
        with self.assertRaises(ProbeError):
            await api.probe(13)

    async def test_generation_fixed_model_no_fallback_or_secret_persistence(self):
        api = GproxyClient('http://test', 'unused')
        api.api = AsyncMock(return_value={'secret': {'project_id': 'project', 'access_token': 'PRIVATE'}})
        session = AsyncMock()
        session.post.return_value = httpx.Response(200, json={'response': {
            'modelVersion': MODEL, 'candidates': [{'content': {'parts': [{'text': 'OK'}]}}],
            'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 4, 'totalTokenCount': 16}}})
        with patch('aetherstream.features.quota_keeper.client.httpx.AsyncClient') as factory:
            factory.return_value.__aenter__.return_value = session
            result = await api.generate(13)
        factory.assert_called_once()
        self.assertFalse(factory.call_args.kwargs['follow_redirects'])
        self.assertFalse(factory.call_args.kwargs['trust_env'])
        session.post.assert_awaited_once()
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], GENERATION_URL)
        self.assertEqual(kwargs['json']['model'], MODEL)
        self.assertEqual(kwargs['json']['request']['generationConfig']['maxOutputTokens'], 64)
        self.assertTrue(result['ok'])
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertNotIn('reply', result)


class ApplicationWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_lifespan_starts_and_stops_native_worker(self):
        import importlib
        module = importlib.import_module('aetherstream.api.app')
        keeper = module.quota_keeper
        with patch.object(keeper, 'lookup', return_value=False), patch.object(keeper, 'client_factory') as factory:
            async with module.app.router.lifespan_context(module.app):
                await asyncio.sleep(0)
                self.assertTrue(keeper.status()['worker_running'])
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url='http://test') as client:
                    self.assertEqual((await client.get('/health')).status_code, 200)
                    self.assertEqual((await client.get('/admin/quota-keeper')).status_code, 401)
            self.assertFalse(keeper.status()['worker_running'])
            factory.assert_not_called()

    async def test_real_anthropic_dependencies_bind_content_guard(self):
        import importlib
        module = importlib.import_module('aetherstream.api.app')
        with patch.object(module, 'find_stop_tag', side_effect=lambda text: text.find('<disclaimer>')):
            deps = module.build_anthropic_messages_deps({'system': '<content>body</content>'})
            self.assertEqual(deps.find_stop_tag('<content>unfinished<disclaimer>'), -1)
            text = '<content>finished</content><disclaimer>'
            self.assertEqual(deps.find_stop_tag(text), text.index('<disclaimer>'))


if __name__ == '__main__':
    unittest.main()
