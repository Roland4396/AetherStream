import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI
from aetherstream.api.quota_keeper_routes import register_routes
from aetherstream.features.quota_keeper import argon
from aetherstream.features.quota_keeper.client import GproxyClient, ProbeError, native_id
from aetherstream.features.quota_keeper.service import QuotaKeeper
from aetherstream.features.quota_keeper.state import StateStore, StateError

NOW = 2_000_000_000
MODEL = 'gemini-4-argon-low'
MEMBERS = [{'id': 31, 'provider_id': 4, 'enabled': True, 'label': 'fake@example.test'}]


def reading(remaining=1, end=NOW + argon.WEEK, available=True):
    return {'available': available, 'models': [MODEL], 'remaining': remaining,
            'end': end, 'observed_at_ms': NOW * 1000}


def diagnostic(remaining=1, model_disabled=False, bucket=True):
    return {'error': None, 'snapshot': {'observedAtMs': NOW * 1000}, 'responses': [
        {'status': 200, 'body': {'models': {MODEL: {'disabled': model_disabled}},
                               'modelExperiments': {'untrusted': 'PRIVATE'}}},
        {'status': 200, 'body': {'groups': [{'buckets': [
            {'bucketId': 'gemini-weekly', 'remainingFraction': 1,
             'resetTime': datetime.fromtimestamp(NOW + argon.WEEK, timezone.utc).isoformat()},
            *([{'bucketId': argon.BUCKET, 'remainingFraction': remaining,
                'resetTime': datetime.fromtimestamp(NOW + argon.WEEK, timezone.utc).isoformat()}]
              if bucket else [])]}]}}]}


class FakeApi:
    def __init__(self, value=None):
        self.value = reading() if value is None else value
        self.probes = []
        self.generations = []
        self.result = {'http_status': 200, 'ok': True,
                       'usage': {'inputTokens': 1, 'outputTokens': 2},
                       'reply': 'PRIVATE', 'credential_label': 'PRIVATE'}

    async def argon_probe(self, cid):
        self.probes.append(cid)
        if isinstance(self.value, BaseException):
            raise self.value
        return copy.deepcopy(self.value)

    async def generate_argon(self, member, model):
        self.generations.append((member['id'], member['provider_id'], model))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class WeeklyTests(unittest.IsolatedAsyncioTestCase):
    async def run_account(self, api, row, now=NOW, persist=lambda: None,
                          may_generate=lambda: True):
        await argon.handle_argon_account(api, MEMBERS, row, persist,
                                        lambda *a, **k: None, model=MODEL,
                                        clock=lambda: now, may_generate=may_generate)

    async def test_existing_use_and_exhaustion_only_wait(self):
        for remaining in [0, 0.1, 0.99999999]:
            row = {}; api = FakeApi(reading(remaining))
            await self.run_account(api, row)
            self.assertEqual(row['status'], 'waiting')
            self.assertEqual(row['next_at'], NOW + argon.WEEK + 30)
            self.assertFalse(api.generations)

    async def test_unused_account_persists_guard_before_exactly_one_post(self):
        api = FakeApi(); row = {}; snapshots = []
        original = api.generate_argon
        async def generate(*args):
            self.assertTrue(snapshots[-1]['uncertain_attempt'])
            self.assertTrue(snapshots[-1]['pending_verify'])
            self.assertEqual(snapshots[-1]['triggers'], 1)
            return await original(*args)
        api.generate_argon = generate
        await self.run_account(api, row, persist=lambda: snapshots.append(copy.deepcopy(row)))
        await self.run_account(api, row)
        self.assertEqual(api.generations, [(31, 4, MODEL)])
        self.assertNotIn('PRIVATE', json.dumps(row))

    async def test_first_verification_confirms_then_next_week_may_start(self):
        row = {}; api = FakeApi()
        await self.run_account(api, row)
        api.value = reading(0.99999)
        api.value['observed_at_ms'] = (NOW+61)*1000
        await self.run_account(api, row, NOW + 61)
        self.assertTrue(row['cycle_confirmed'])
        self.assertFalse(row['uncertain_attempt'])
        await self.run_account(api, row, NOW + 3600)
        self.assertEqual(len(api.generations), 1)
        api.value = reading(1, NOW + 2 * argon.WEEK + 31)
        await self.run_account(api, row, NOW + argon.WEEK + 31)
        self.assertEqual(len(api.generations), 2)
        self.assertEqual(row['triggers'], 2)

    async def test_normal_use_seen_by_fresh_probe_skips_post(self):
        api = FakeApi(reading(.5)); row = {'next_at': NOW - 1}
        await self.run_account(api, row)
        self.assertFalse(api.generations)

    async def test_no_permission_is_daily_metadata_only(self):
        row = {}; api = FakeApi(reading(available=False))
        await self.run_account(api, row)
        await self.run_account(api, row, NOW + 3600)
        self.assertEqual(row['status'], 'unavailable')
        self.assertEqual(len(api.probes), 1)
        self.assertFalse(api.generations)

    async def test_wrong_variant_never_falls_back_to_another_model(self):
        row = {}; api = FakeApi(); api.value['models'] = ['gemini-4-argon-high']
        await self.run_account(api, row)
        self.assertFalse(api.generations)
        self.assertEqual(row['status'], 'unavailable')

    async def test_timeout_does_not_retry_after_restart_week_or_resume(self):
        row = {}; api = FakeApi(); api.result = httpx.ReadTimeout('PRIVATE')
        await self.run_account(api, row)
        self.assertEqual(row['status'], 'delivery_uncertain')
        await self.run_account(api, row, NOW + 61)
        self.assertTrue(row['paused'])
        await self.run_account(api, row, NOW + 2 * argon.WEEK)
        row.update(paused=False, next_at=0)  # explicit recheck, not a paid retry
        api.value = reading(1, NOW + 3 * argon.WEEK)
        await self.run_account(api, row, NOW + 2 * argon.WEEK)
        self.assertTrue(row['paused'])
        self.assertEqual(len(api.generations), 1)
        self.assertNotIn('PRIVATE', json.dumps(row))

    async def test_cancel_and_bare_guard_survive_expiry_without_resubmission(self):
        row = {}; api = FakeApi(); api.result = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.run_account(api, row)
        self.assertTrue(row['pending_verify'])
        await self.run_account(api, row, NOW + 61)
        self.assertTrue(row['paused'])
        bare = {'last_trigger_at': NOW - 2 * argon.WEEK, 'guard_until': NOW - 1}
        await self.run_account(FakeApi(), bare)
        self.assertTrue(bare['paused'])

    async def test_503_verifies_once_without_reposting(self):
        api = FakeApi(); api.result = {'ok': False, 'http_status': 503}; row = {}
        await self.run_account(api, row)
        await self.run_account(api, row, NOW + 61)
        self.assertEqual(len(api.generations), 1)
        self.assertTrue(row['paused'])

    async def test_successful_http_without_quota_change_pauses(self):
        api = FakeApi(); row = {}
        await self.run_account(api, row)
        await self.run_account(api, row, NOW + 61)
        self.assertTrue(row['paused'])
        self.assertFalse(row['cycle_confirmed'])

    async def test_current_use_can_confirm_an_uncertain_request(self):
        row = {'uncertain_attempt': True}; api = FakeApi(reading(.8, NOW + argon.WEEK - 60))
        await self.run_account(api, row)
        api.value['observed_at_ms'] = (NOW+61)*1000
        await self.run_account(api, row, NOW+61)
        self.assertTrue(row['cycle_confirmed'])
        self.assertFalse(api.generations)

    async def test_disabled_or_stopping_during_probe_prevents_post(self):
        row = {}; api = FakeApi()
        await self.run_account(api, row, may_generate=lambda: False)
        self.assertFalse(api.generations)
        self.assertNotIn('last_trigger_at', row)

    async def test_missing_nonfinite_stale_or_invalid_quota_never_posts(self):
        for value in [reading(float('nan')), reading(True), reading(-1), reading(1.01),
                      reading(1, NOW - 31), reading(1, NOW + 2 * argon.WEEK)]:
            with self.subTest(value=value):
                api = FakeApi(value)
                with self.assertRaises(ProbeError):
                    await self.run_account(api, {})
                self.assertFalse(api.generations)

    async def test_probe_errors_back_off_then_pause_only_weekly_state(self):
        weekly = {'pending_verify': True, 'uncertain_attempt': True}; writes = []
        for count in range(3):
            argon.quota_failure(weekly, NOW, ProbeError('PRIVATE'),
                                lambda: writes.append(1), lambda *a, **k: None)
        self.assertTrue(weekly['paused'])
        self.assertTrue(weekly['uncertain_attempt'])
        self.assertNotIn('PRIVATE', json.dumps(weekly))


class ParserTests(unittest.IsolatedAsyncioTestCase):
    async def probe(self, data):
        api = GproxyClient('http://mock.invalid:8787', 'unused')
        api.api = AsyncMock(return_value=data)
        with patch('aetherstream.features.quota_keeper.client.time.time', return_value=NOW):
            result = await api.argon_probe(31)
        api.api.assert_awaited_once_with('credentials/v3-credentials-31/quota-diagnostics', {})
        return result

    async def test_only_argon_bucket_and_enabled_models_are_retained(self):
        r = await self.probe(diagnostic(.8))
        self.assertTrue(r['available']); self.assertEqual(r['remaining'], .8)
        self.assertEqual(r['models'], [MODEL]); self.assertNotIn('PRIVATE', json.dumps(r))

    async def test_disabled_accounts_do_not_borrow_normal_gemini_quota(self):
        r = await self.probe(diagnostic(model_disabled=True, bucket=False))
        self.assertFalse(r['available'])

    async def test_missing_independent_bucket_does_not_mean_full(self):
        with self.assertRaises(ProbeError):
            await self.probe(diagnostic(bucket=False))

    async def test_invalid_reads_fail_closed(self):
        cases = [None, [], {'snapshot': []}, {'snapshot': {'observedAtMs': float('nan')}},
                 {'snapshot': {'observedAtMs': NOW * 1000 + 10000}},
                 {'snapshot': {'observedAtMs': NOW * 1000}, 'responses': {}},
                 {'snapshot': {'observedAtMs': NOW * 1000}, 'responses': [None]}]
        for val in [None, True, 'NaN', '-.1', 2]:
            cases.append(diagnostic(val))
        d = diagnostic(); d['error'] = 'PRIVATE'; cases.append(d)
        d = diagnostic(); d['snapshot']['observedAtMs'] -= 3000; cases.append(d)
        d = diagnostic(); d['responses'][0]['status'] = 403; cases.append(d)
        d = diagnostic(); bs = d['responses'][1]['body']['groups'][0]['buckets']; bs.append(copy.deepcopy(bs[-1])); cases.append(d)
        d = diagnostic(); d['responses'][1]['body']['groups'][0]['buckets'][-1]['resetTime'] = '2033-01-01'; cases.append(d)
        for d in cases:
            with self.subTest(data=d), self.assertRaises(ProbeError):
                await self.probe(d)

    async def test_generation_uses_exact_gproxy_account_no_secret_or_story_router(self):
        api = GproxyClient('http://mock.invalid', 'unused')
        api.api = AsyncMock(return_value={'ok': True, 'status': 200,
            'reply': 'PRIVATE', 'credentialLabel': 'PRIVATE', 'error': 'PRIVATE',
            'usage': {'inputTokens': 1, 'outputTokens': 2, 'reasoningTokens': 0, 'PRIVATE': 'secret'}})
        r = await api.generate_argon(MEMBERS[0], MODEL)
        api.api.assert_awaited_once_with('models/test', {
            'providerId': 'v3-providers-4', 'credentialId': 'v3-credentials-31', 'model': MODEL})
        self.assertNotIn('PRIVATE', json.dumps(r))
        with self.assertRaises(ValueError):
            await api.generate_argon(MEMBERS[0], 'claude-opus')

    async def test_real_management_transport_sends_origin_cookie_and_no_redirects(self):
        with tempfile.TemporaryDirectory() as td:
            credentials = Path(td, 'fake.env'); credentials.write_text('GPROXY_ADMIN_USER=fake\nGPROXY_ADMIN_PASSWORD=fake\n')
            requests = []
            def handler(request):
                requests.append(request)
                self.assertEqual(request.headers.get('origin'), 'http://gproxy:8787')
                if request.url.path.endswith('/login'):
                    return httpx.Response(200, json={}, headers={'Set-Cookie': 'session=fake; Path=/'})
                self.assertIn('session=fake', request.headers.get('cookie', ''))
                if request.url.path.endswith('/quota-diagnostics'):
                    return httpx.Response(200, json=diagnostic())
                self.assertTrue(request.url.path.endswith('/models/test'))
                body = json.loads(request.content); self.assertEqual(body['credentialId'], 'v3-credentials-31')
                return httpx.Response(200, json={'status': 200, 'ok': True})
            session = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
            with patch('aetherstream.features.quota_keeper.client.httpx.AsyncClient', return_value=session):
                async with GproxyClient('http://gproxy:8787', str(credentials)) as api:
                    with patch('aetherstream.features.quota_keeper.client.time.time', return_value=NOW):
                        await api.argon_probe(31)
                    await api.generate_argon(MEMBERS[0], MODEL)
            self.assertEqual(len(requests), 3)
            self.assertTrue(session.is_closed)


class SettingsAndStateTests(unittest.TestCase):
    def test_settings_opt_in_and_restricted_model(self):
        self.assertFalse(argon.settings(None)['enabled'])
        self.assertEqual(argon.settings({'enabled': True})['model'], MODEL)
        for value in [True, {'enabled': 1}, {'enabled': 'true'}, {'model': 'gpt-5'},
                      {'enabled': True, 'interval': 1}]:
            with self.subTest(value=value), self.assertRaises(ValueError):argon.settings(value)

    def test_native_identity_and_path_traversal_rejection(self):
        self.assertEqual(native_id('credentials', 31), 'v3-credentials-31')
        self.assertEqual(native_id('credentials', 'v3-credentials-31'), 'v3-credentials-31')
        for bad in [True, None, 0, -1, '0', '../secret', '', 'x/y', 'x?token=private']:
            with self.subTest(value=bad), self.assertRaises(ValueError):native_id('credentials', bad)

    def test_legacy_state_is_additively_preserved_and_bad_weekly_guards_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = StateStore(td); old = {'accounts': {'fake': {'next_at': NOW, 'unknown_legacy_field': 7}}, 'schema_version': 1}
            store.save(old); self.assertEqual(store.load(), old)
            row = old['accounts']['fake']; row['argon_weekly'] = {'pending_verify': True, 'uncertain_attempt': True, 'guard_until': NOW + argon.WEEK}
            store.save(old); self.assertEqual(store.load(), old)
            for value in [{'guard_until': True}, {'guard_until': -1}, {'pending_verify': 'true'}, []]:
                row['argon_weekly'] = value; store.save(old)
                with self.assertRaises(StateError):store.load()

    def test_status_is_sanitized(self):
        row = {'last_generation': {'ok': True, 'reply': 'PRIVATE', 'usage': {'outputTokens': 2, 'secret': 'PRIVATE'}},
               'window': {'remaining': .5, 'end': NOW, 'secret': 'PRIVATE'}, 'secret': 'PRIVATE'}
        self.assertNotIn('PRIVATE', json.dumps(argon.status(row)))


class FeatureApi(FakeApi):
    def __init__(self):
        super().__init__()
        self.api_calls = []; self.claude_calls = []
        self.members = copy.deepcopy(MEMBERS)

    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def api(self, name):
        self.api_calls.append(name)
        return ([{'id': 4, 'channel': 'antigravity', 'enabled': True}]
                if name == 'providers' else self.members)

    async def cached(self, cid):
        self.claude_calls.append(cid)
        return {'sources': [], 'entries': [
            {'id': name, 'source_id': 'subscription', 'value': {
                'kind': 'window', 'used_percent': 10, 'period_end': getattr(self, 'claude_now', NOW) + duration}}
            for name, duration in [('3p-weekly', argon.WEEK), ('3p-5h', 18000)]]}

    async def probe(self, cid): raise AssertionError('Unexpected Claude probe')
    async def generate(self, cid): raise AssertionError('Unexpected Claude generation')


class FeatureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.flags = {'enabled': True, 'argon_weekly': {'enabled': True}}
        self.api = FeatureApi(); self.logs = []
        self.keeper = QuotaKeeper(lookup=lambda root, key: self.flags.get(key),
            state_dir=self.tmp.name, base_url='http://unused.invalid',
            credentials_file='unused', log=self.logs.append,
            client_factory=lambda: self.api, clock=lambda: NOW)

    def row(self): return next(iter(self.keeper.store.load()['accounts'].values()))

    async def test_default_or_invalid_config_leaves_claude_working_without_argon(self):
        for value in [None, {'enabled': False}, {'enabled': 'true'}, {'model': 'bad'}]:
            self.flags['argon_weekly'] = value
            await self.keeper.sweep()
            self.assertNotIn('argon_weekly', self.row())
        self.assertTrue(self.api.claude_calls)
        self.assertFalse(self.api.probes); self.assertFalse(self.api.generations)

    async def test_claude_pause_does_not_pause_argon_and_duplicates_share_one_post(self):
        self.api.members.append(dict(MEMBERS[0], id=32))
        self.keeper.store.save({'accounts': {'fake@example.test': {
            'paused': True, 'triggers': 9, 'guard_until': NOW + 1000}}})
        await self.keeper.sweep(); await self.keeper.sweep()
        row = self.row()
        self.assertEqual(row['triggers'], 9); self.assertTrue(row['paused'])
        self.assertEqual(self.api.generations, [(31, 4, MODEL)])
        self.assertEqual(row['argon_weekly']['triggers'], 1)

    async def test_argon_failure_does_not_pause_or_reschedule_claude(self):
        self.api.value = ProbeError('PRIVATE')
        await self.keeper.sweep()
        row = self.row()
        self.assertEqual(row['status'], 'waiting')
        self.assertFalse(row.get('paused')); self.assertEqual(row['next_at'], NOW + 18030)
        self.assertEqual(row['argon_weekly']['failures'], 1)
        self.assertNotIn('PRIVATE', ''.join(self.logs))

    async def test_resume_rechecks_only_weekly_and_preserves_uncertainty_across_restart(self):
        await self.keeper.sweep(); before = self.row()
        self.assertEqual(await self.keeper.resume_argon(31), 'scheduled')
        after = self.row()
        self.assertEqual({k: v for k, v in before.items() if k != 'argon_weekly'},
                         {k: v for k, v in after.items() if k != 'argon_weekly'})
        self.assertEqual(after['argon_weekly']['guard_until'], before['argon_weekly']['guard_until'])
        self.assertTrue(after['argon_weekly']['uncertain_attempt'])
        replacement = QuotaKeeper(lookup=self.keeper.lookup, state_dir=self.tmp.name,
            base_url='http://unused.invalid', credentials_file='unused', log=self.logs.append,
            client_factory=lambda: self.api, clock=lambda: NOW + 2 * argon.WEEK)
        self.api.value = reading(1, NOW + 3 * argon.WEEK)
        self.api.claude_now = NOW + 2 * argon.WEEK
        await replacement.sweep()
        self.assertEqual(len(self.api.generations), 1)
        self.assertTrue(self.row()['argon_weekly']['paused'])

    async def test_process_lock_prevents_all_network_even_with_weekly_enabled(self):
        with StateStore(self.tmp.name).locked() as acquired:
            self.assertTrue(acquired); await self.keeper.sweep()
        self.assertFalse(self.api.api_calls); self.assertFalse(self.api.probes)

    async def test_stop_waits_for_paid_post_and_preserves_state_lock_until_completion(self):
        started = asyncio.Event(); released = asyncio.Event()
        async def generate(member, model):
            self.api.generations.append((member['id'], member['provider_id'], model))
            started.set(); await released.wait(); return self.api.result
        self.api.generate_argon = generate
        await self.keeper.start(); await started.wait()
        stop = asyncio.create_task(self.keeper.stop())
        try:
            await asyncio.sleep(.01); self.assertFalse(stop.done())
            with StateStore(self.tmp.name).locked() as acquired: self.assertFalse(acquired)
            self.assertTrue(self.keeper.status()['in_sweep'])
            self.assertEqual(await self.keeper.resume_argon(31), 'busy')
        finally:
            released.set(); await stop
        self.assertTrue(self.row()['argon_weekly']['uncertain_attempt'])
        with StateStore(self.tmp.name).locked() as acquired: self.assertTrue(acquired)
        await self.keeper.sweep(); self.assertEqual(len(self.api.generations), 1)

    async def test_private_status_and_resume_routes_are_network_free(self):
        await self.keeper.sweep()
        app = FastAPI(); token = Path(self.tmp.name, 'token'); token.write_text('x' * 40)
        register_routes(app, keeper=self.keeper, admin_token_file=str(token))
        calls = (len(self.api.api_calls), len(self.api.probes), len(self.api.generations))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://mock.invalid') as c:
            route = '/admin/quota-keeper/credentials/31/argon/resume'
            self.assertEqual((await c.post(route)).status_code, 401)
            h = {'Authorization': 'Bearer ' + 'x' * 40}
            r = await c.get('/admin/quota-keeper', headers=h)
            self.assertTrue(r.json()['argon_weekly']['enabled'])
            self.assertNotIn('fake@example.test', r.text)
            self.assertEqual((await c.post(route, headers=h)).status_code, 200)
            self.assertEqual((await c.post(route.replace('/31/', '/0/'), headers=h)).status_code, 400)
            self.assertEqual((await c.post(route.replace('/31/', '/99/'), headers=h)).status_code, 404)
            async with self.keeper._mutation_lock:
                self.assertEqual((await c.post(route, headers=h)).status_code, 409)
            self.keeper.store.path.write_text('{broken')
            self.assertEqual((await c.post(route, headers=h)).status_code, 503)
        self.assertEqual(calls, (len(self.api.api_calls), len(self.api.probes), len(self.api.generations)))
