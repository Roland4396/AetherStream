"""Explicit per-channel/model modes, private UI API and shared config writes.

All catalogs and route selections are synthetic; no model/GPU/network calls.
"""
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from aetherstream.api.request_mode_routes import register_routes
from aetherstream.routing.policy_config import enrich_selection, dedupe_identity, RegistryError
from aetherstream.routing.request_mode import resolve, set_mode, migrate_flags, ModeConfigError
from aetherstream.runtime.flags import RuntimeFlags, read_flags
from aetherstream.runtime.request_modes import RequestModeStore, ConfigConflict
from aetherstream.streaming.modes import upstream_stream_mode


def flags():
    return {'streaming': {'exact_dedupe': True, 'json_keepalive': True},
            'untouched': {'secret': 'synthetic-secret-do-not-project'},
            'channel_policies': {'schema': 1, 'defaults': {}, 'profiles': {},
                'model_rules': [], 'channels': {
                    'first': {'policy': {'streaming': {'force_upstream': 'stream'}}},
                    'second': {'policy': {}}}}}


def selection(channel='first', model='actual-model'):
    return {'channel': channel, 'route': 'pool-owned-route', 'transport': 'chat',
            'model': 'public-alias', 'upstream_model': model, 'revision': 'pool-revision'}


class ModePolicyTests(unittest.TestCase):
    def test_request_mode_decision_is_visible_in_default_logs(self):
        from aetherstream.observability.logging import ProxyLogger
        logger = object.__new__(ProxyLogger); logger.verbose_trace = False
        self.assertTrue(logger._should_emit('[TRACE test] request_mode client_stream=false upstream_stream=false mode=auto source=channel'))

    def test_profile_migration_is_explicit_and_preserves_unrelated_settings(self):
        value = flags()
        value['streaming'].update(force_upstream='auto', nonstream_to_stream=False)
        value['claude'] = {'stream': {'nonstream_to_stream': False}, 'other': 42}
        value['channel_policies']['profiles'] = {'old': {'streaming': {'force_upstream': 'stream'}, 'injections': {'opus_note': True}}}
        value['channel_policies']['channels'] = {'subscription': {'routes': {'responses': {'profiles': ['old']}}}}
        expanded, report = migrate_flags(value, keep_inert_legacy=True)
        contracted, _ = migrate_flags(expanded)
        self.assertEqual(report['channel_modes'], {'subscription': 'stream'})
        self.assertEqual(contracted['channel_policies']['channels']['subscription']['policy']['streaming']['force_upstream'], 'stream')
        self.assertNotIn('force_upstream', contracted['channel_policies']['profiles']['old'].get('streaming', {}))
        self.assertNotIn('nonstream_to_stream', contracted['streaming'])
        self.assertNotIn('force_upstream', contracted['streaming'])
        self.assertEqual(contracted['claude'], {'stream': {}, 'other': 42})
        self.assertEqual(contracted['untouched'], value['untouched'])
        self.assertEqual(contracted['channel_policies']['profiles']['old']['injections'], {'opus_note': True})

    def test_migration_refuses_conflicting_routes_instead_of_inventing_a_default(self):
        value = flags()
        value['channel_policies']['channels']['first']['routes'] = {
            'a': {'policy': {'streaming': {'force_upstream': 'json'}}}, 'b': {}}
        with self.assertRaises(ModeConfigError): migrate_flags(value)

    def test_only_one_selector_and_follow_is_the_default(self):
        for downstream in (True, False):
            self.assertEqual(upstream_stream_mode({}, downstream), downstream)
            # The removed conversion boolean cannot affect execution.
            self.assertEqual(upstream_stream_mode({'streaming': {'nonstream_to_stream': True}}, downstream), downstream)
            for mode in ('auto', 'stream', 'json'):
                expected = downstream if mode == 'auto' else mode == 'stream'
                self.assertEqual(upstream_stream_mode({'streaming': {'force_upstream': mode}}, downstream), expected)
        with self.assertRaises(ValueError):
            upstream_stream_mode({'streaming': {'force_upstream': 'surprise'}}, False)

    def test_model_follow_replaces_channel_force_and_never_leaks_to_another_channel(self):
        value = set_mode(flags(), channel='first', model='actual-model', mode='auto')
        plan = enrich_selection(lambda key: value.get(key), selection())
        self.assertEqual(plan['request_mode']['source'], 'model')
        self.assertEqual(plan['policy']['streaming']['force_upstream'], 'auto')
        self.assertNotIn('nonstream_to_stream', plan['policy']['streaming'])
        other = enrich_selection(lambda key: value.get(key), selection('second'))
        self.assertEqual(other['request_mode']['source'], 'default')
        different = enrich_selection(lambda key: value.get(key), selection(model='different'))
        self.assertEqual(different['request_mode']['source'], 'channel')
        self.assertEqual(different['request_mode']['mode'], 'stream')
        value = set_mode(value, channel='first', model='actual-model', mode='inherit')
        self.assertEqual(resolve(value['channel_policies'], 'first', 'actual-model')['mode'], 'stream')

    def test_unconfigured_new_channel_and_model_follow_the_caller(self):
        value = flags()
        plan = enrich_selection(lambda key: value.get(key), selection('brand-new-channel', 'new-model'))
        self.assertEqual(plan['request_mode']['source'], 'default')
        self.assertEqual(plan['request_mode']['mode'], 'auto')

    def test_hidden_global_profile_route_or_unscoped_model_conversion_is_rejected(self):
        variants = []
        value = flags(); value['streaming']['nonstream_to_stream'] = True; variants.append(value)
        value = flags(); value['streaming']['force_upstream'] = 'json'; variants.append(value)
        value = flags(); value['channel_policies']['profiles']['hidden'] = {'streaming': {'force_upstream': 'stream'}}; variants.append(value)
        value = flags(); value['channel_policies']['defaults']['streaming'] = {'force_upstream': 'stream'}; variants.append(value)
        value = flags(); value['channel_policies']['channels']['first']['routes'] = {'route': {'policy': {'streaming': {'force_upstream': 'json'}}}}; variants.append(value)
        value = flags(); value['channel_policies']['model_rules'] = [{'match': {'regex': 'gemini'}, 'policy': {'streaming': {'force_upstream': 'stream'}}}]; variants.append(value)
        value = flags(); value['channel_policies']['model_rules'] = [{'match': {'exact': ['a']}, 'channels': ['first', 'second'], 'policy': {'streaming': {'force_upstream': 'stream'}}}]; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(RegistryError):
                enrich_selection(lambda key: value.get(key), selection())

    def test_duplicate_model_modes_are_rejected_instead_of_last_rule_wins(self):
        value = set_mode(flags(), channel='first', model='actual-model', mode='json')
        value['channel_policies']['model_rules'] *= 2
        with self.assertRaises(RegistryError):
            enrich_selection(lambda key: value.get(key), selection())

    def test_changing_one_model_preserves_other_models_and_other_features(self):
        value = flags()
        value['channel_policies']['model_rules'] = [{'match': {'exact': ['a', 'b']}, 'channels': ['first'],
            'policy': {'streaming': {'force_upstream': 'stream'}, 'injections': {'opus_note': True}}}]
        updated = set_mode(value, channel='first', model='a', mode='json')
        self.assertEqual(resolve(updated['channel_policies'], 'first', 'b')['mode'], 'stream')
        self.assertEqual(resolve(updated['channel_policies'], 'first', 'a')['mode'], 'json')
        removed = set_mode(updated, channel='first', model='a', mode='inherit')
        self.assertEqual(resolve(removed['channel_policies'], 'first', 'a')['source'], 'channel')
        for model in ('a', 'b'):
            plan = enrich_selection(lambda key: removed.get(key), selection(model=model))
            self.assertTrue(plan['policy']['injections']['opus_note'])
        self.assertEqual(removed['untouched'], value['untouched'])

    def test_generic_shared_identity_survives_the_rollout(self):
        value = flags(); plan = enrich_selection(lambda key: value.get(key), selection())
        payload = {'model': 'actual-model', 'stream': True, 'messages': []}
        old_policy = copy.deepcopy(plan['policy'])
        old_policy['streaming']['nonstream_to_stream'] = False
        identity = {'scope': 'stream-channel-json-v1', 'route': plan['route'],
                    'upstream_model': plan['upstream_model'], 'policy': old_policy, 'request': payload}
        old_key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(dedupe_identity(plan, payload, {})['key'], old_key)

    def test_legacy_claude_flag_cleanup_does_not_invalidate_shared_request_identity(self):
        from aetherstream.routing.request_mode import compatible_dedupe_policy
        value = flags()
        value['claude'] = {'stream': {'header_keepalive': {'enabled': True}, 'nonstream_to_stream': False}}
        migrated, _ = migrate_flags(value)
        old_plan = enrich_selection(lambda key: value.get(key), selection())
        new_plan = enrich_selection(lambda key: migrated.get(key), selection())
        self.assertNotIn('nonstream_to_stream', new_plan['policy']['runtime']['claude']['stream'])
        self.assertEqual(compatible_dedupe_policy(old_plan['policy']), compatible_dedupe_policy(new_plan['policy']))


def writer_process(path, backups, revision, channel, queue, start):
    store = RequestModeStore(RuntimeFlags(path=path, poll_sec=0, log=lambda _: None), backup_dir=backups)
    start.wait(5)
    try:
        store.update(revision=revision, channel=channel, model=None, mode='json')
        queue.put('saved')
    except ConfigConflict:
        queue.put('conflict')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'flags.json'
        self.path.write_text(json.dumps(flags()))
        self.runtime = RuntimeFlags(path=str(self.path), poll_sec=0, log=lambda _: None)
        self.store = RequestModeStore(self.runtime, backup_dir=str(Path(self.temp.name) / 'backups'))

    def test_cas_in_place_write_backup_and_next_snapshot(self):
        original, revision = self.store.snapshot(); inode = self.path.stat().st_ino
        old_plan = enrich_selection(self.runtime.lookup, selection())
        new_revision = self.store.update(revision=revision, channel='first', model='actual-model', mode='json')
        self.assertEqual(self.path.stat().st_ino, inode)
        updated, actual_revision = self.store.snapshot(); self.assertEqual(new_revision, actual_revision)
        self.assertEqual(updated['untouched'], original['untouched'])
        self.assertEqual(json.loads((self.store.backup_dir / (revision + '.json')).read_text()), original)
        self.assertEqual((self.store.backup_dir / (revision + '.json')).stat().st_mode & 0o777, 0o600)
        new_plan = enrich_selection(self.runtime.lookup, selection())
        self.assertEqual(old_plan['request_mode']['mode'], 'stream')
        self.assertEqual(new_plan['request_mode']['mode'], 'json')
        with self.assertRaises(ConfigConflict):
            self.store.update(revision=revision, channel='first', model=None, mode='auto')
        self.assertEqual(self.store.snapshot()[1], new_revision)

    def test_failed_partial_write_restores_original(self):
        original = self.path.read_bytes(); _, revision = self.store.snapshot()
        write = self.store._write; calls = []
        def fail_first(fd, raw):
            calls.append(1)
            if len(calls) == 1:
                os.pwrite(fd, b'{partial', 0)
                raise OSError('synthetic disk failure')
            write(fd, raw)
        with patch.object(self.store, '_write', side_effect=fail_first), self.assertRaises(OSError):
            self.store.update(revision=revision, channel='first', model=None, mode='json')
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(read_flags(str(self.path))[0], flags())

    def test_cross_process_writers_do_not_overwrite_each_other(self):
        _, revision = self.store.snapshot(); ctx = multiprocessing.get_context('fork')
        queue, start = ctx.Queue(), ctx.Event()
        children = [ctx.Process(target=writer_process, args=(str(self.path), str(self.store.backup_dir), revision, channel, queue, start))
                    for channel in ('first', 'second')]
        for process in children: process.start()
        start.set()
        results = [queue.get(timeout=5) for _ in children]
        for process in children:
            process.join(5); self.assertEqual(process.exitcode, 0)
        self.assertEqual(sorted(results), ['conflict', 'saved'])


class PrivateAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'flags.json'; self.path.write_text(json.dumps(flags()))
        self.token = 'synthetic-private-token-' + 'x' * 48
        token_file = Path(self.temp.name) / 'admin.token'; token_file.write_text(self.token)
        self.runtime = RuntimeFlags(path=str(self.path), poll_sec=0, log=lambda _: None)
        self.catalog = AsyncMock(return_value={'models': [{'id': 'public-alias', 'owned_by': 'first'},
                                                        {'id': 'new-model', 'owned_by': 'third'}]})
        self.resolve = AsyncMock(return_value=selection())
        self.app = FastAPI()
        register_routes(self.app, runtime_flags=self.runtime, token_file=str(token_file),
                        backup_dir=str(Path(self.temp.name) / 'backups'), catalog=self.catalog,
                        resolve_route=self.resolve, log=lambda _: None)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://offline.invalid')
        self.addAsyncCleanup(self.client.aclose)
        self.headers = {'Authorization': 'Bearer ' + self.token}

    async def test_shell_has_no_credentials_and_all_data_operations_require_admin_token(self):
        shell = await self.client.get('/admin/request-modes')
        self.assertEqual(shell.status_code, 200)
        self.assertNotIn(self.token, shell.text)
        self.assertNotIn('synthetic-secret-do-not-project', shell.text)
        for method, path in [('GET', 'state'), ('POST', 'state'), ('GET', 'resolve?model=public-alias')]:
            response = await self.client.request(method, '/admin/request-modes/' + path)
            self.assertEqual(response.status_code, 401)
        self.catalog.assert_not_awaited(); self.resolve.assert_not_awaited()

    async def test_safe_projection_preview_and_scoped_update(self):
        response = await self.client.get('/admin/request-modes/state', headers=self.headers)
        state = response.json(); self.assertEqual(response.status_code, 200)
        self.assertNotIn('synthetic-secret-do-not-project', response.text)
        first = next(c for c in state['channels'] if c['id'] == 'first')
        self.assertEqual(first['mode'], 'stream'); self.assertEqual(first['source'], 'channel')
        self.assertIn('third', {c['id'] for c in state['channels']})
        response = await self.client.get('/admin/request-modes/resolve?model=public-alias', headers=self.headers)
        self.assertEqual(response.json()['upstream_model'], 'actual-model')
        self.resolve.assert_awaited_once_with('public-alias')
        response = await self.client.post('/admin/request-modes/state', headers=self.headers,
            json={'revision': state['revision'], 'channel': 'first', 'model': 'actual-model', 'mode': 'auto'})
        self.assertEqual(response.status_code, 200, response.text)
        first = next(c for c in response.json()['channels'] if c['id'] == 'first')
        self.assertEqual(first['mode'], 'stream')
        self.assertEqual(first['models'][0]['mode'], 'auto')
        self.assertEqual(first['models'][0]['source'], 'model')

    async def test_stale_write_returns_conflict_and_never_overwrites_other_changes(self):
        state = (await self.client.get('/admin/request-modes/state', headers=self.headers)).json()
        changed = flags(); changed['unrelated-edit'] = 42; self.path.write_text(json.dumps(changed))
        response = await self.client.post('/admin/request-modes/state', headers=self.headers,
            json={'revision': state['revision'], 'channel': 'first', 'model': None, 'mode': 'json'})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(json.loads(self.path.read_text()), changed)

    async def test_catalog_view_does_not_gate_scoped_settings_or_hide_model_only_channels(self):
        state = (await self.client.get('/admin/request-modes/state', headers=self.headers)).json()
        self.catalog.side_effect = RuntimeError('Synthetic catalog unavailable')
        response = await self.client.post('/admin/request-modes/state', headers=self.headers,
            json={'revision': state['revision'], 'channel': 'dynamic-private-channel',
                  'model': 'actual-private-model', 'mode': 'json'})
        self.assertEqual(response.status_code, 200, response.text)
        channel = next(c for c in response.json()['channels'] if c['id'] == 'dynamic-private-channel')
        self.assertEqual(channel['mode'], 'auto')
        self.assertEqual(channel['models'][0]['mode'], 'json')
        self.assertTrue(response.json()['catalog_error'])
        value = json.loads(self.path.read_text())
        selected = selection('dynamic-private-channel', 'actual-private-model')
        plan = enrich_selection(lambda key: value.get(key), selected)
        self.assertEqual(plan['request_mode']['mode'], 'json')


if __name__ == '__main__':
    unittest.main()
