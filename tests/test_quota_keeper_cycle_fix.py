"""Offline regressions for false-success, stale names and fixed-cycle verification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

import httpx
from aetherstream.features.quota_keeper import argon, claude
from aetherstream.features.quota_keeper.client import GproxyClient, MODEL, ProbeError
from aetherstream.features.quota_keeper.policy import decision, windows, reset_clock_running
from aetherstream.features.quota_keeper.service import handle_account, QuotaKeeper
from aetherstream.features.quota_keeper.state import StateStore, StateError
from test_quota_keeper import FakeApi, snapshot, diagnostic, MEMBERS, NOW
from test_argon_weekly_keeper import FakeApi as ArgonApi, reading, MEMBERS as ARGON_MEMBERS


class CyclePolicyTests(unittest.TestCase):
    def test_fixed_clock_requires_two_distinct_observations(self):
        end=NOW+18000
        self.assertFalse(reset_clock_running(end,NOW,18000,NOW))
        self.assertFalse(reset_clock_running(end,NOW+61,18000,NOW+61))
        self.assertTrue(reset_clock_running(end,NOW+61,18000,NOW+61,
            previous_end=end,previous_observed=NOW))

    def test_sliding_unused_clock_is_never_confirmed(self):
        self.assertFalse(reset_clock_running(NOW+18061,NOW+61,18000,NOW+61,
            previous_end=NOW+18000,previous_observed=NOW))
        self.assertFalse(reset_clock_running(NOW+18000,NOW,18000,NOW+61,
            previous_end=NOW+18000,previous_observed=NOW))

    def test_stale_cached_placeholder_does_not_become_running_with_age(self):
        q=windows(snapshot())
        self.assertEqual(decision(q,NOW+3600,observed_at=NOW)[0],'unused')

    def test_zero_usage_fixed_clock_waits_after_stability_evidence(self):
        q=windows(snapshot())
        self.assertEqual(decision(q,NOW+61,observed_at=NOW+61,
            previous=q,previous_observed=NOW),('wait',NOW+18030))

    def test_single_shortened_full_window_only_requests_an_observation(self):
        q=windows(snapshot(short_end=NOW+1000))
        self.assertEqual(decision(q,NOW,observed_at=NOW),('observe',NOW+60))

    def test_missing_usage_can_be_confirmed_by_reset_clock(self):
        q=windows(snapshot(short_used=None,week_used=None))
        self.assertEqual(decision(q,NOW+61,observed_at=NOW+61,
            previous=q,previous_observed=NOW),('wait',NOW+18030))

    def test_bad_clock_values_never_confirm(self):
        for value in [None,True,float('nan'),float('inf'),-1]:
            self.assertFalse(reset_clock_running(value,NOW,18000,NOW))
        self.assertFalse(reset_clock_running(NOW+18000,NOW+100,18000,NOW,
            previous_end=NOW+18000,previous_observed=NOW-100))


class ClaudeRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def run_account(self,api,row,now=NOW,**kwargs):
        await handle_account(api,MEMBERS,row,lambda:None,lambda *a,**k:None,
                             clock=lambda:now,**kwargs)

    async def test_retired_model_not_in_fresh_catalogue_never_posts(self):
        api=FakeApi();row={}
        await self.run_account(api,row,model='claude-opus-4-6-thinking')
        self.assertEqual(row['status'],'paused_model_unavailable')
        self.assertTrue(row['paused']);self.assertFalse(api.generations)
        self.assertNotIn('last_trigger_at',row)

    async def test_zero_usage_pending_generation_confirms_from_fixed_reset(self):
        api=FakeApi();row={}
        await self.run_account(api,row)
        api.fresh['sources'][0]['observed_at_ms']=(NOW+61)*1000
        await self.run_account(api,row,NOW+61)
        self.assertTrue(row['cycle_confirmed']);self.assertFalse(row['uncertain_attempt'])
        self.assertEqual(api.generations,[13]);self.assertEqual(row['next_at'],NOW+18030)

    async def test_nonzero_usage_without_fixed_clock_is_not_cycle_success(self):
        api=FakeApi();row={};await self.run_account(api,row)
        for elapsed in (61,122):
            api.fresh=snapshot(short_used=.01,short_end=NOW+elapsed+18000,
                               week_end=NOW+elapsed+604800)
            api.fresh['sources'][0]['observed_at_ms']=(NOW+elapsed)*1000
            await self.run_account(api,row,NOW+elapsed)
        self.assertFalse(row['cycle_confirmed']);self.assertTrue(row['paused'])
        self.assertTrue(row['uncertain_attempt']);self.assertEqual(api.generations,[13])

    async def test_uncertainty_survives_guard_expiry_and_resume(self):
        api=FakeApi();row={'last_trigger_at':NOW-100000,'guard_until':NOW-1,
                          'pending_verify':False,'cycle_confirmed':False}
        await self.run_account(api,row)
        self.assertTrue(row['paused']);self.assertFalse(api.generations)
        row.update(paused=False,next_at=0)
        await self.run_account(api,row,NOW+61)
        self.assertFalse(api.generations);self.assertTrue(row['uncertain_attempt'])

    async def test_single_shortened_full_clock_cannot_cause_a_paid_post(self):
        api=FakeApi(snapshot(short_end=NOW+1000));row={}
        await self.run_account(api,row)
        self.assertEqual(row['status'],'awaiting_cycle_clock');self.assertFalse(api.generations)
        api.fresh['sources'][0]['observed_at_ms']=(NOW+61)*1000
        await self.run_account(api,row,NOW+61)
        self.assertTrue(row['cycle_confirmed']);self.assertFalse(api.generations)

    async def test_http200_without_usage_remains_unverified(self):
        api=FakeApi();api.result={'http_status':200,'ok':True,'reply':'unsupported model','usage':{}}
        row={};events=[]
        await handle_account(api,MEMBERS,row,lambda:None,lambda event,**k:events.append(event),clock=lambda:NOW)
        self.assertFalse(row['last_generation']['ok'])
        self.assertTrue(row['last_generation']['transport_ok'])
        self.assertEqual(row['status'],'response_unverified')
        self.assertNotIn('claude_triggered',events)
        self.assertNotIn('unsupported model',json.dumps(row))
        self.assertTrue(row['pending_verify']);self.assertTrue(row['uncertain_attempt'])

    async def test_disable_model_during_probe_blocks_paid_post(self):
        api=FakeApi();row={}
        await self.run_account(api,row,may_generate=lambda:False)
        self.assertFalse(api.generations)

    async def test_invalid_generation_reply_preserves_delivery_guard(self):
        api=FakeApi();api.result=ProbeError('PRIVATE');row={}
        await self.run_account(api,row)
        self.assertEqual(row['status'],'delivery_uncertain')
        self.assertTrue(row['uncertain_attempt']);self.assertNotIn('PRIVATE',json.dumps(row))


class NativeTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_false_success_and_error_wrapping_never_validate_generation(self):
        cases=[{'ok':True,'status':200,'reply':'This model is unavailable'},
               {'ok':True,'status':200,'model':MODEL,'usage':{'inputTokens':5}},
               {'ok':True,'status':200,'usage':{'outputTokens':16},'error':'PRIVATE'},
               {'ok':True,'status':503,'usage':{'outputTokens':16}},
               {'ok':True,'status':200,'usage':{'outputTokens':True}},
               {'ok':True,'status':200,'usage':{'outputTokens':float('nan')}}]
        for value in cases:
            api=GproxyClient('http://mock.invalid','unused');api.api=AsyncMock(return_value=value)
            result=await api.generate(31,provider_id=4)
            self.assertFalse(result['ok']);self.assertFalse(result['generation_evidence'])
            self.assertNotIn('PRIVATE',json.dumps(result))
            api.api.assert_awaited_once_with('models/test',{'providerId':'v3-providers-4',
                'credentialId':'v3-credentials-31','model':MODEL})

    async def test_genuine_native_usage_is_distinct_from_cycle_confirmation(self):
        api=GproxyClient('http://mock.invalid','unused')
        api.api=AsyncMock(return_value={'ok':True,'status':200,'model':MODEL,
            'reply':'PRIVATE','usage':{'inputTokens':10,'outputTokens':16}})
        with patch('aetherstream.features.quota_keeper.client.httpx.AsyncClient') as direct:
            result=await api.generate(31,provider_id=4,model='claude-opus-5-5-high')
        direct.assert_not_called();self.assertTrue(result['generation_evidence'])
        self.assertNotIn('cycle_confirmed',result);self.assertNotIn('response_model',result)
        self.assertNotIn('PRIVATE',json.dumps(result))
        self.assertEqual(api.api.call_args.args[1]['model'],'claude-opus-5-5-high')

    async def test_native_probe_preserves_claude_and_available_model_evidence(self):
        api=GproxyClient('http://mock.invalid','unused');api.api=AsyncMock(return_value=diagnostic())
        with patch('aetherstream.features.quota_keeper.client.time.time',return_value=NOW):
            result=await api.probe(31)
        self.assertEqual(result['available_models'],[MODEL])
        self.assertEqual(set(windows(result)),{'3p-5h','3p-weekly'})
        api.api.assert_awaited_once_with('credentials/v3-credentials-31/quota-diagnostics',{})

    async def test_missing_duplicate_or_implausible_bucket_never_becomes_full_quota(self):
        for mode in ('missing','duplicate','distant'):
            d=diagnostic();b=d['responses'][1]['body']['groups'][0]['buckets']
            if mode=='missing':b.pop()
            elif mode=='duplicate':b.append(copy.deepcopy(b[0]))
            else:b[0]['resetTime']=datetime.fromtimestamp(NOW+100000,timezone.utc).isoformat()
            api=GproxyClient('http://mock.invalid','unused');api.api=AsyncMock(return_value=d)
            with patch('aetherstream.features.quota_keeper.client.time.time',return_value=NOW):
                with self.assertRaises(ProbeError):await api.probe(31)


class ArgonClockTests(unittest.IsolatedAsyncioTestCase):
    async def run_account(self,api,row,now=NOW):
        await argon.handle_argon_account(api,ARGON_MEMBERS,row,lambda:None,lambda *a,**k:None,
            model='gemini-4-argon-low',clock=lambda:now)

    async def test_full_balance_fixed_weekly_clock_never_sends_keepalive(self):
        api=ArgonApi(reading(1,NOW+10000));row={}
        await self.run_account(api,row)
        self.assertFalse(api.generations);self.assertFalse(row['cycle_confirmed'])
        api.value['observed_at_ms']=(NOW+61)*1000
        await self.run_account(api,row,NOW+61)
        self.assertTrue(row['cycle_confirmed']);self.assertFalse(api.generations)

    async def test_full_balance_pending_weekly_attempt_can_be_confirmed_by_clock(self):
        api=ArgonApi();row={};await self.run_account(api,row)
        api.value['observed_at_ms']=(NOW+61)*1000
        await self.run_account(api,row,NOW+61)
        self.assertTrue(row['cycle_confirmed']);self.assertEqual(len(api.generations),1)


class SettingsStatusTests(unittest.TestCase):
    def test_model_is_configurable_not_a_fallback(self):
        self.assertEqual(claude.settings({'model':'claude-opus-5-5-high'})['model'],'claude-opus-5-5-high')
        for bad in [{'enabled':'true'},{'model':'gemini-4-argon'},{'model':'../secret'},True,{'fallback':MODEL}]:
            with self.assertRaises(ValueError):claude.settings(bad)
        self.assertFalse(claude.settings({'enabled':False})['enabled'])

    def test_legacy_empty_usage_success_is_not_shown_as_verified(self):
        with tempfile.TemporaryDirectory() as td:
            k=QuotaKeeper(lookup=lambda *a:None,state_dir=td,base_url='http://unused.invalid',
                          credentials_file='unused',log=lambda *a:None)
            k.store.save({'accounts':{'fake':{'credential_ids':[31],'last_generation':{
                'http_status':200,'ok':True,'response_model':None,'usage':{}}}}})
            result=k.status()['accounts'][0]['last_generation']
            self.assertFalse(result['ok']);self.assertFalse(result['generation_evidence'])

    def test_bad_uncertainty_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            s=StateStore(td);s.save({'accounts':{'fake':{'uncertain_attempt':'false'}}})
            with self.assertRaises(StateError):s.load()


class ConfigIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_claude_disable_or_invalid_config_does_not_disable_argon(self):
        from test_argon_weekly_keeper import FeatureApi
        for config in [{'enabled':False},{'enabled':'bad'}]:
            with tempfile.TemporaryDirectory() as td:
                flags={'enabled':True,'claude':config,'argon_weekly':{'enabled':True}}
                api=FeatureApi()
                k=QuotaKeeper(lookup=lambda root,key:flags.get(key),state_dir=td,
                    base_url='http://unused.invalid',credentials_file='unused',log=lambda *a:None,
                    client_factory=lambda:api,clock=lambda:NOW)
                await k.sweep()
                self.assertFalse(api.claude_calls)
                self.assertEqual(api.generations,[(31,4,'gemini-4-argon-low')])

    async def test_configured_claude_model_reaches_exact_account_and_changed_config_blocks_post(self):
        with tempfile.TemporaryDirectory() as td:
            flags={'enabled':True,'claude':{'enabled':True,'model':'claude-opus-5-5-high'}}
            api=FakeApi();api.fresh['available_models']=['claude-opus-5-5-high']
            args=[]
            async def generate(cid,**kwargs):args.append((cid,kwargs));return api.result
            api.generate=generate
            k=QuotaKeeper(lookup=lambda root,key:flags.get(key),state_dir=td,
                base_url='http://unused.invalid',credentials_file='unused',log=lambda *a:None,
                client_factory=lambda:api,clock=lambda:NOW)
            await k.sweep()
            self.assertEqual(args,[(13,{'provider_id':4,'model':'claude-opus-5-5-high'})])
            self.assertEqual(k.status()['model'],'claude-opus-5-5-high')
        with tempfile.TemporaryDirectory() as td:
            flags={'enabled':True,'claude':{'enabled':True,'model':MODEL}}
            api=FakeApi();original=api.probe
            async def probe(cid):
                result=await original(cid);flags['claude']['model']='claude-opus-5-5-high';return result
            api.probe=probe
            k=QuotaKeeper(lookup=lambda root,key:flags.get(key),state_dir=td,
                base_url='http://unused.invalid',credentials_file='unused',log=lambda *a:None,
                client_factory=lambda:api,clock=lambda:NOW)
            await k.sweep();self.assertFalse(api.generations)

    async def test_resume_old_pause_rechecks_cycle_without_generation_or_erasing_other_state(self):
        with tempfile.TemporaryDirectory() as td:
            api=FakeApi(snapshot(short_used=.0144,short_end=NOW+1000))
            flags={'enabled':True,'claude':{'enabled':True,'model':MODEL}}
            k=QuotaKeeper(lookup=lambda root,key:flags.get(key),state_dir=td,
                base_url='http://unused.invalid',credentials_file='unused',log=lambda *a:None,
                client_factory=lambda:api,clock=lambda:NOW)
            old={'credential_ids':[13],'paused':True,'status':'paused_unconfirmed_generation',
                 'last_trigger_at':NOW-100000,'guard_until':NOW-50000,'triggers':9,
                 'cycle_confirmed':False,'pending_verify':False,
                 'windows':windows(snapshot(short_end=NOW-50000)),
                 'observed_at_ms':(NOW-68000)*1000,'argon_weekly':{'guard_until':NOW+604800},
                 'unrelated_legacy':77}
            k.store.save({'accounts':{'private@example.test':old}})
            self.assertEqual(await k.resume(13),'scheduled')
            await k.sweep();row=next(iter(k.store.load()['accounts'].values()))
            self.assertFalse(row['paused']);self.assertFalse(row['cycle_confirmed'])
            api.fresh['sources'][0]['observed_at_ms']=(NOW+61)*1000;k.clock=lambda:NOW+61
            await k.sweep();row=next(iter(k.store.load()['accounts'].values()))
            self.assertTrue(row['cycle_confirmed']);self.assertFalse(row['uncertain_attempt'])
            self.assertEqual(row['triggers'],9);self.assertEqual(row['unrelated_legacy'],77)
            self.assertEqual(row['argon_weekly'],old['argon_weekly']);self.assertFalse(api.generations)
