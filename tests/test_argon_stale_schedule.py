"""Hourly metadata reconciliation cannot erase permission/paid-work barriers."""
import copy
import unittest
from test_argon_weekly_keeper import NOW, MODEL, MEMBERS, reading, FakeApi, FeatureApi
from aetherstream.features.quota_keeper import argon
from aetherstream.features.quota_keeper.service import QuotaKeeper
import tempfile

class StaleScheduleTests(unittest.IsolatedAsyncioTestCase):
    async def run_row(self, row, api=None, now=NOW, may_generate=lambda: True):
        api=api or FakeApi()
        await argon.handle_argon_account(api,MEMBERS,row,lambda:None,lambda *a,**k:None,
            model=MODEL,clock=lambda:now,may_generate=may_generate)
        return api
    def stale(self, **changes):
        return dict(status='waiting',next_at=NOW+86400,last_probe_at=NOW-86400,
            window={'remaining':0,'end':NOW+86400},observed_at_ms=(NOW-86400)*1000,**changes)
    async def test_stale_future_wait_sees_early_replenishment_once(self):
        row=self.stale(); api=await self.run_row(row); await self.run_row(row,api)
        self.assertEqual(len(api.probes),1); self.assertEqual(len(api.generations),1)
        self.assertTrue(row['uncertain_attempt']); self.assertGreater(row['guard_until'],NOW)
    async def test_recent_wait_remains_network_free(self):
        row=self.stale(); row['last_probe_at']=NOW-argon.QUOTA_RECHECK+1
        api=await self.run_row(row); self.assertFalse(api.probes); self.assertFalse(api.generations)
    async def test_metadata_hourly_bound_preserves_used_window(self):
        row=self.stale(); api=FakeApi(reading(.2,NOW+86400))
        await self.run_row(row,api); self.assertFalse(api.generations)
        await self.run_row(row,api,NOW+argon.QUOTA_RECHECK-1); self.assertEqual(len(api.probes),1)
        api.value['observed_at_ms']=(NOW+argon.QUOTA_RECHECK)*1000
        await self.run_row(row,api,NOW+argon.QUOTA_RECHECK); self.assertEqual(len(api.probes),2)
        self.assertEqual(row['next_at'],NOW+86400+30); self.assertFalse(api.generations)
    async def test_stale_unused_with_guard_never_posts(self):
        row=self.stale(guard_until=NOW+10000); api=await self.run_row(row)
        self.assertEqual(len(api.probes),1); self.assertFalse(api.generations)
        self.assertEqual(row['guard_until'],NOW+10000); self.assertEqual(row['status'],'request_guard')
    async def test_guard_wait_metadata_rechecked_without_early_post(self):
        row=self.stale(guard_until=NOW+10000); row['status']='request_guard'
        api=await self.run_row(row); self.assertEqual(len(api.probes),1); self.assertFalse(api.generations)
    async def test_stale_ambiguity_preserved_indefinitely(self):
        for fields in ({'uncertain_attempt':True},{'pending_verify':True},
                       {'last_trigger_at':NOW-3*argon.WEEK,'cycle_confirmed':False}):
            row=self.stale(**fields); api=await self.run_row(row)
            self.assertFalse(api.generations); self.assertTrue(row['uncertain_attempt']); self.assertTrue(row['paused'])
    async def test_fixed_clock_at_full_quota_is_not_unused(self):
        row=self.stale(); api=await self.run_row(row,FakeApi(reading(1,NOW+5000)))
        self.assertFalse(api.generations); self.assertEqual(row['status'],'awaiting_cycle_clock')
        self.assertFalse(row['cycle_confirmed'])
        api.value['observed_at_ms']=(NOW+61)*1000
        await self.run_row(row,api,NOW+61)
        self.assertEqual(row['status'],'waiting'); self.assertTrue(row['cycle_confirmed'])
        self.assertFalse(api.generations)
    async def test_pause_backoff_and_unavailable_deadlines_remain_authoritative(self):
        for status in ('quota_retry_wait','unavailable','pending_verification','delivery_uncertain','submitted'):
            row=self.stale(); row['status']=status; before=copy.deepcopy(row)
            api=await self.run_row(row); self.assertFalse(api.probes); self.assertEqual(row,before)
        row=self.stale(paused=True); api=await self.run_row(row); self.assertFalse(api.probes)
    async def test_not_generating_does_not_clear_existing_clock_or_guard(self):
        row=self.stale(); before=row['next_at']; api=await self.run_row(row,may_generate=lambda:False)
        self.assertEqual(len(api.probes),1); self.assertFalse(api.generations)
        self.assertEqual(row['next_at'],before); self.assertNotIn('last_trigger_at',row)
    async def test_missing_saved_probe_time_rechecks_legacy_wait(self):
        row=self.stale(); del row['last_probe_at']; api=await self.run_row(row)
        self.assertEqual(len(api.probes),1)
    async def test_metadata_failure_uses_existing_retry_backoff(self):
        row=self.stale(uncertain_attempt=True,guard_until=NOW+10000)
        argon.quota_failure(row,NOW,ValueError(),lambda:None,lambda *a,**k:None)
        api=await self.run_row(row); self.assertFalse(api.probes)
        self.assertTrue(row['uncertain_attempt']); self.assertEqual(row['guard_until'],NOW+10000)

class PermissionGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_401_403_any_family_prevents_argon_metadata_and_generation(self):
        for family in ('parent','argon_weekly','hydrogen'):
            for status in (401,403):
                with self.subTest(family=family,status=status), tempfile.TemporaryDirectory() as tmp:
                    api=FeatureApi(); flags={'enabled':True,'claude':{'enabled':False},'argon_weekly':{'enabled':True}}
                    keeper=QuotaKeeper(lookup=lambda root,key:flags.get(key),state_dir=tmp,
                        base_url='http://unused.invalid',credentials_file='unused',log=lambda x:None,
                        client_factory=lambda:api,clock=lambda:NOW)
                    row={'argon_weekly':{'guard_until':NOW+10000,'uncertain_attempt':True,'pending_verify':True},'paused':True}
                    target=row if family=='parent' else row.setdefault(family,{})
                    target['last_generation']={'http_status':status,'ok':False}
                    keeper.store.save({'accounts':{'fake@example.test':row}})
                    await keeper.sweep(); after=next(iter(keeper.store.load()['accounts'].values()))['argon_weekly']
                    self.assertEqual(after['status'],'paused_account_permission'); self.assertTrue(after['paused'])
                    self.assertEqual(after['guard_until'],NOW+10000); self.assertTrue(after['uncertain_attempt']); self.assertTrue(after['pending_verify'])
                    self.assertFalse(api.probes); self.assertFalse(api.generations)
