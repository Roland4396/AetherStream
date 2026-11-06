import copy
import tempfile
import unittest
from aetherstream.features.quota_keeper import claude
from aetherstream.features.quota_keeper.client import MODEL
from aetherstream.features.quota_keeper.service import QuotaKeeper
from test_quota_keeper import FakeApi,snapshot,NOW

OLD='claude-opus-4-6-thinking';NEW='claude-opus-5-5-low'


class AccountSettingsTests(unittest.TestCase):
    def test_default_preserves_old_account_model(self):
        self.assertEqual(MODEL,OLD)
        self.assertEqual(claude.select_model(claude.settings(None),[{'id':13}]),OLD)

    def test_only_explicit_new_account_overrides_change_model(self):
        c=claude.settings({'model':OLD,'account_models':{'12':NEW,'31':NEW}})
        for cid in [4,9,10,11,13,14,15,16,21,23,24,25,26,27]:
            self.assertEqual(claude.select_model(c,[{'id':cid}]),OLD)
        for cid in [12,31,'v3-credentials-31']:
            self.assertEqual(claude.select_model(c,[{'id':cid}]),NEW)

    def test_native_identity_and_legacy_id_have_one_config_identity(self):
        c=claude.settings({'account_models':{'31':NEW,'v3-credentials-31':NEW}})
        self.assertEqual(c['account_models'],{'v3-credentials-31':NEW})
        with self.assertRaises(ValueError):
            claude.settings({'account_models':{'31':NEW,'v3-credentials-31':OLD}})

    def test_duplicate_account_imports_share_the_override_and_conflicts_fail_closed(self):
        members=[{'id':4},{'id':24}]
        c=claude.settings({'account_models':{'4':NEW}})
        self.assertEqual(claude.select_model(c,members),NEW)
        c=claude.settings({'account_models':{'4':NEW,'24':OLD}})
        with self.assertRaises(ValueError):claude.select_model(c,members)

    def test_bad_override_configuration_fails_closed(self):
        for value in [[],{'31':'gemini-4-argon'},{'../private':NEW},{31:NEW},{'0':NEW}]:
            with self.assertRaises(ValueError):claude.settings({'account_models':value})


class MixedApi(FakeApi):
    def __init__(self):
        super().__init__();self.submissions=[]
    async def api(self,name):
        self.api_calls.append(name)
        if name=='providers':return [{'id':4,'channel':'antigravity','enabled':True}]
        return [{'id':cid,'provider_id':4,'enabled':True,'label':f'fake{cid}@example.test'} for cid in (13,31)]
    async def probe(self,cid):
        self.probes.append(cid);s=snapshot();s['available_models']=[OLD if cid==13 else NEW];return s
    async def generate(self,cid,**kwargs):
        self.submissions.append((cid,kwargs));return self.result


class MixedPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_and_new_accounts_use_different_models_on_exact_provider(self):
        with tempfile.TemporaryDirectory() as td:
            flags={'enabled':True,'claude':{'enabled':True,'model':OLD,'account_models':{'31':NEW}}}
            api=MixedApi();k=QuotaKeeper(lookup=lambda root,key:flags.get(key),state_dir=td,
                base_url='http://unused.invalid',credentials_file='unused',log=lambda *a:None,
                client_factory=lambda:api,clock=lambda:NOW)
            await k.sweep()
            self.assertEqual(api.submissions,[(13,{'provider_id':4,'model':OLD}),(31,{'provider_id':4,'model':NEW})])
            status={a['credential_id']:a['configured_model'] for a in k.status()['accounts']}
            self.assertEqual(status,{13:OLD,31:NEW})
            await k.sweep();self.assertEqual(len(api.submissions),2)

    async def test_missing_override_pauses_only_new_account_without_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            flags={'enabled':True,'claude':{'model':OLD}}
            api=MixedApi();k=QuotaKeeper(lookup=lambda root,key:flags.get(key),state_dir=td,
                base_url='http://unused.invalid',credentials_file='unused',log=lambda *a:None,
                client_factory=lambda:api,clock=lambda:NOW)
            await k.sweep()
            self.assertEqual(api.submissions,[(13,{'provider_id':4,'model':OLD})])
            rows={a['credential_id']:a for a in k.status()['accounts']}
            self.assertEqual(rows[31]['status'],'paused_model_unavailable')
            self.assertEqual(rows[13]['status'],'pending_verification')
