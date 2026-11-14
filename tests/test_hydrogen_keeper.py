import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI
from aetherstream.api.quota_keeper_routes import register_routes
from aetherstream.features.quota_keeper import hydrogen as h
from aetherstream.features.quota_keeper.client import GproxyClient, ProbeError
from aetherstream.features.quota_keeper.service import QuotaKeeper
from aetherstream.features.quota_keeper.state import StateStore, StateError

NOW=2_000_000_000
MODEL=h.DEFAULT_MODEL
MEMBERS=[{'id':12,'provider_id':4,'enabled':True,'label':'fake@example.test'}]
def reading(remaining=1,weekly=.8,offset=0):
 return {'available':True,'models':[MODEL], 'observed_at_ms':(NOW+offset)*1000,'windows':{
  h.SHORT:{'remaining':remaining,'end':NOW+offset+18000,'disabled':False},
  h.WEEKLY:{'remaining':weekly,'end':NOW+offset+604800,'disabled':False}}}
class Fake:
 def __init__(self,value=None):
  self.value=reading() if value is None else value;self.probes=[];self.generations=[];self.members=copy.deepcopy(MEMBERS)
  self.result={'http_status':200,'ok':True,'usage':{'inputTokens':2,'outputTokens':3},'reply':'PRIVATE'}
 async def hydrogen_probe(self,cid):
  self.probes.append(cid)
  if isinstance(self.value,BaseException):raise self.value
  return copy.deepcopy(self.value)
 async def generate_hydrogen(self,member,model):
  self.generations.append((member['id'],member['provider_id'],model))
  if isinstance(self.result,BaseException):raise self.result
  return self.result
 async def __aenter__(self):return self
 async def __aexit__(self,*a):pass
 async def api(self,name):return [{'id':4,'channel':'antigravity','enabled':True}] if name=='providers' else self.members

class Policy(unittest.IsolatedAsyncioTestCase):
 async def run_one(self,api,row,now=NOW,**kw):
  await h.handle_account(api,MEMBERS,row,kw.get('persist',lambda:None),lambda *a,**k:None,model=MODEL,clock=lambda:now,may_generate=kw.get('may_generate',lambda:True))
 async def test_unused_durable_guard_exactly_one_request(self):
  api=Fake();row={};writes=[];original=api.generate_hydrogen
  async def generate(*a):
   self.assertTrue(writes[-1]['uncertain_attempt']);self.assertTrue(writes[-1]['pending_verify']);return await original(*a)
  api.generate_hydrogen=generate
  await self.run_one(api,row,persist=lambda:writes.append(copy.deepcopy(row)));await self.run_one(api,row)
  self.assertEqual(api.generations,[(12,4,MODEL)]);self.assertEqual(row['guard_until'],NOW+18030);self.assertNotIn('PRIVATE',json.dumps(row))
 async def test_shared_gemini_usage_including_other_models_suppresses_wakeup(self):
  for rem in (0,.2,.999999):
   api=Fake(reading(rem));row={};await self.run_one(api,row);self.assertFalse(api.generations);self.assertEqual(row['next_at'],NOW+18030)
 async def test_weekly_exhaustion_overrides_full_short_and_disabled_short(self):
  api=Fake(reading(1,0));api.value['windows'][h.SHORT]['disabled']=True;row={};await self.run_one(api,row)
  self.assertFalse(api.generations);self.assertEqual(row['status'],'weekly_exhausted');self.assertEqual(row['next_at'],NOW+604830)
 async def test_missing_or_invalid_quota_fails_closed(self):
  cases=[]
  for bucket in h.DURATIONS:
   for key,val in [('remaining',None),('remaining',True),('remaining',float('nan')),('remaining',-1),('remaining',1.1),('end',NOW),('end',NOW-10),('end',NOW+h.DURATIONS[bucket]+301),('disabled',True)]:
    r=reading();r['windows'][bucket][key]=val;cases.append(r)
   r=reading();del r['windows'][bucket];cases.append(r)
  for value in cases:
   with self.subTest(value=value),self.assertRaises(ProbeError):await self.run_one(Fake(value),{})
 async def test_fixed_clock_even_full_suppresses_wakeup(self):
  api=Fake();api.value['observed_at_ms']=(NOW+61)*1000;row={'windows':reading()['windows'],'observed_at_ms':NOW*1000}
  await self.run_one(api,row,NOW+61);self.assertFalse(api.generations);self.assertTrue(row['cycle_confirmed'])
 async def test_possible_clock_needs_observation_not_generation(self):
  api=Fake();api.value['windows'][h.SHORT]['end']-=200;row={};await self.run_one(api,row)
  self.assertEqual(row['status'],'awaiting_cycle_clock');self.assertFalse(api.generations)
  api.value['observed_at_ms']=(NOW+61)*1000;await self.run_one(api,row,NOW+61);self.assertTrue(row['cycle_confirmed'])
 async def test_new_cycle_only_after_verified_previous_ends(self):
  api=Fake();row={};await self.run_one(api,row);api.value['observed_at_ms']=(NOW+61)*1000;await self.run_one(api,row,NOW+61)
  self.assertTrue(row['cycle_confirmed']);self.assertFalse(row['uncertain_attempt']);api.value=reading(offset=18031);await self.run_one(api,row,NOW+18031);self.assertEqual(len(api.generations),2)
 async def test_sliding_full_deadline_never_confirms_attempt_or_retries(self):
  api=Fake();row={};await self.run_one(api,row);api.value=reading(offset=61);await self.run_one(api,row,NOW+61)
  self.assertTrue(row['paused']);self.assertFalse(row['cycle_confirmed']);row.update(paused=False,next_at=0);api.value=reading(offset=604900);await self.run_one(api,row,NOW+604900);self.assertEqual(len(api.generations),1);self.assertTrue(row['paused'])
 async def test_ambiguous_delivery_timeout_cancel_never_reposts(self):
  for error in (httpx.ReadTimeout('PRIVATE'),asyncio.CancelledError()):
   api=Fake();api.result=error;row={}
   if isinstance(error,asyncio.CancelledError):
    with self.assertRaises(asyncio.CancelledError):await self.run_one(api,row)
   else:await self.run_one(api,row)
   self.assertTrue(row['uncertain_attempt']);api.value=reading(offset=61);await self.run_one(api,row,NOW+61);self.assertTrue(row['paused']);self.assertEqual(len(api.generations),1)
 async def test_permission_error_immediately_pauses(self):
  for status in (401,403):
   api=Fake();api.result={'http_status':status,'ok':False};row={};await self.run_one(api,row);self.assertTrue(row['paused']);self.assertTrue(row['uncertain_attempt']);await self.run_one(api,row,NOW+999999);self.assertEqual(len(api.generations),1)
 async def test_model_unavailable_daily_metadata_only(self):
  for change in ({'available':False},{'models':['gemini-4-argon-low']}):
   api=Fake();api.value.update(change);row={};await self.run_one(api,row);await self.run_one(api,row,NOW+60);self.assertFalse(api.generations);self.assertEqual(len(api.probes),1)
 async def test_disable_during_probe_and_existing_guard(self):
  for row,allow in (({},False),({'guard_until':NOW+500},True)):
   api=Fake();await self.run_one(api,row,may_generate=lambda:allow);self.assertFalse(api.generations)
 async def test_usage_change_alone_not_clock_confirmation(self):
  api=Fake();row={};await self.run_one(api,row)
  for n in (61,122):
   api.value=reading(.99,offset=n);await self.run_one(api,row,NOW+n)
  self.assertTrue(row['paused']);self.assertEqual(len(api.generations),1)

class Parser(unittest.IsolatedAsyncioTestCase):
 def data(self):
  return {'snapshot':{'observedAtMs':NOW*1000},'responses':[
   {'status':200,'body':{'models':{MODEL:{},'gemini-4-argon-low':{},'gemini-4-hydrogen-high':{'disabled':True}}}},
   {'status':200,'body':{'groups':[{'buckets':[{'bucketId':n,'remainingFraction':.8,'resetTime':datetime.fromtimestamp(NOW+d,timezone.utc).isoformat()} for n,d in {**h.DURATIONS,'gemini-argon-wtu-weekly':604800}.items()]}]}}]}
 async def probe(self,data):
  api=GproxyClient('http://mock.invalid','unused');api.api=AsyncMock(return_value=data)
  with patch('aetherstream.features.quota_keeper.client.time.time',return_value=NOW):return await api.hydrogen_probe(12)
 async def test_only_shared_windows_and_enabled_hydrogen_models(self):
  x=await self.probe(self.data());self.assertEqual(x['models'],[MODEL]);self.assertEqual(set(x['windows']),set(h.DURATIONS))
 async def test_missing_duplicate_wrong_family_never_substituted(self):
  for name in h.DURATIONS:
   for duplicate in (False,True):
    d=self.data();bs=d['responses'][1]['body']['groups'][0]['buckets']
    if duplicate:bs.append(copy.deepcopy(next(b for b in bs if b['bucketId']==name)))
    else:bs[:]=[b for b in bs if b['bucketId']!=name]
    with self.assertRaises(ProbeError):await self.probe(d)
 async def test_no_model_permission_not_inferred_from_quota(self):
  d=self.data();d['responses'][0]['body']['models'][MODEL]['disabled']=True;r=await self.probe(d);self.assertFalse(r['available'])
 async def test_exact_account_native_model_test_and_no_fallback(self):
  api=GproxyClient('http://mock.invalid','unused');api.api=AsyncMock(return_value={'status':200,'ok':True,'usage':{'outputTokens':2}});await api.generate_hydrogen(MEMBERS[0],MODEL)
  api.api.assert_awaited_once_with('models/test',{'providerId':'v3-providers-4','credentialId':'v3-credentials-12','model':MODEL})
  with self.assertRaises(ValueError):await api.generate_hydrogen(MEMBERS[0],'gemini-4-argon-low')

class StateSettings(unittest.TestCase):
 def test_config_opt_in(self):
  self.assertFalse(h.settings(None)['enabled']);self.assertEqual(h.settings({'enabled':True})['model'],MODEL)
  for v in ({'enabled':'yes'},{'model':'claude-opus-4-6-thinking'},{'foo':1},True):
   with self.assertRaises(ValueError):h.settings(v)
 def test_state_is_additive_and_bad_guards_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   store=StateStore(td);base={'accounts':{'fake':{'paused':True,'argon_weekly':{'guard_until':NOW},'hydrogen':{'uncertain_attempt':True,'guard_until':NOW+100}}},'schema_version':1};store.save(base);self.assertEqual(store.load(),base)
   for bad in ([],{'guard_until':True},{'next_at':-1},{'uncertain_attempt':'true'}):
    base['accounts']['fake']['hydrogen']=bad;store.save(base)
    with self.assertRaises(StateError):store.load()
 def test_status_sanitizes(self):
  self.assertNotIn('PRIVATE',json.dumps(h.status({'windows':{h.SHORT:{'remaining':'PRIVATE','secret':'PRIVATE'}},'secret':'PRIVATE'})))

class Feature(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.api=Fake();self.flags={'enabled':True,'claude':{'enabled':False},'hydrogen':{'enabled':True}};self.logs=[]
  self.keeper=QuotaKeeper(lookup=lambda root,key:self.flags.get(key),state_dir=self.temp.name,base_url='http://mock.invalid',credentials_file='unused',log=self.logs.append,client_factory=lambda:self.api,clock=lambda:NOW)
 def row(self):return next(iter(self.keeper.store.load()['accounts'].values()))
 async def test_disabled_default_leaves_existing_families_unchanged(self):
  self.flags.pop('hydrogen');await self.keeper.sweep();self.assertNotIn('hydrogen',self.row());self.assertFalse(self.api.probes)
 async def test_known403_prevents_new_family_even_with_full_quota(self):
  for family in ('root','argon_weekly'):
   original={'paused':True,'guard_until':NOW+100,'argon_weekly':{'guard_until':NOW+900}}
   target=original if family=='root' else original['argon_weekly'];target['last_generation']={'http_status':403}
   self.keeper.store.save({'accounts':{'fake@example.test':original}});await self.keeper.sweep();r=self.row();self.assertTrue(r['hydrogen']['paused']);self.assertFalse(self.api.probes);self.assertFalse(self.api.generations);self.assertEqual(r['guard_until'],NOW+100);self.assertEqual(r['argon_weekly'],original['argon_weekly'])
 async def test_nonpermission_claude_pause_independent_and_duplicates_deduped(self):
  self.api.members.append(dict(MEMBERS[0],id=99));self.keeper.store.save({'accounts':{'fake@example.test':{'paused':True,'guard_until':NOW+1}}});await self.keeper.sweep();await self.keeper.sweep();self.assertEqual(len(self.api.generations),1);self.assertTrue(self.row()['paused'])
 async def test_resume_preserves_uncertainty_and_other_families(self):
  await self.keeper.sweep();old=self.row();await self.keeper.resume_hydrogen(12);r=self.row();self.assertEqual(r['hydrogen']['guard_until'],old['hydrogen']['guard_until']);self.assertTrue(r['hydrogen']['uncertain_attempt']);self.assertEqual({k:v for k,v in r.items() if k!='hydrogen'},{k:v for k,v in old.items() if k!='hydrogen'})
 async def test_shared_lock_prevents_network_and_stop_waits(self):
  with StateStore(self.temp.name).locked():await self.keeper.sweep()
  self.assertFalse(self.api.probes);entered=asyncio.Event();release=asyncio.Event()
  async def generate(*args):entered.set();await release.wait();return self.api.result
  self.api.generate_hydrogen=generate;await self.keeper.start();await asyncio.wait_for(entered.wait(),2);task=asyncio.create_task(self.keeper.stop())
  try:
   await asyncio.sleep(.01);self.assertFalse(task.done())
   with StateStore(self.temp.name).locked() as acquired:self.assertFalse(acquired)
  finally:release.set();await task
  self.assertTrue(self.row()['hydrogen']['uncertain_attempt'])
 async def test_status_resume_private_cpu_only(self):
  await self.keeper.sweep();count=len(self.api.probes);app=FastAPI();token=Path(self.temp.name)/'token';token.write_text('x'*40);register_routes(app,keeper=self.keeper,admin_token_file=str(token))
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://mock.invalid') as c:
   path='/admin/quota-keeper/credentials/12/hydrogen/resume';self.assertEqual((await c.post(path)).status_code,401);auth={'Authorization':'Bearer '+'x'*40};r=await c.get('/admin/quota-keeper',headers=auth);self.assertTrue(r.json()['hydrogen']['enabled']);self.assertEqual((await c.post(path,headers=auth)).status_code,200)
   self.assertEqual((await c.post(path.replace('/12/','/99/'),headers=auth)).status_code,404)
  self.assertEqual(len(self.api.probes),count)

if __name__=='__main__':unittest.main()
