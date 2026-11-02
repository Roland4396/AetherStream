"""Pure ownership migration audit, fake keys only, no provider access."""
import json,sys
from pathlib import Path
sys.path.insert(0,'/app/test-acc-source')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from channel_config import ChannelRegistry
from aetherstream.routing.policy_config import enrich_selection,dedupe_identity,merge

legacy=json.loads(Path('/tmp/migration-fixture/pool.json').read_text())
flags=json.loads(Path('/tmp/migration-fixture/flags.json').read_text())
old=ChannelRegistry(legacy)
models=list(old.exact)
models += ['free/custom-wire','free/claude-new-model','free/gemini-new-model','pro/custom-wire']
count=keys=0
for model in models:
 resolution=old.resolve(model)
 if resolution is None:continue
 selected=resolution.descriptor()
 plan=enrich_selection(lambda key:flags.get(key),selected)
 local=plan['policy'].copy();local.pop('_local_policy');local.pop('account_overrides')
 assert local==resolution.policy,(model,'policy changed')
 for account in legacy['accounts']:
  oldstop=old.selected_policy(resolution,account).get('early_stop',{})
  newstop=merge(local.get('early_stop',{}),plan['policy']['account_overrides'].get(account['id'],{}).get('early_stop',{}))
  assert newstop==oldstop,(model,account['id'],'post-selection policy changed')
 for payload in [{'model':model,'messages':[{'role':'user','content':'Synthetic'}],'stream':False},
                 {'model':model,'messages':[],'input':'Synthetic','temperature':0.6,'stream':False}]:
  for headers in [{'Authorization':'Bearer service-test','Content-Type':'application/json'},
                  {'x-api-key':'service-test','anthropic-version':'2023-06-01','Content-Type':'application/json'},
                  {'Authorization':'Bearer service-test','X-ACC-Passthrough-Authorization':'Bearer caller-test'}]:
   previous=old.dedupe_identity(resolution,payload,headers)
   current=dedupe_identity(plan,payload,headers)
   if resolution.dedupe:assert current['key']==previous['key'] and current['shape']==previous['shape'],(model,'shared identity changed')
   keys+=1
 count+=1
print(json.dumps({'policy_models_unchanged':count,'dedupe_checks':keys,'account_overrides_unchanged':True,'pool_selection_unchanged':True}))
