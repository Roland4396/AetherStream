"""Regression: a lone JSON client disconnect reaches the ultimate fake socket."""
import asyncio,json,tempfile,time
from pathlib import Path
import httpx
from fastapi import FastAPI,Request
from test_channel_chain import (set_fixture,server,acc,Upstream,policy,account_channels,ChannelGatewayTests,
 chat_completions,protocol_entry,RuntimeLifecycle,LifecycleMiddleware)
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.runtime.shared import SharedRuntimeState
from unittest.mock import AsyncMock

async def main():
 upstream=Upstream();fake=await asyncio.start_server(upstream.handle,'127.0.0.1',0)
 port=fake.sockets[0].getsockname()[1]
 for name in ('reload_config_if_needed','apply_daily_resets','apply_limited_recovery_probes','persist_config'):
  setattr(acc,name,AsyncMock())
 acc.CONFIG_PATH=Path('/tmp/unused-acc-config.json')
 acc_port,acc_server,acc_task=await server(acc.app)
 feature_flags = {}
 lookup=lambda *keys: {'enabled':True,'base_url':f'http://127.0.0.1:{acc_port}','api_key':'service-test'} if keys==('account_routing',) else feature_flags.get(keys[0])
 deps=ChannelGatewayTests().deps({});deps.resolve_account_route=lambda model,**kw:account_channels.resolve(lookup,model,**kw)
 deps.build_account_dedupe=lambda *args:account_channels.dedupe_key(lookup,*args)
 web=FastAPI();runtime=RuntimeLifecycle(instance='cancel-test',control_file='',state_dir='/tmp/cancel-life',log=print)
 web.add_middleware(LifecycleMiddleware,runtime=runtime)
 @web.post('/v1/chat/completions')
 async def chat(request:Request):return await protocol_entry(request,deps,lambda:chat_completions(request,deps),interval=.05)
 stream_port,stream_server,stream_task=await server(web)
 coalescer=None
 try:
  for shared in (False,True):
   with tempfile.TemporaryDirectory() as directory:
    coalescer=ExactRequestCoalescer(ttl=60,log=print,shared_state=SharedRuntimeState(directory) if shared else None)
    deps.run_exact_nonstream_once=coalescer.run
    upstream.reset('headers','chat')
    account={'id':'fixture','status':'ok','base_url':f'http://127.0.0.1:{port}/v1',
     'api_key':'test','models':['*'],'auto_reset_enabled':False,'status_tracking_enabled':False}
    set_fixture({'accounts':[account],'routing':{'schema':1,'enabled':True,'client_keys':['service-test'],
     'defaults':policy(streaming={'force_upstream':'json','exact_dedupe':True}),
     'channels':[{'id':'test','accounts':['fixture'],'routes':[{'id':'test','transport':'chat','models':{'test':'wire'}}]}]}}, feature_flags)
    async with httpx.AsyncClient(trust_env=False,timeout=5) as client:
     task=asyncio.create_task(client.post(f'http://127.0.0.1:{stream_port}/v1/chat/completions',json={
      'model':'test','stream':False,'messages':[{'role':'user','content':'Synthetic'}]}))
     await asyncio.wait_for(upstream.received.wait(),3);closed_at=time.monotonic();task.cancel()
     try:await task
     except asyncio.CancelledError:pass
     try:await asyncio.wait_for(upstream.closed.wait(),1)
     except asyncio.TimeoutError:
      print(json.dumps({'shared_state':shared,'provider_socket_closed':False,'coalescer_state':await coalescer.state()}),flush=True)
      raise AssertionError('Client disconnected but unobserved upstream continues')
     for _ in range(100):
      if not runtime.active and not coalescer._inflight:break
      await asyncio.sleep(.01)
     assert not runtime.active and not coalescer._inflight
     assert upstream.calls==1
     print(json.dumps({'shared_state':shared,'provider_socket_closed':True,'close_ms':round((upstream.eof_at-closed_at)*1000,2),
      'coalescer_state':await coalescer.state()}),flush=True)
  print('LAST_WAITER_PASS 2',flush=True)
 finally:
  if coalescer:
   tasks=[x['task'] for x in coalescer._inflight.values()]
   for t in tasks:t.cancel()
   await asyncio.gather(*tasks,return_exceptions=True)
  stream_server.should_exit=True;acc_server.should_exit=True
  await asyncio.gather(stream_task,acc_task)
  fake.close();await fake.wait_closed()
  for w in list(upstream.writers):w.close()

if __name__=='__main__':asyncio.run(main())
