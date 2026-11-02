"""Real socket EOF in each native/public protocol, JSON and SSE, no inference."""
import asyncio,json,tempfile,time
from pathlib import Path
from unittest.mock import AsyncMock
import httpx
from fastapi import FastAPI,Request
from test_channel_chain import (set_fixture,server,acc,policy,events,account_channels,ChannelGatewayTests,
 chat_completions,protocol_entry,route_protocol_request,RuntimeLifecycle,LifecycleMiddleware)
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.runtime.shared import SharedRuntimeState

async def main():
    mode='clean';protocol='chat';calls=0;ended=asyncio.Event();eof_at=None
    async def provider(reader,writer):
        nonlocal calls,eof_at
        head=await reader.readuntil(b'\r\n\r\n')
        length=next((int(x.split(':',1)[1]) for x in head.decode().split('\r\n') if x.lower().startswith('content-length:')),0)
        if length:await reader.readexactly(length)
        calls+=1
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n')
        if mode!='empty':
            for chunk in events(protocol)[:2 if protocol=='chat' else 3 if protocol=='messages' else 2]:
                writer.write(f'{len(chunk):x}\r\n'.encode()+chunk+b'\r\n')
        if mode=='broken':writer.write(b'100\r\ndata: incomplete')
        else:writer.write(b'0\r\n\r\n')
        await writer.drain();eof_at=time.monotonic();writer.close();await writer.wait_closed();ended.set()
    fake=await asyncio.start_server(provider,'127.0.0.1',0);port=fake.sockets[0].getsockname()[1]
    acc.CONFIG_PATH=Path('/tmp/eof-unused.json')
    for name in ('reload_config_if_needed','apply_daily_resets','apply_limited_recovery_probes','persist_config'):setattr(acc,name,AsyncMock())
    acc_port,acc_server,acc_task=await server(acc.app)
    feature_flags = {}
    lookup=lambda *keys:{'enabled':True,'base_url':f'http://127.0.0.1:{acc_port}','api_key':'service-test'} if keys==('account_routing',) else feature_flags.get(keys[0])
    deps=ChannelGatewayTests().deps({});deps.resolve_account_route=lambda model,**kw:account_channels.resolve(lookup,model,**kw)
    deps.build_account_dedupe=lambda *args:account_channels.dedupe_key(lookup,*args)
    web=FastAPI();runtime=RuntimeLifecycle(instance='eof-test',control_file='',state_dir='/tmp/eof-life',log=lambda *_:None)
    web.add_middleware(LifecycleMiddleware,runtime=runtime)
    @web.post('/v1/chat/completions')
    async def chat(request:Request):return await protocol_entry(request,deps,lambda:chat_completions(request,deps),interval=.05)
    @web.post('/v1/messages')
    async def messages(request:Request):return await protocol_entry(request,deps,lambda:route_protocol_request(request,deps,protocol='anthropic'),interval=.05)
    @web.post('/v1/responses')
    async def responses(request:Request):return await protocol_entry(request,deps,lambda:route_protocol_request(request,deps,protocol='responses'),interval=.05)
    web_port,web_server,web_task=await server(web);results=[]
    try:
        for protocol in ('chat','messages','responses'):
            for public in ('chat/completions','messages','responses'):
                for stream in (True,False):
                    for mode in ('empty','clean','broken'):
                        with tempfile.TemporaryDirectory() as directory:
                            coalescer=ExactRequestCoalescer(ttl=60,log=lambda *_:None,shared_state=SharedRuntimeState(directory))
                            deps.run_exact_nonstream_once=coalescer.run
                            calls=0;ended.clear();eof_at=None
                            account={'id':'fake','status':'ok','base_url':f'http://127.0.0.1:{port}/v1','api_key':'test','models':['*'],
                                'auto_reset_enabled':False,'status_tracking_enabled':False}
                            set_fixture({'accounts':[account],'routing':{'schema':1,'enabled':True,'client_keys':['service-test'],
                                'defaults':policy(streaming={'force_upstream':'stream','exact_dedupe':True},early_stop={'enabled':False}),
                                'channels':[{'id':'test','accounts':['fake'],'routes':[{'id':'test','transport':protocol,'models':{'test':'wire-model'}}]}]}}, feature_flags)
                            async with httpx.AsyncClient(trust_env=False,timeout=5) as client:
                                task=asyncio.create_task(client.post(f'http://127.0.0.1:{web_port}/v1/{public}',json={'model':'test','stream':stream,
                                    'messages':[{'role':'user','content':'Synthetic'}],'input':'Synthetic'}))
                                await asyncio.wait_for(ended.wait(),3)
                                response=await asyncio.wait_for(task,1)
                            for _ in range(100):
                                if not runtime.active and not coalescer._inflight and not len(deps.active_stream_registry):break
                                await asyncio.sleep(.01)
                            assert calls==1 and not runtime.active and not coalescer._inflight and not len(deps.active_stream_registry)
                            elapsed=(time.monotonic()-eof_at)*1000
                            row={'native':protocol,'public':public,'stream':stream,'close':mode,'after_eof_ms':round(elapsed,2),
                                 'calls':calls,'inflight':0,'status':response.status_code}
                            results.append(row);print(json.dumps(row),flush=True)
        print('UPSTREAM_EOF_PASS',len(results),flush=True)
    finally:
        web_server.should_exit=True;acc_server.should_exit=True
        await asyncio.gather(web_task,acc_task)
        fake.close();await fake.wait_closed()

if __name__=='__main__':asyncio.run(main())
