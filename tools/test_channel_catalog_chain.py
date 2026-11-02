#!/usr/bin/env python3
"""Real local HTTP regression: authenticated catalog views never gate inference.

Run in --network none with ACC_TEST_SOURCE; no vendor/GPU calls are possible.
"""
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, StreamingResponse

from test_channel_chain import server, acc, account_channels, ChannelGatewayTests, policy, events
from test_channel_chain import chat_completions, protocol_entry, route_protocol_request
from test_channel_chain import RuntimeLifecycle, LifecycleMiddleware
from channel_catalog import CatalogCache


async def main():
    calls = []
    selected_protocol = 'chat'
    fake = FastAPI()

    @fake.api_route('/{path:path}', methods=['GET', 'POST'])
    async def provider(path: str, request: Request):
        calls.append((request.method, path))
        if request.method == 'GET':
            assert path == 'mio/v1/models', path
            assert request.headers['authorization'] == 'Bearer client-test'
            return {'data': [{'id': 'mio-only-model'}]}
        expected = 'Bearer client-test' if path.startswith('mio/') else 'Bearer static-test'
        if path.endswith('/messages'):
            assert request.headers['x-api-key'] == 'static-test'
        else:
            assert request.headers['authorization'] == expected
        async def body():
            for chunk in events(selected_protocol):
                yield chunk
                await asyncio.sleep(0)
        return StreamingResponse(body(), media_type='text/event-stream')

    fake_port, fake_server, fake_task = await server(fake)
    acc.CONFIG_PATH = Path('/tmp/catalog-chain-unused.json')
    for name in ('reload_config_if_needed', 'apply_daily_resets', 'apply_limited_recovery_probes', 'persist_config'):
        setattr(acc, name, AsyncMock())
    acc_port, acc_server, acc_task = await server(acc.app)
    lookup = lambda *keys: {'enabled': True, 'base_url': f'http://127.0.0.1:{acc_port}',
        'api_key': 'service-test'} if keys == ('account_routing',) else None
    deps = ChannelGatewayTests().deps({})
    deps.resolve_account_route = lambda model, **kw: account_channels.resolve(lookup, model, **kw)
    deps.build_account_dedupe = lambda *args: account_channels.dedupe_key(lookup, *args)
    web = FastAPI()
    runtime = RuntimeLifecycle(instance='catalog-chain', control_file='',
        state_dir='/tmp/catalog-chain-state', log=lambda *_: None)
    web.add_middleware(LifecycleMiddleware, runtime=runtime)

    @web.post('/v1/chat/completions')
    async def chat(request: Request):
        return await protocol_entry(request, deps, lambda: chat_completions(request, deps), interval=.05)

    @web.post('/v1/messages')
    async def messages(request: Request):
        return await protocol_entry(request, deps,
            lambda: route_protocol_request(request, deps, protocol='anthropic'), interval=.05)

    @web.post('/v1/responses')
    async def responses(request: Request):
        return await protocol_entry(request, deps,
            lambda: route_protocol_request(request, deps, protocol='responses'), interval=.05)

    stream_port, stream_server, stream_task = await server(web)

    def configure(protocol):
        settings = policy(streaming={'force_upstream': 'stream', 'exact_dedupe': True},
                          early_stop={'enabled': False})
        account = {'id': 'static', 'status': 'ok', 'base_url': f'http://127.0.0.1:{fake_port}/static/v1',
            'api_key': 'static-test', 'models': ['*'], 'auto_reset_enabled': False,
            'status_tracking_enabled': False}
        mio = {**account, 'id': 'mio', 'base_url': f'http://127.0.0.1:{fake_port}/mio/v1',
            'api_key': '', 'auth_mode': 'caller'}
        acc.state = acc.PoolState(config={'accounts': [account, mio], 'routing': {
            'schema': 1, 'enabled': True, 'client_keys': ['service-test'], 'defaults': settings,
            'channels': [
                {'id': 'static', 'accounts': ['static'], 'routes': [{
                    'id': 'static-route', 'transport': protocol, 'models': {'test': 'wire-model'},
                    'dedupe': {'kind': 'payload', 'shape': 'chat', 'model': '$upstream'}}]},
                {'id': 'mio', 'accounts': ['mio'], 'routes': [{
                    'id': 'mio-route', 'transport': 'chat', 'models': {},
                    'discovery': {'account': 'mio'},
                    'policy': {'adapter': {'caller_authorization': True}},
                    'dedupe': {'kind': 'payload', 'shape': 'chat', 'model': '$upstream'}}]},
            ]}})
        acc.catalog_cache = CatalogCache()
        calls.clear()

    count = 0
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            for protocol in ('chat', 'messages', 'responses'):
                selected_protocol = protocol
                for public in ('chat/completions', 'messages', 'responses'):
                    for streaming in (False, True):
                        for warm_catalog in (False, True):
                            configure(protocol)
                            if warm_catalog:
                                listing = await client.get(f'http://127.0.0.1:{acc_port}/v1/models', headers={
                                    'Authorization': 'Bearer service-test',
                                    'X-ACC-Passthrough-Authorization': 'Bearer client-test'})
                                assert listing.status_code == 200, listing.text
                                assert 'mio-only-model' in [m['id'] for m in listing.json()['data']]
                            expected_gets = 1 if warm_catalog else 0
                            result = await client.post(f'http://127.0.0.1:{stream_port}/v1/{public}',
                                headers={'Authorization': 'Bearer client-test'}, json={
                                    'model': 'test', 'stream': streaming, 'input': 'Synthetic',
                                    'messages': [{'role': 'user', 'content': 'Synthetic'}]})
                            assert result.status_code == 200, (protocol, public, streaming, result.text)
                            assert 'OK' in result.text, result.text
                            assert 'configuration changed' not in result.text.lower()
                            assert len([r for r in calls if r[0] == 'GET']) == expected_gets, calls
                            assert len([r for r in calls if r[0] == 'POST']) == 1, calls
                            for _ in range(100):
                                if not runtime.active and len(deps.active_stream_registry) == 0:
                                    break
                                await asyncio.sleep(.01)
                            assert not runtime.active and len(deps.active_stream_registry) == 0
                            count += 1
                            print(json.dumps({'protocol': protocol, 'public': public, 'stream': streaming,
                                'catalog_warmed': warm_catalog, 'catalog_gets': expected_gets,
                                'inference_calls': 1, 'status': 200}), flush=True)
            # A direct dynamic Mio request still works without a prior models GET.
            selected_protocol = 'chat'
            configure('chat')
            result = await client.post(f'http://127.0.0.1:{stream_port}/v1/chat/completions',
                headers={'Authorization': 'Bearer client-test'}, json={'model': 'mio-only-model',
                'stream': False, 'messages': [{'role': 'user', 'content': 'Synthetic'}]})
            assert result.status_code == 200 and 'OK' in result.text, result.text
            assert calls == [('GET', 'mio/v1/models'), ('POST', 'mio/v1/chat/completions')], calls
            count += 1
            print('CATALOG_CHAIN_PASS', count, 'real HTTP cases', flush=True)
    finally:
        for service in (stream_server, acc_server, fake_server):
            service.should_exit = True
        await asyncio.gather(stream_task, acc_task, fake_task)


if __name__ == '__main__':
    asyncio.run(main())
