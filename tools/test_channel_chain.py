#!/usr/bin/env python3
"""Real loopback Stream -> ACC -> never-ending upstream; no external access.

Run inside --network none, with ACC_TEST_SOURCE pointing to the ACC source.
Covers manual body abort, pre-header abort, strict identity preflight abort and
policy stop across all three public APIs and all three native SSE transports.
"""
import asyncio
import copy
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import uvicorn
from fastapi import FastAPI, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_channel_gateway import events, policy, ChannelGatewayTests
from aetherstream.api import app as stream
from aetherstream.api.channel_gateway import chat_completions
from aetherstream.api.request_entry import protocol_entry
from aetherstream.api.protocol_gateway import route_protocol_request
from aetherstream.routing import account_channels
from aetherstream.runtime.lifecycle import RuntimeLifecycle, LifecycleMiddleware
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.runtime.shared import SharedRuntimeState

acc_path = Path(os.environ['ACC_TEST_SOURCE'])
sys.path.insert(0, str(acc_path))
spec = importlib.util.spec_from_file_location('chain_acc', acc_path / 'app.py')
acc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = acc
spec.loader.exec_module(acc)


async def server(app):
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen()
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level='warning', lifespan='off', access_log=False)
    service = uvicorn.Server(config)
    service.install_signal_handlers = lambda: None
    task = asyncio.create_task(service.serve(sockets=[sock]))
    while not service.started:
        if task.done():
            await task
        await asyncio.sleep(0.01)
    return port, service, task


class Upstream:
    def __init__(self):
        self.mode = 'body'
        self.protocol = 'chat'
        self.calls = 0
        self.received = asyncio.Event()
        self.closed = asyncio.Event()
        self.eof_at = None
        self.writers = set()

    def reset(self, mode, protocol):
        self.mode, self.protocol = mode, protocol
        self.calls = 0
        self.received.clear()
        self.closed.clear()
        self.eof_at = None

    async def handle(self, reader, writer):
        self.writers.add(writer)
        try:
            head = await reader.readuntil(b'\r\n\r\n')
            first, *headers = head.decode().split('\r\n')
            length = next((int(h.split(':', 1)[1]) for h in headers if h.lower().startswith('content-length:')), 0)
            if length:
                await reader.readexactly(length)
            if first.startswith('GET '):
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}')
                await writer.drain()
                return
            self.calls += 1
            mode, protocol = self.mode, self.protocol
            self.received.set()
            if mode != 'headers':
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n')
                if mode == 'preflight':
                    chunks = [b': awaiting model identity\n\n']
                else:
                    chunks = events(protocol, stop=(mode == 'automatic'))
                    if mode != 'automatic':
                        # No native terminal event: only the body prefix.
                        chunks = chunks[:2 if protocol == 'chat' else 3 if protocol == 'messages' else 2]
                for chunk in chunks:
                    writer.write(f'{len(chunk):x}\r\n'.encode() + chunk + b'\r\n')
                await writer.drain()
            # The provider never finishes. EOF must come from cancellation.
            rest = await reader.read()
            assert not rest
            self.eof_at = time.monotonic()
            self.closed.set()
        except (ConnectionResetError, BrokenPipeError):
            self.eof_at = time.monotonic()
            self.closed.set()
        finally:
            writer.close()
            await writer.wait_closed()
            self.writers.discard(writer)



def set_fixture(config, feature_flags):
    """Pool chooses channels; Stream owns every feature setting in this test."""
    config = copy.deepcopy(config)
    routing = config['routing']
    defaults = routing.pop('defaults', {})
    flags = {k: defaults.pop(k) for k in ('early_stop','streaming','filters') if k in defaults}
    flags.update(defaults.pop('runtime', {}))
    policies = {'schema':1,'defaults':defaults,'profiles':routing.pop('profiles',{}),
                'model_rules':routing.pop('model_rules',[]),'channels':{},
                'account_overrides':{},'dedupe_credentials':routing.pop('dedupe_credentials',{})}
    for account in config.get('accounts', []):
        if 'policy' in account:policies['account_overrides'][account['id']] = account.pop('policy')
    for channel in routing['channels']:
        cp = {k:channel.pop(k) for k in ('profiles','policy') if k in channel}
        cp['routes'] = {}
        for route in channel['routes']:
            cp['routes'][route['id']] = {k:route.pop(k) for k in ('profiles','policy','dedupe') if k in route}
        policies['channels'][channel['id']] = cp
    # Test fixtures explicitly assign the requested mode to their channels;
    # production no longer accepts a global conversion default.
    settings = flags.get('streaming', {})
    mode = settings.pop('force_upstream', 'auto')
    convert = settings.pop('nonstream_to_stream', False)
    if convert and mode == 'auto': mode = 'stream'
    for channel in policies['channels'].values():
        channel.setdefault('policy', {}).setdefault('streaming', {})['force_upstream'] = mode
    flags['channel_policies'] = policies
    routing['schema'] = 2
    feature_flags.clear();feature_flags.update(flags)
    acc.state = acc.PoolState(config=config)

async def main():
    upstream = Upstream()
    fake = await asyncio.start_server(upstream.handle, '127.0.0.1', 0)
    upstream_port = fake.sockets[0].getsockname()[1]
    acc.CONFIG_PATH = Path('/tmp/chain-acc-config-unused.json')
    for name in ('reload_config_if_needed', 'apply_daily_resets', 'apply_limited_recovery_probes', 'persist_config'):
        setattr(acc, name, AsyncMock())
    acc_port, acc_server, acc_task = await server(acc.app)
    feature_flags = {}
    lookup = lambda *keys: {'enabled': True, 'base_url': f'http://127.0.0.1:{acc_port}', 'api_key': 'service-test'} if keys == ('account_routing',) else feature_flags.get(keys[0])
    deps = ChannelGatewayTests().deps({})
    deps.resolve_account_route = lambda model, **kw: account_channels.resolve(lookup, model, **kw)
    deps.build_account_dedupe = lambda *args: account_channels.dedupe_key(lookup, *args)
    coalescer = ExactRequestCoalescer(ttl=60, log=print, shared_state=SharedRuntimeState('/tmp/chain-shared'))
    deps.run_exact_nonstream_once = coalescer.run
    web = FastAPI()
    runtime = RuntimeLifecycle(instance='chain-test', control_file='', state_dir='/tmp/chain-state', log=print)
    web.add_middleware(LifecycleMiddleware, runtime=runtime)
    @web.post('/v1/chat/completions')
    async def chat(request: Request):
        return await protocol_entry(request, deps, lambda: chat_completions(request, deps), interval=.05)
    @web.post('/v1/messages')
    async def messages(request: Request):
        return await protocol_entry(request, deps, lambda: route_protocol_request(request, deps, protocol='anthropic'), interval=.05)
    @web.post('/v1/responses')
    async def responses(request: Request):
        return await protocol_entry(request, deps, lambda: route_protocol_request(request, deps, protocol='responses'), interval=.05)
    stream_port, stream_server, stream_task = await server(web)
    results = []
    try:
        for protocol in ('chat', 'messages', 'responses'):
            for public in ('chat/completions', 'messages', 'responses'):
                for mode in ('automatic', 'body', 'headers', 'preflight', 'deadline'):
                    upstream.reset(mode, protocol)
                    account = {'id': 'fixture', 'status': 'ok', 'base_url': f'http://127.0.0.1:{upstream_port}/v1',
                        'api_key': 'upstream-test', 'models': ['*'], 'sort_enabled': False,
                        'auto_reset_enabled': False, 'status_tracking_enabled': False,
                        'model_verification_mode': 'strict' if mode == 'preflight' else 'off'}
                    if mode == 'preflight':
                        account['verify_upstream_model'] = True
                    set_fixture({'accounts': [account], 'routing': {
                        'schema': 1, 'enabled': True, 'client_keys': ['service-test'], 'defaults': policy(streaming={'deadline_seconds': .5 if mode == 'deadline' else 0}),
                        'channels': [{'id': 'channel', 'accounts': ['fixture'], 'routes': [{
                            'id': 'route', 'transport': protocol, 'models': {'test': 'wire-model'}}]}]}}, feature_flags)
                    payload = {'model': 'test', 'stream': True, 'messages': [{'role': 'user', 'content': 'Synthetic'}], 'input': 'Synthetic'}
                    start = time.monotonic()
                    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                        async with client.stream('POST', f'http://127.0.0.1:{stream_port}/v1/{public}', json=payload) as response:
                            assert response.status_code == 200, (protocol, public, mode, response.status_code)
                            await asyncio.wait_for(upstream.received.wait(), 3)
                            if mode in ('automatic', 'deadline'):
                                body = await response.aread()
                                assert (b'visible' if mode == 'automatic' else b'OK') in body, body
                                assert not any(end in body for end in (b'[DONE]', b'response.completed', b'message_stop')), body
                            elif mode == 'body':
                                body = b''
                                async for chunk in response.aiter_bytes():
                                    body += chunk
                                    if b'OK' in body:
                                        break
                            closed_at = time.monotonic()
                        # Client has closed the HTTP body, not read to natural end.
                        await asyncio.wait_for(upstream.closed.wait(), 2)
                    for _ in range(100):
                        if not runtime.active and len(deps.active_stream_registry) == 0:
                            break
                        await asyncio.sleep(0.01)
                    assert not runtime.active, runtime.active
                    assert len(deps.active_stream_registry) == 0
                    assert upstream.calls == 1, (protocol, public, mode, upstream.calls)
                    elapsed = max(0, upstream.eof_at - closed_at)
                    assert elapsed < 2, elapsed
                    row = {'upstream': protocol, 'public': public, 'case': mode, 'calls': upstream.calls,
                           'eof_after_client_close_ms': round(elapsed * 1000, 2), 'lifecycle_active': 0}
                    results.append(row)
                    print(json.dumps(row), flush=True)
        # JSON upstreams have no stream iterator at the provider boundary.
        # Both client-SSE replay and client-JSON must still propagate cancel.
        for protocol in ('chat', 'messages', 'responses', 'gemini'):
            for public in ('chat/completions', 'messages', 'responses'):
                for downstream_stream, keepalive in ((True, True), (False, True), (False, False)):
                    upstream.reset('headers', protocol)
                    settings = policy(streaming={'force_upstream': 'json', 'json_keepalive': keepalive, 'exact_dedupe': True})
                    account = {'id': 'fixture', 'status': 'ok', 'base_url': f'http://127.0.0.1:{upstream_port}/v1',
                        'api_key': 'test', 'models': ['*'], 'sort_enabled': False,
                        'auto_reset_enabled': False, 'status_tracking_enabled': False}
                    set_fixture({'accounts': [account], 'routing': {
                        'schema': 1, 'enabled': True, 'client_keys': ['service-test'], 'defaults': settings,
                        'channels': [{'id': 'channel', 'accounts': ['fixture'], 'routes': [{
                            'id': 'route', 'transport': protocol, 'models': {'test': 'wire-model'}}]}]}}, feature_flags)
                    payload = {'model': 'test', 'stream': downstream_stream,
                        'messages': [{'role': 'user', 'content': 'Synthetic'}], 'input': 'Synthetic'}
                    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                        request_task = asyncio.create_task(client.post(f'http://127.0.0.1:{stream_port}/v1/{public}', json=payload))
                        await asyncio.wait_for(upstream.received.wait(), 3)
                        closed_at = time.monotonic()
                        request_task.cancel()
                        try:
                            await request_task
                        except asyncio.CancelledError:
                            pass
                        await asyncio.wait_for(upstream.closed.wait(), 2)
                    for _ in range(100):
                        if not runtime.active and len(deps.active_stream_registry) == 0:
                            break
                        await asyncio.sleep(.01)
                    assert not runtime.active and len(deps.active_stream_registry) == 0
                    assert not coalescer._inflight
                    assert upstream.calls == 1
                    elapsed = max(0, upstream.eof_at - closed_at)
                    assert elapsed < 2
                    row = {'upstream': protocol, 'public': public, 'case': 'json-upstream-cancel',
                           'downstream_stream': downstream_stream, 'keepalive': keepalive,
                           'calls': 1, 'eof_after_client_close_ms': round(elapsed * 1000, 2), 'lifecycle_active': 0}
                    results.append(row)
                    print(json.dumps(row), flush=True)
        print('CHAIN_PASS', len(results), 'real socket cases', flush=True)
    finally:
        stream_server.should_exit = True
        acc_server.should_exit = True
        await asyncio.gather(stream_task, acc_task)
        fake.close()
        await fake.wait_closed()
        for writer in list(upstream.writers):
            writer.close()


if __name__ == '__main__':
    asyncio.run(main())
