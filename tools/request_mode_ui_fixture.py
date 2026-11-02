#!/usr/bin/env python3
"""Loopback-only UI fixture with synthetic catalog, token and config."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
import uvicorn
from aetherstream.api.request_mode_routes import register_routes
from aetherstream.runtime.flags import RuntimeFlags

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', required=True, type=Path)
args = parser.parse_args()
os.umask(0o077)
args.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
path = args.directory / 'flags.json'
path.write_text(json.dumps({'channel_policies': {'schema': 1, 'profiles': {}, 'model_rules': [],
    'channels': {'anti': {'policy': {}}, 'subscription': {'policy': {'streaming': {'force_upstream': 'stream'}}}}},
    'untouched': {'test_marker': 'must survive every save'}}))
token = 'fixture-only-' + secrets.token_hex(24)
token_file = args.directory / 'admin.token'; token_file.write_text(token)
runtime = RuntimeFlags(path=str(path), poll_sec=0, log=lambda _: None)


async def catalog():
    return {'models': [{'id': '[按量]gemini-4-argon', 'owned_by': 'anti'},
                       {'id': '<img src=x onerror=window.PWNED=1>', 'owned_by': 'anti'},
                       {'id': 'codex-example', 'owned_by': 'subscription'},
                       {'id': 'a-new-model', 'owned_by': 'fresh-channel'}]}


async def route(model):
    channel = 'subscription' if model == 'codex-example' else 'anti'
    return {'channel': channel, 'route': 'synthetic-route',
            'upstream_model': 'gemini-4-argon' if model == '[按量]gemini-4-argon' else model}


app = FastAPI()
register_routes(app, runtime_flags=runtime, token_file=str(token_file),
    backup_dir=str(args.directory / 'backups'), catalog=catalog, resolve_route=route, log=lambda _: None)
sock = socket.socket(); sock.bind(('127.0.0.1', 0)); sock.listen(128)
meta = {'url': f'http://127.0.0.1:{sock.getsockname()[1]}', 'token': token, 'config': str(path)}
(args.directory / 'fixture.json').write_text(json.dumps(meta))
print('Loopback UI fixture ready at ' + meta['url'], flush=True)
uvicorn.Server(uvicorn.Config(app, log_level='warning')).run(sockets=[sock])
