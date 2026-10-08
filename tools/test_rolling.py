#!/usr/bin/env python3
"""Isolated, no-inference Nginx promotion/rollback acceptance (Docker required)."""
import argparse
import concurrent.futures
import ipaddress
import tempfile
from types import SimpleNamespace
import uuid
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import requests
import websocket

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--image',required=True,help='Locally built candidate image; never calls a model')
args=parser.parse_args()
WORK=Path(tempfile.mkdtemp(prefix='stream-release-test-'))
print('Test artifacts: '+str(WORK),flush=True)
suffix=uuid.uuid4().hex[:8]
for sub in ['control','state','logs','secrets']: (WORK/sub).mkdir(exist_ok=True,mode=0o700)
TOKEN='test-only-'+('x'*55);(WORK/'secrets/admin.token').write_text(TOKEN)
for slot in ['blue','green']:
    (WORK/('flags-'+slot+'.json')).write_text('{"quota_keeper":{"enabled":false}}')
image=args.image;net='stream-test-'+suffix;names=['stream-test-'+suffix+'-'+slot for slot in ['blue','green','front']]
def cmd(*args,check=True):return subprocess.run(args,capture_output=True,text=True,check=check)
cmd('docker','network','create','--internal',net)
subnet=ipaddress.ip_network(json.loads(cmd('docker','network','inspect',net).stdout)[0]['IPAM']['Config'][0]['Subnet'])
front_ip=str(subnet[10])
spec=importlib.util.spec_from_file_location('release',ROOT/'stream-proxy/tools/release.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
cli.SLOTS={'blue':str(subnet[11]),'green':str(subnet[12])};cli.CONTROL=WORK/'control';cli.STATE=WORK/'state';cli.JOURNAL=cli.STATE/'release.json';cli.TOKEN_FILE=WORK/'secrets/admin.token'
def mapped_command(*args,check=True):
    args=tuple({'stream-proxy':names[2],'stream-blue':names[0],'stream-green':names[1]}.get(x,x) for x in args)
    return cmd(*args,check=check)
cli.command=mapped_command
entry='http://'+front_ip+':3002';session=requests.Session();session.trust_env=False
headers={'Authorization':'Bearer '+TOKEN,'Connection':'close'}
def ready():
    r=session.get(entry+'/ready',timeout=3);r.raise_for_status();return r.json()
cli.entry_ready=ready
cli.atomic(cli.CONTROL/'active.json',{'active':'blue','state_schema':1})
cli.atomic(cli.CONTROL/'upstream.conf',cli.upstream('blue'))
nginx=(ROOT/'stream-proxy/deploy/nginx/nginx.conf').read_text().replace('worker_processes auto;','worker_processes 2;')
(WORK/'nginx.conf').write_text(nginx)
records=[];pool=concurrent.futures.ThreadPoolExecutor(max_workers=4);sockets=[]
def collect(response):
    records=[]
    for line in response.iter_lines(chunk_size=1):
        if line.startswith(b'data: '):records.append(line[6:].decode())
    return records
try:
    for slot in ['blue','green']:
        cmd('docker','run','-d','--name','stream-test-'+suffix+'-'+slot,'--network',net,'--ip',cli.SLOTS[slot],
            '-v',str(WORK/'control')+':/app/control:ro','-v',str(WORK/'state')+':/app/state',
            '-v',str(WORK/('flags-'+slot+'.json'))+':/app/runtime-flags.json:ro','-v',str(WORK/'secrets')+':/app/secrets:ro',
            '-e','STREAM_INSTANCE_ID='+slot,'-e','STREAM_ACTIVE_CONTROL_FILE=/app/control/active.json',
            '-e','STREAM_LIFECYCLE_STATE_DIR=/app/state','-e','STREAM_SHARED_STATE_DIR=/app/state/shared',
            '-e','STREAM_ADMIN_TOKEN_FILE=/app/secrets/admin.token','-e','STREAM_DIAGNOSTICS_ENABLED=true',image,
            'uvicorn','proxy:app','--host',cli.SLOTS[slot],'--port','3002')
    for _ in range(60):
        try:
            if cli.endpoint('blue')['background_owner'] and cli.endpoint('green')['ready']:break
        except requests.RequestException:pass
        time.sleep(1)
    else:raise RuntimeError('fixture apps did not become ready')
    cmd('docker','run','-d','--name',names[2],'--network',net,'--ip',front_ip,
        '-v',str(WORK/'nginx.conf')+':/etc/nginx/nginx.conf:ro',
        '-v',str(WORK/'control')+':/etc/nginx/stream-control:ro',
        'nginx@sha256:5616878291a2eed594aee8db4dade5878cf7edcb475e59193904b198d9b830de')
    for _ in range(50):
        try:
            if ready()['instance']=='blue':break
        except requests.RequestException:pass
        time.sleep(.2)
    assert ready()['instance']=='blue'
    journal={'state_schema':1,'phase':'stable','active':'blue','previous':None,'draining':None}
    cli.atomic(cli.JOURNAL,journal)
    # Bad readiness is observable before any route change.
    (WORK/'flags-green.json').write_text('{bad')
    assert session.get('http://'+cli.SLOTS['green']+':3002/ready').status_code==503
    assert ready()['instance']=='blue'
    real_time=cli.time
    cli.time=SimpleNamespace(sleep=lambda _:None,monotonic=time.monotonic)
    try:
        try:
            cli.candidate_ready('green',ready()['contract'])
        except RuntimeError as exc:
            assert 'readiness' in str(exc)
        else:
            raise AssertionError('Bad candidate was accepted')
    finally:
        cli.time=real_time
    assert ready()['instance']=='blue'
    records.append({'test':'bad_readiness','candidate_rejected':True,'active_unaffected':True})
    (WORK/'flags-green.json').write_text('{"quota_keeper":{"enabled":false}}')
    assert ready()['instance']=='blue'
    assert session.get(entry+'/admin/runtime').status_code==401
    response=session.get(entry+'/admin/runtime/probe/stream?count=100&interval_ms=100',headers=headers,stream=True,timeout=(3,15))
    response.raise_for_status();stream=pool.submit(collect,response)
    ws=websocket.create_connection(entry.replace('http:','ws:')+'/admin/runtime/probe/ws',header=['Authorization: Bearer '+TOKEN],timeout=10);sockets.append(ws)
    ws.send('before');assert json.loads(ws.recv())=={'instance':'blue','echo':'before'}
    result=cli.promote(journal,'green',1)
    assert result is False
    assert cli.inspect('blue')['State']['Running']
    assert ready()['instance']=='green'
    ws.send('after promotion');assert json.loads(ws.recv())['instance']=='blue'
    seq=stream.result(15);assert seq[-1]=='[DONE]';assert [json.loads(x)['seq'] for x in seq[:-1]]==list(range(100))
    assert {json.loads(x)['instance'] for x in seq[:-1]}=={'blue'}
    records.append({'test':'promote','sse_chunks':len(seq)-1,'sse_terminal':True,'old_ws_survived':True,'timeout_kept_old_alive':True,'new_entry':'green'})
    # Roll back before old WebSocket drains; both generations remain valid.
    response2=session.get(entry+'/admin/runtime/probe/stream?count=60&interval_ms=100',headers=headers,stream=True,timeout=(3,15));stream2=pool.submit(collect,response2)
    ws2=websocket.create_connection(entry.replace('http:','ws:')+'/admin/runtime/probe/ws',header=['Authorization: Bearer '+TOKEN],timeout=10);sockets.append(ws2)
    ws2.send('green before rollback');assert json.loads(ws2.recv())['instance']=='green'
    assert cli.promote(journal,'blue',1,rollback=True) is False
    assert ready()['instance']=='blue'
    ws2.send('green after rollback');assert json.loads(ws2.recv())['instance']=='green'
    ws.send('blue after rollback');assert json.loads(ws.recv())['instance']=='blue'
    seq2=stream2.result(15);assert seq2[-1]=='[DONE]';assert [json.loads(x)['seq'] for x in seq2[:-1]]==list(range(60))
    records.append({'test':'rollback_while_draining','sse_chunks':len(seq2)-1,'both_websockets_survived':True,'new_entry':'blue'})
    for socket in sockets:socket.close()
    sockets.clear()
    assert cli.drain(journal,30)
    assert not cli.inspect('green')['State']['Running']
    assert ready()['instance']=='blue'
    assert cli.endpoint('blue')['active_http']==0 and cli.endpoint('blue')['active_websockets']==0
    assert cli.endpoint('blue')['background_owner']
    records.append({'test':'drain','old_stopped_only_after_release':True,'owner':'blue'})
    (WORK/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records),flush=True)
finally:
    for socket in sockets:socket.close()
    pool.shutdown(wait=True)
    for name in names:
        out=cmd('docker','logs',name,check=False)
        (WORK/(name+'.log')).write_text(out.stdout+out.stderr)
    # Fixture requests have completed; no real upstreams were reachable.
    for name in reversed(names):cmd('docker','stop','--time','30',name,check=False);cmd('docker','rm',name,check=False)
    cmd('docker','network','rm',net,check=False)
