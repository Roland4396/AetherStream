#!/usr/bin/env python3
"""Safe single-host blue/green releases. Drain timeout never kills requests."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import requests

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / 'stream-proxy/deploy'
CONTROL, STATE = DEPLOY / 'control', DEPLOY / 'state'
JOURNAL = STATE / 'release.json'
SLOTS = {'blue': '172.28.16.11', 'green': '172.28.16.12'}
TOKEN_FILE = Path('/home/ubuntu/.config/stream-proxy/quota-keeper-secrets/admin.token')
SESSION = requests.Session()
SESSION.trust_env = False


def event(name, **data):
    print(json.dumps({'event': name, **data}, ensure_ascii=False), flush=True)


def command(*args, check=True):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=check)


def atomic(path, value):
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        os.chmod(temp, 0o600)
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fd=os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def read_journal():
    value = json.loads(JOURNAL.read_text())
    if value.get('active') not in SLOTS or value.get('state_schema') != 1:
        raise RuntimeError('Unknown release journal; no action taken')
    return value


def inspect(slot):
    return json.loads(command('docker', 'inspect', 'stream-'+slot).stdout)[0]


def headers():
    return {'Authorization': 'Bearer '+TOKEN_FILE.read_text().strip(), 'Connection': 'close'}


def endpoint(slot, path='/admin/runtime'):
    response = SESSION.get(f'http://{SLOTS[slot]}:3002{path}', headers=headers(), timeout=5)
    response.raise_for_status()
    return response.json()


def entry_ready():
    response=SESSION.get('http://127.0.0.1:3002/ready',headers={'Connection':'close'},timeout=5)
    response.raise_for_status(); return response.json()


def compatible(old, new):
    return (old['lifecycle'] == new['lifecycle'] == 1 and old['api'] == new['api'] and
            old['state_read_min'] <= new['state_write'] <= old['state_read_max'] and
            new['state_read_min'] <= old['state_write'] <= new['state_read_max'] and
            old['state_write'] == new['state_write'] == 1)


def workers():
    output=command('docker','top','stream-proxy','-eo','pid,args').stdout
    return [line.split()[0] for line in output.splitlines() if 'nginx: worker process' in line]


def upstream(slot):
    # Old workers retain their old upstream while new workers use this address.
    return f'upstream stream_active {{ server {SLOTS[slot]}:3002; keepalive 64; }}\n'


def switch_route(slot):
    path=CONTROL/'upstream.conf'; before=path.read_text()
    atomic(path,upstream(slot))
    try:
        command('docker','exec','stream-proxy','nginx','-t')
        command('docker','exec','stream-proxy','nginx','-s','reload')
        for _ in range(50):
            try:
                if entry_ready()['instance']==slot: return
            except requests.RequestException: pass
            time.sleep(.2)
        raise RuntimeError('Nginx did not confirm the new route')
    except Exception:
        atomic(path,before)
        command('docker','exec','stream-proxy','nginx','-t')
        command('docker','exec','stream-proxy','nginx','-s','reload')
        raise


def candidate_ready(slot, old_contract, allow_owner=False):
    for _ in range(90):
        try:
            data=endpoint(slot,'/ready')
            if data.get('ready') and data.get('instance')==slot:
                if not compatible(old_contract,data['contract']):
                    raise RuntimeError('Incompatible API/state contract; promotion refused')
                validate_candidate_network(slot)
                if endpoint(slot)['background_owner'] and not allow_owner:
                    raise RuntimeError('Standby unexpectedly owns background jobs')
                return data
        except requests.RequestException: pass
        time.sleep(1)
    raise RuntimeError('Candidate failed readiness; active version unchanged')


def validate_candidate_network(slot):
    info=inspect(slot)
    if any(info['NetworkSettings'].get('Ports',{}).values()):
        raise RuntimeError('Candidate must not publish host ports')
    pid=info['State']['Pid']
    lines=command('sudo','-n','nsenter','-t',str(pid),'-n','ss','-Hlnt').stdout.splitlines()
    listeners={line.split()[3] for line in lines if len(line.split())>=5 and line.split()[3].endswith(':3002')}
    if listeners != {SLOTS[slot]+':3002'}:
        raise RuntimeError('Candidate must listen only on its private backplane IP')


def outgoing(slot):
    pid=inspect(slot)['State']['Pid']
    if not pid: return []
    rows=command('sudo','-n','nsenter','-t',str(pid),'-n','ss','-Htn','state','established').stdout
    return [line for line in rows.splitlines() if len(line.split())>=4 and not line.split()[2].endswith(':3002')]


def quiescent(slot):
    data=endpoint(slot)
    return (data['active_http']==0 and data['active_websockets']==0 and
            not data['background_owner'] and not any(data.get('pending_tasks',{}).values()) and not outgoing(slot))


def drain(journal, timeout):
    old=journal.get('draining')
    if not old: return True
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        # A CLI interrupted after docker stop must be safely resumable.
        if not inspect(old)['State']['Running']:
            journal.update(phase='stable',draining=None,old_nginx_workers=[])
            atomic(JOURNAL,journal)
            return True
        # App accounting + Nginx old generation + established upstream sockets.
        if (not set(journal.get('old_nginx_workers',[])).intersection(workers())
                and endpoint(journal['active'])['background_owner'] and quiescent(old)):
            time.sleep(1)
            if quiescent(old):
                event('drain_complete',slot=old)
                command('docker','stop','--time','120','stream-'+old)
                journal.update(phase='stable',draining=None,old_nginx_workers=[])
                atomic(JOURNAL,journal)
                return True
        time.sleep(2)
    event('drain_pending',slot=old,detail='Old instance left alive; run drain later. No request was killed.')
    return False


def promote(journal, target, timeout, rollback=False):
    current=journal['active']
    candidate_ready(target,endpoint(current,'/ready')['contract'], allow_owner=rollback)
    journal.update(phase='switching',target=target,old_nginx_workers=workers())
    atomic(JOURNAL,journal)
    switch_route(target)
    atomic(CONTROL/'active.json',{'active':target,'state_schema':1})
    journal.update(active=target,previous=current,draining=current,phase='draining',target=None)
    atomic(JOURNAL,journal)
    for _ in range(120):
        if endpoint(target)['background_owner'] and not endpoint(current)['background_owner']: break
        time.sleep(1)
    else:
        raise RuntimeError('Background handoff pending; both instances retained for investigation')
    event('promoted',active=target,previous=current,image=inspect(target)['Image'])
    return drain(journal,timeout)


def deploy(image, timeout):
    journal=read_journal()
    if journal.get('phase')!='stable':
        raise RuntimeError('Pending release: finish drain/recover before another deployment')
    current=journal['active']; target='green' if current=='blue' else 'blue'
    existing=command('docker','inspect','stream-'+target,check=False)
    if existing.returncode==0 and json.loads(existing.stdout)[0]['State']['Running']:
        if not quiescent(target): raise RuntimeError('Inactive slot still busy; refusing recreation')
    image_id=json.loads(command('docker','image','inspect',image).stdout)[0]['Id']
    command('docker','tag',image_id,'migration-stream-proxy-'+target+':current')
    command('docker','compose','up','-d','--no-deps','--no-build','--pull','never','stream-'+target)
    if inspect(target)['Image']!=image_id: raise RuntimeError('Unexpected candidate image')
    event('candidate_started',slot=target,image=image_id)
    return promote(journal,target,timeout)


def recover(timeout):
    journal=read_journal()
    if journal['phase']=='switching':
        # Observe which generation is actually serving after an interrupted CLI.
        live=entry_ready()['instance']
        if live not in {journal['active'],journal.get('target')}:
            raise RuntimeError('Unexpected live route; manual inspection required')
        old=journal['active'] if live!=journal['active'] else journal.get('target')
        # Converge both the config on disk and worker generation. A crash before
        # HUP otherwise leaves the recorded workers serving forever.
        switch_route(live)
        atomic(CONTROL/'active.json',{'active':live,'state_schema':1})
        journal.update(active=live,previous=old,draining=old,phase='draining',target=None)
        atomic(JOURNAL,journal)
    return drain(journal,timeout)


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','deploy','rollback','drain','recover'])
    parser.add_argument('--image')
    parser.add_argument('--wait',type=float,default=1800,help='Drain deadline; expiry never kills requests')
    args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (STATE/'deploy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='status':
            event('status',journal=read_journal(),entry=entry_ready()); return
        if args.action=='deploy':
            if not args.image: parser.error('--image is required')
            ok=deploy(args.image,args.wait)
        elif args.action=='rollback':
            j=read_journal()
            if j['phase'] not in {'stable','draining'}:
                raise RuntimeError('Recover the interrupted route switch before rollback')
            target=j.get('previous')
            if target not in SLOTS: raise RuntimeError('No retained rollback slot')
            command('docker','start','stream-'+target)
            ok=promote(j,target,args.wait,rollback=True)
        else: ok=recover(args.wait)
        if not ok: raise SystemExit(2)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        event('release_failed',error_type=type(exc).__name__,detail=str(exc) if isinstance(exc,RuntimeError) else 'Inspect local release state; no forced cleanup performed.')
        raise SystemExit(1)
