#!/usr/bin/env python3
"""Run all regression tests in a candidate image without network or live secrets."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--image',required=True)
parser.add_argument('--flags',type=Path,help='Read only early-stop settings from this file; never mount the original')
args=parser.parse_args()
source=args.flags or (ROOT/'runtime-flags.json' if (ROOT/'runtime-flags.json').exists() else ROOT/'runtime-flags.example.json')
config=json.loads(source.read_text())
with tempfile.TemporaryDirectory(prefix='stream-offline-') as directory:
    flags=Path(directory)/'runtime-flags.json'
    flags.write_text(json.dumps({'early_stop':config.get('early_stop',{}),'quota_keeper':{'enabled':False}}))
    flags.chmod(0o600)
    command=['docker','run','--rm','--network','none']
    for src,dst in [(ROOT/'tests','/app/tests'),(ROOT/'tools','/app/tools'),(ROOT/'deploy/nginx','/app/deploy/nginx'),(flags,'/app/runtime-flags.json')]:
        command+=['-v',f'{src}:{dst}:ro']
    command += [args.image,'python','-m','unittest','discover','-s','tests','-v']
    raise SystemExit(subprocess.call(command))
