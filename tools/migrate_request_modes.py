#!/usr/bin/env python3
"""Stage a request-mode migration; never overwrite the live input config."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aetherstream.routing.request_mode import migrate_flags


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--keep-inert-legacy', action='store_true', help='Expand phase for old release slots')
args = parser.parse_args()
if args.input.resolve() == args.output.resolve():
    parser.error('output must be a staging file, not the live input')
os.umask(0o077)
flags, report = migrate_flags(json.loads(args.input.read_text()), keep_inert_legacy=args.keep_inert_legacy)
args.output.write_text(json.dumps(flags, ensure_ascii=False, indent=2) + '\n')
args.output.chmod(0o600)
print(json.dumps({'staged': str(args.output), **report}, ensure_ascii=False))
