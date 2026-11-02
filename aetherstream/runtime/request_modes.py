"""Revision-checked, process-safe edits to the existing runtime config only."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

from aetherstream.routing.policy_config import global_policy, validate_config
from aetherstream.routing.request_mode import set_mode
from aetherstream.runtime.flags import read_flags


class ConfigConflict(ValueError):
    pass


def validate_flags(flags: dict) -> None:
    validate_config(flags['channel_policies'], global_policy(lambda key: flags.get(key)))


class RequestModeStore:
    def __init__(self, runtime_flags, *, backup_dir: str):
        self.runtime_flags = runtime_flags
        self.path = Path(runtime_flags.path)
        self.backup_dir = Path(backup_dir)

    def snapshot(self):
        flags, raw, _ = read_flags(str(self.path))
        validate_flags(flags)
        return flags, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _write(fd: int, raw: bytes):
        offset = 0
        while offset < len(raw):
            written = os.pwrite(fd, raw[offset:], offset)
            if written <= 0:
                raise OSError('Incomplete runtime config write')
            offset += written
        os.ftruncate(fd, len(raw))
        os.fsync(fd)

    def _backup(self, revision: str, raw: bytes):
        self.backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.backup_dir / (revision + '.json')
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if hashlib.sha256(path.read_bytes()).hexdigest() != revision:
                raise OSError('Existing config backup is incomplete')
            return
        with os.fdopen(fd, 'wb') as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())

    def update(self, *, revision: str, channel: str, model: str | None, mode: str) -> str:
        # This is a feature override, not another Pool admission list. A channel
        # absent from one caller's catalog may still be a valid dynamic route.
        # Preserve this inode: existing release slots bind-mount this exact file.
        # Kernel flock, not a TTL, serializes edits across application slots.
        with self.path.open('r+b', buffering=0) as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConfigConflict('Another settings write is in progress; reload and retry') from exc
            raw = file.read()
            current_revision = hashlib.sha256(raw).hexdigest()
            if not isinstance(revision, str) or revision != current_revision:
                raise ConfigConflict('Settings changed; reload before saving')
            original = json.loads(raw)
            validate_flags(original)
            updated = set_mode(original, channel=channel, model=model, mode=mode)
            validate_flags(updated)
            if updated == original:
                return current_revision
            new = (json.dumps(updated, ensure_ascii=False, indent=2) + '\n').encode()
            stat = os.fstat(file.fileno())
            if os.stat(self.path).st_ino != stat.st_ino:
                raise ConfigConflict('Config file was replaced; reload before saving')
            self._backup(current_revision, raw)
            try:
                self._write(file.fileno(), new)
            except OSError:
                self._write(file.fileno(), raw)
                raise
        self.runtime_flags.invalidate()
        # Only old full-config backups are pruned, never the current config.
        try:
            backups = sorted(self.backup_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in backups[10:]:
                if len(old.stem) == 64 and all(c in '0123456789abcdef' for c in old.stem):
                    old.unlink(missing_ok=True)
        except OSError:
            pass  # A successful config commit is not an error if pruning fails.
        return hashlib.sha256(new).hexdigest()
