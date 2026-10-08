"""Private atomic state and a cross-process lock shared with the legacy timer."""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile


class StateError(Exception):
    pass


class StateStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.path = self.directory / 'state.json'

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.directory / 'run.lock', os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield True
        finally:
            os.close(fd)

    def load(self) -> dict:
        if not self.path.exists():
            return {'accounts': {}}
        try:
            state = json.loads(self.path.read_text())
            if not isinstance(state, dict) or not isinstance(state.get('accounts'), dict):
                raise ValueError('Invalid state root')
            for key, row in state['accounts'].items():
                if not isinstance(key, str) or not isinstance(row, dict):
                    raise ValueError('Invalid account row')
                for field in ('next_at', 'guard_until', 'last_trigger_at'):
                    if field in row and (isinstance(row[field], bool) or
                            not isinstance(row[field], (int, float)) or
                            not math.isfinite(row[field]) or row[field] < 0):
                        raise ValueError('Invalid deadline')
            return state
        except (ValueError, TypeError, OSError) as exc:
            # Never replace unreadable/corrupt state with an empty schedule.
            raise StateError('Keeper state cannot be safely read') from exc

    def save(self, state: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        previous = self.path.stat() if self.path.exists() else None
        fd, name = tempfile.mkstemp(prefix='.state-', suffix='.tmp', dir=self.directory)
        try:
            with os.fdopen(fd, 'w') as file:
                os.fchmod(file.fileno(), 0o600)
                # A root container must not break the ubuntu-owned rollback CLI.
                if previous is not None and os.geteuid() == 0:
                    os.fchown(file.fileno(), previous.st_uid, previous.st_gid)
                json.dump(state, file, ensure_ascii=False, indent=2, allow_nan=False)
                file.write('\n')
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)
