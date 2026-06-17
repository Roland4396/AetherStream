"""Hot-reloaded runtime flag file support."""

import json
import os
import time
from collections.abc import Callable
from typing import Any


class RuntimeFlags:
    """Small polling JSON store for runtime switches.

    The file is intentionally polled instead of watched so it works inside simple
    containers without extra filesystem dependencies.
    """

    def __init__(self, *, path: str, poll_sec: float, log: Callable[[str], None]):
        self.path = path
        self.poll_sec = poll_sec
        self._log = log
        self._cache: dict[str, Any] | None = None
        self._cache_mtime: float | None = None
        self._checked_at = 0.0
        self._error_state: object = None

    def load(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._checked_at < self.poll_sec and isinstance(self._cache, dict):
            return self._cache
        self._checked_at = now

        if not self.path:
            self._cache = {}
            self._cache_mtime = None
            self._error_state = None
            return self._cache

        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            if self._error_state != ('missing', None):
                self._log(f'Runtime flags file not found, using env defaults: {self.path}')
                self._error_state = ('missing', None)
            self._cache = {}
            self._cache_mtime = None
            return self._cache
        except Exception as e:
            marker = ('stat_error', str(e))
            if self._error_state != marker:
                self._log(f'Failed to stat runtime flags file {self.path}: {e}')
                self._error_state = marker
            return self._cache if isinstance(self._cache, dict) else {}

        if self._cache_mtime == stat.st_mtime and isinstance(self._cache, dict):
            return self._cache

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError('top-level JSON must be an object')
            self._cache = loaded
            self._cache_mtime = stat.st_mtime
            self._error_state = None
            self._log(f'Runtime flags loaded from {self.path}')
            return self._cache
        except Exception as e:
            marker = ('load_error', stat.st_mtime, str(e))
            if self._error_state != marker:
                self._log(f'Failed to load runtime flags from {self.path}: {e}')
                self._error_state = marker
            if isinstance(self._cache, dict):
                return self._cache
            return {}

    def lookup(self, *keys: str) -> Any:
        current: Any = self.load()
        for key in keys:
            if not isinstance(current, dict):
                return None
            if key not in current:
                return None
            current = current[key]
        return current
