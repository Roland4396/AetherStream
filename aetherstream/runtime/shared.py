"""Cross-release session and exact-response state using DiskCache/SQLite.

The response lock is kernel-backed and held until the upstream operation ends;
there is no expiring lease that could authorize a duplicate paid request.
"""
import asyncio
import hashlib
from pathlib import Path
import time
import uuid

from diskcache import Cache
from filelock import FileLock, Timeout


class SharedRuntimeState:
    def __init__(self, directory):
        self.directory = str(Path(directory) / 'cache')
        self.locks = Path(directory) / 'request-locks'
        self.locks.mkdir(parents=True, exist_ok=True, mode=0o700)
        with Cache(self.directory) as cache, cache.transact():
            schema = cache.get('__state_schema__')
            if schema is None:
                cache.set('__state_schema__', 1)
            elif schema != 1:
                raise RuntimeError('Incompatible shared runtime state schema')

    def session(self, key, ttl, now=None):
        now = time.time() if now is None else now
        with Cache(self.directory) as cache, cache.transact():
            state = cache.get('session:'+key)
            mode = 'reused'
            if not state or now - state['last_used_at'] >= ttl or now < state['last_used_at']:
                state = {'session_id': str(uuid.uuid4()), 'last_used_at': now}
                mode = 'new'
            state['last_used_at'] = now
            cache.set('session:'+key, state, expire=ttl)
            return state['session_id'], mode

    def retire_session(self, key, expected_session_id):
        """Retire only the session that actually received the upstream refusal.

        A late response from another in-flight request must not retire a newer
        session created by a different worker or rolling-release instance.
        """
        with Cache(self.directory) as cache, cache.transact():
            current = cache.get('session:'+key)
            if not current or current.get('session_id') != expected_session_id:
                return False
            cache.delete('session:'+key)
            return True

    def _get(self, key):
        with Cache(self.directory) as cache:
            return cache.get('response:'+key)

    def _set(self, key, value, ttl):
        with Cache(self.directory) as cache:
            cache.set('response:'+key, value, expire=max(ttl, 0.01))

    async def exact_response(self, key, ttl, runner):
        # Bounded lock files; a rare hash collision serializes requests safely.
        stripe = int(hashlib.sha256(key.encode()).hexdigest()[:8],16) % 1024
        lock = FileLock(self.locks / f'{stripe}.lock', thread_local=False, mode=0o600)
        while True:
            cached = await asyncio.to_thread(self._get, key)
            if cached is not None:
                return cached, True
            try:
                lock.acquire(timeout=0)
                break
            except Timeout:
                await asyncio.sleep(.1)
        try:
            cached = await asyncio.to_thread(self._get, key)
            if cached is not None:
                return cached, True
            result = await runner()
            await asyncio.to_thread(self._set, key, result, ttl)
            return result, False
        finally:
            lock.release()

    def stream_get(self, caller):
        with Cache(self.directory) as cache:
            return cache.get('stream:'+caller)

    def stream_register(self, caller, metadata, supersede):
        with Cache(self.directory) as cache, cache.transact():
            previous = cache.get('stream:'+caller)
            if supersede and previous and previous['model'] == metadata['model']:
                cache.set('cancel:'+previous['trace_id'], True, expire=86400)
            cache.set('stream:'+caller, metadata)
            return previous

    def stream_cancelled(self, trace):
        with Cache(self.directory) as cache:
            return bool(cache.get('cancel:'+trace, False))

    def stream_release(self, caller, trace):
        with Cache(self.directory) as cache, cache.transact():
            current = cache.get('stream:'+caller)
            if current and current['trace_id'] == trace:
                cache.delete('stream:'+caller)
            cache.delete('cancel:'+trace)

    def replace_token(self, key, token):
        with Cache(self.directory) as cache:
            cache.set('generation:'+key, token)

    def current_token(self, key):
        with Cache(self.directory) as cache:
            return cache.get('generation:'+key)

    def release_token(self, key, token):
        with Cache(self.directory) as cache, cache.transact():
            if cache.get('generation:'+key) == token:
                cache.delete('generation:'+key)
