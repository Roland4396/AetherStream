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

from aetherstream.runtime.response_cache import response_can_replay


class SharedRuntimeState:
    def __init__(self, directory):
        self.directory = str(Path(directory) / 'cache')
        self.locks = Path(directory) / 'request-locks'
        self.locks.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.waiter_locks = Path(directory) / 'response-waiters'
        self.waiter_locks.mkdir(parents=True, exist_ok=True, mode=0o700)
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

    def claim_completion_warning(self, key, threshold):
        """Consume exactly one warning atomically, across release processes.

        New prefixed keys are additive to schema 1. No expiry, state migration,
        provider call, automatic retry or shared-data deletion is involved.
        """
        with Cache(self.directory) as cache, cache.transact():
            name = 'normal-completion-guard:v1:' + key
            if cache.get(name, 0) < threshold:
                return False
            cache.set(name, 0)
            return True

    def record_completion(self, key, normal, threshold):
        with Cache(self.directory) as cache, cache.transact():
            name = 'normal-completion-guard:v1:' + key
            streak = min(threshold, cache.get(name, 0) + 1) if normal else 0
            cache.set(name, streak)
            return streak

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

    def _live_response_members(self, members):
        live = []
        for token in members:
            lock = FileLock(self.waiter_locks / (token + '.lock'), thread_local=False, mode=0o600)
            try:
                lock.acquire(timeout=0)
            except Timeout:
                live.append(token)
            else:
                lock.release()
                Path(lock.lock_file).unlink(missing_ok=True)
        return live

    @staticmethod
    def _discard_unsuccessful_response(cache, key):
        cached = cache.get('response:' + key)
        if cached is not None and not response_can_replay(cached):
            cache.delete('response:' + key)

    def join_response(self, key):
        """A live HTTP waiter owns a kernel lock, not an expiring heartbeat.

        Separate UUID files work across container PID namespaces. A crashed
        process releases its locks automatically; observers prune only locks
        they can actually acquire, never a slow but still-live subscriber.
        """
        token = uuid.uuid4().hex
        lock = FileLock(self.waiter_locks / (token + '.lock'), thread_local=False, mode=0o600)
        lock.acquire(timeout=0)
        try:
            with Cache(self.directory) as cache, cache.transact():
                members = self._live_response_members(cache.get('response-waiters:' + key, []))
                if not members:
                    # Also evict empty responses written by an older release,
                    # or left behind by a crashed cohort. Never evict while
                    # live waiters still need that result: doing so would make
                    # a concurrent follower send another paid request.
                    self._discard_unsuccessful_response(cache, key)
                cache.set('response-waiters:' + key, [*members, token])
        except BaseException:
            lock.release()
            Path(lock.lock_file).unlink(missing_ok=True)
            raise
        return token, lock

    def leave_response(self, key, lease):
        token, lock = lease
        try:
            with Cache(self.directory) as cache, cache.transact():
                members = [x for x in cache.get('response-waiters:' + key, []) if x != token]
                if members:
                    cache.set('response-waiters:' + key, members)
                else:
                    cache.delete('response-waiters:' + key)
                    self._discard_unsuccessful_response(cache, key)
        finally:
            lock.release()
            Path(lock.lock_file).unlink(missing_ok=True)

    def response_waiters(self, key):
        with Cache(self.directory) as cache, cache.transact():
            members = cache.get('response-waiters:' + key, [])
            live = self._live_response_members(members)
            if live:
                cache.set('response-waiters:' + key, live)
            else:
                cache.delete('response-waiters:' + key)
            return len(live)

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
