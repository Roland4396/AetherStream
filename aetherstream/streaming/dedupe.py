"""Exact-request sharing that cancels work when its last live client leaves."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import anyio

from aetherstream.runtime.response_cache import response_can_replay


class ExactRequestCoalescer:
    def __init__(self, *, ttl: float, log: Callable[[str], None], shared_state=None):
        self.shared_state = shared_state
        self.ttl = max(0.0, float(ttl))
        self._log = log
        self._lock = asyncio.Lock()
        self._inflight: dict[str, dict[str, Any]] = {}
        self._recent: dict[str, dict[str, Any]] = {}

    @staticmethod
    def build_key(request_payload: dict) -> str:
        body = json.dumps(request_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(body.encode('utf-8', errors='replace')).hexdigest()

    def _cleanup_recent(self, now: float) -> None:
        for key in [key for key, item in self._recent.items()
                    if now - float(item.get('stored_at') or 0.0) > self.ttl]:
            self._recent.pop(key, None)

    def _has_waiters(self, key, entry):
        if entry['waiters']:
            return True
        return self.shared_state is not None and self.shared_state.response_waiters(key) > 0

    @staticmethod
    def _retrieve_exception(task):
        # A shared task can outlive the local HTTP waiter while another process
        # still consumes it. Its failure must never become an unobserved task.
        if not task.cancelled():
            task.exception()

    async def _execute(self, *, key, trace_id, upstream_label, runner) -> dict:
        operation = None
        try:
            async def work():
                if self.shared_state is not None:
                    return await self.shared_state.exact_response(key, self.ttl, runner)
                return await runner(), False
            operation = asyncio.create_task(work())
            while True:
                # Local last-client cancellation is immediate in run.finally.
                # Only an owner retained for remote clients needs this bounded
                # cross-process observation; no upstream bytes are drained.
                done, _ = await asyncio.wait({operation}, timeout=.025)
                if operation in done:
                    break
                async with self._lock:
                    entry = self._inflight.get(key)
                    if entry and entry['task'] is asyncio.current_task() and not self._has_waiters(key, entry):
                        entry['closing'] = True
                        self._log(f'[TRACE {trace_id}] nonstream_dedupe_cancel_unobserved key={key[:16]} scope=shared')
                        raise asyncio.CancelledError()
            result, cross_shared = await operation
            async with self._lock:
                entry = self._inflight.get(key)
                if entry and entry['task'] is asyncio.current_task():
                    entry['cross_shared'] = cross_shared
                    if response_can_replay(result):
                        self._recent[key] = {'payload': copy.deepcopy(result), 'cross_shared': cross_shared,
                                             'stored_at': time.monotonic(), 'trace_id': trace_id, 'upstream': upstream_label}
                        self._log(f'[TRACE {trace_id}] nonstream_dedupe_store upstream={upstream_label} '
                                  f'key={key[:16]} waiters={entry["waiters"]}')
                    else:
                        self._log(f'[TRACE {trace_id}] nonstream_dedupe_skip_recent upstream={upstream_label} '
                                  f'key={key[:16]} reason=empty_or_unsuccessful waiters={entry["waiters"]}')
            return result
        finally:
            # Retain the drain barrier until the actual HTTP owners are closed.
            with anyio.CancelScope(shield=True):
                if operation is not None:
                    if not operation.done():
                        operation.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await operation
                async with self._lock:
                    entry = self._inflight.get(key)
                    if entry and entry['task'] is asyncio.current_task():
                        self._inflight.pop(key, None)

    async def run(self, *, dedupe_key: str, trace_id: str, upstream_label: str,
                  runner: Callable[[], Awaitable[dict]]) -> tuple[dict, bool]:
        lease = self.shared_state.join_response(dedupe_key) if self.shared_state is not None else None
        entry = None
        task = None
        leader = False
        registered = False
        try:
            while True:
                async with self._lock:
                    now = time.monotonic()
                    self._cleanup_recent(now)
                    recent = self._recent.get(dedupe_key)
                    if recent and not response_can_replay(recent['payload']):
                        self._recent.pop(dedupe_key, None)
                        recent = None
                    if recent:
                        self._log(f'[TRACE {trace_id}] nonstream_dedupe_recent_hit upstream={upstream_label} '
                                  f'key={dedupe_key[:16]} age={now-recent["stored_at"]:.1f}s')
                        return copy.deepcopy(recent['payload']), True
                    entry = self._inflight.get(dedupe_key)
                    if entry is None:
                        task = asyncio.create_task(self._execute(key=dedupe_key, trace_id=trace_id,
                                                               upstream_label=upstream_label, runner=runner))
                        task.add_done_callback(self._retrieve_exception)
                        entry = {'task': task, 'trace_id': trace_id, 'started_at': now,
                                 'waiters': 0, 'upstream': upstream_label, 'closing': False}
                        self._inflight[dedupe_key] = entry
                        leader = True
                        self._log(f'[TRACE {trace_id}] nonstream_dedupe_leader upstream={upstream_label} key={dedupe_key[:16]}')
                    task = entry['task']
                    if not entry['closing']:
                        entry['waiters'] += 1
                        registered = True
                        if not leader:
                            self._log(f'[TRACE {trace_id}] nonstream_dedupe_wait upstream={upstream_label} '
                                      f'key={dedupe_key[:16]} leader_trace={entry["trace_id"]} waiters={entry["waiters"]}')
                        break
                # A fresh client arriving during cancellation waits for socket
                # cleanup before starting its own request. It never joins an
                # already-cancelled task or overlaps the closing inference.
                await asyncio.wait({task})
            result = await asyncio.shield(task)
            return copy.deepcopy(result), (not leader or bool(entry.get('cross_shared')))
        finally:
            with anyio.CancelScope(shield=True):
                if lease is not None:
                    self.shared_state.leave_response(dedupe_key, lease)
                cancel = None
                if registered:
                    async with self._lock:
                        entry['waiters'] -= 1
                        if self._inflight.get(dedupe_key) is entry and task.done():
                            self._inflight.pop(dedupe_key, None)
                        if (self._inflight.get(dedupe_key) is entry and not task.done()
                                and not entry['closing'] and not self._has_waiters(dedupe_key, entry)):
                            entry['closing'] = True
                            self._log(f'[TRACE {trace_id}] nonstream_dedupe_cancel_unobserved '
                                      f'key={dedupe_key[:16]} scope=local')
                            task.cancel()
                            cancel = task
                    if cancel is not None:
                        with suppress(asyncio.CancelledError, Exception):
                            await cancel
                        async with self._lock:
                            if self._inflight.get(dedupe_key) is entry and cancel.done():
                                self._inflight.pop(dedupe_key, None)

    async def state(self) -> dict[str, int]:
        async with self._lock:
            self._cleanup_recent(time.monotonic())
            return {'inflight': len(self._inflight), 'recent': len(self._recent)}
