"""Exact-request coalescing for upstreams that cannot stream keepalives."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any


class ExactRequestCoalescer:
    def __init__(self, *, ttl: float, log: Callable[[str], None]):
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
        expired = [
            key
            for key, item in self._recent.items()
            if now - float(item.get('stored_at') or 0.0) > self.ttl
        ]
        for key in expired:
            self._recent.pop(key, None)

    async def _execute(
        self,
        *,
        key: str,
        trace_id: str,
        upstream_label: str,
        runner: Callable[[], Awaitable[dict]],
    ) -> dict:
        try:
            result = await runner()
        except BaseException:
            async with self._lock:
                current = self._inflight.get(key)
                if current and current.get('task') is asyncio.current_task():
                    self._inflight.pop(key, None)
            raise

        async with self._lock:
            current = self._inflight.get(key)
            if current and current.get('task') is asyncio.current_task():
                self._inflight.pop(key, None)
                self._recent[key] = {
                    'payload': copy.deepcopy(result),
                    'stored_at': time.monotonic(),
                    'trace_id': trace_id,
                    'upstream': upstream_label,
                }
                self._log(
                    f"[TRACE {trace_id}] nonstream_dedupe_store "
                    f"upstream={upstream_label} key={key[:16]} "
                    f"waiters={current.get('waiters', 0)}"
                )
        return result

    async def run(
        self,
        *,
        dedupe_key: str,
        trace_id: str,
        upstream_label: str,
        runner: Callable[[], Awaitable[dict]],
    ) -> tuple[dict, bool]:
        """Return ``(payload, shared)`` for an exact non-stream request."""
        now = time.monotonic()
        leader = False

        async with self._lock:
            self._cleanup_recent(now)
            recent = self._recent.get(dedupe_key)
            if recent:
                age = now - float(recent.get('stored_at') or now)
                self._log(
                    f"[TRACE {trace_id}] nonstream_dedupe_recent_hit "
                    f"upstream={upstream_label} key={dedupe_key[:16]} age={age:.1f}s"
                )
                return copy.deepcopy(recent['payload']), True

            entry = self._inflight.get(dedupe_key)
            if entry is None:
                task = asyncio.create_task(self._execute(
                    key=dedupe_key,
                    trace_id=trace_id,
                    upstream_label=upstream_label,
                    runner=runner,
                ))
                self._inflight[dedupe_key] = {
                    'task': task,
                    'trace_id': trace_id,
                    'started_at': now,
                    'waiters': 0,
                    'upstream': upstream_label,
                }
                leader = True
                self._log(
                    f"[TRACE {trace_id}] nonstream_dedupe_leader "
                    f"upstream={upstream_label} key={dedupe_key[:16]}"
                )
            else:
                task = entry['task']
                entry['waiters'] = int(entry.get('waiters') or 0) + 1
                age = now - float(entry.get('started_at') or now)
                self._log(
                    f"[TRACE {trace_id}] nonstream_dedupe_wait "
                    f"upstream={upstream_label} key={dedupe_key[:16]} "
                    f"leader_trace={entry.get('trace_id')} age={age:.1f}s "
                    f"waiters={entry['waiters']}"
                )

        result = await asyncio.shield(task)
        return copy.deepcopy(result), not leader

    async def state(self) -> dict[str, int]:
        async with self._lock:
            self._cleanup_recent(time.monotonic())
            return {
                'inflight': len(self._inflight),
                'recent': len(self._recent),
            }
