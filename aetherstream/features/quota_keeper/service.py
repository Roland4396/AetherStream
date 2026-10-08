"""Native background lifecycle; quota work is independent of chat routing."""

import asyncio
from contextlib import suppress
import json
import time
from typing import Callable

import httpx

from aetherstream.utils.coerce import coerce_bool
from .client import GproxyClient, MODEL, ProbeError
from .policy import FIVE_HOURS, GRACE, RETRY_DELAYS, account_groups, decision, observed, windows
from .state import StateError, StateStore


async def handle_account(api, members, row, persist, emit, *, clock=time.time, may_generate=lambda: True):
    now = clock()
    if row.get("paused") or now < row.get("next_at", 0):
        return
    # Duplicate imports of one email share a single schedule and request guard.
    choices = [(c, await api.cached(c["id"])) for c in members]
    c, cached = max(choices, key=lambda pair: (observed(pair[1]),
                                              pair[0].get("health") != "dead"))
    cid = c["id"]
    row["credential_id"] = cid
    quota = windows(cached)
    action, next_at = decision(quota, now)
    if action == "wait" and not row.get("pending_verify"):
        row.update(next_at=next_at, status="waiting", windows=quota, failures=0)
        persist()
        return
    snapshot = await api.probe(cid)
    now = clock()
    quota = windows(snapshot)
    action, next_at = decision(quota, now)
    row.update(windows=quota, last_probe_at=now, observed_at_ms=observed(snapshot))
    if action == "wait":
        short = quota.get("3p-5h", {})
        confirmed = (short.get("used") or 0) > 0 and (short.get("end") or 0) > now
        if row.get("pending_verify"):
            row["pending_verify"] = False
            row["cycle_confirmed"] = confirmed
            if confirmed:
                row["guard_until"] = short["end"] + GRACE
            emit("cycle_checked", credential_id=cid, confirmed=confirmed,
                 next_at=next_at)
        row.update(next_at=next_at, status="waiting", failures=0)
        persist()
        return
    if action != "unused":
        # Unknown or stale quota must not cause a generation.
        raise ProbeError("Claude window unavailable or reset has not advanced")
    guard = row.get("guard_until", 0)
    if now < guard:
        # A timed-out/crashed request might already have reached the upstream.
        # Query for evidence, but never blindly submit another completion.
        unconfirmed = bool(row.get("pending_verify"))
        row.update(next_at=guard,
                   status="paused_unconfirmed_generation" if unconfirmed else "request_guard",
                   pending_verify=False, cycle_confirmed=False, failures=0)
        if unconfirmed:
            row["paused"] = True
            emit("account_paused", credential_id=cid, reason="generation_cycle_not_confirmed")
        persist()
        return
    if not may_generate():
        persist()
        return
    row.update(last_trigger_at=now, guard_until=now + FIVE_HOURS + GRACE,
               pending_verify=True, cycle_confirmed=False, next_at=now + 60,
               status="submitted", triggers=row.get("triggers", 0) + 1, failures=0)
    persist()  # Durable BEFORE POST: a restart must not duplicate this request.
    try:
        result = await api.generate(cid)
    except (httpx.HTTPError, ValueError, KeyError) as e:
        row.update(status="delivery_uncertain", generation_error=type(e).__name__)
        persist()
        emit("delivery_uncertain", credential_id=cid, error_type=type(e).__name__)
        return
    row["last_generation"] = result
    if result["ok"]:
        row["status"] = "pending_verification"
        emit("claude_triggered", credential_id=cid, **result)
    else:
        # Verification failures/model errors are not retried indefinitely.
        row.update(status="paused_generation_error", paused=True, pending_verify=False)
        emit("account_paused", credential_id=cid, http_status=result["http_status"])
    persist()


def quota_failure(row, now, error, persist, emit):
    count = row.get("failures", 0) + 1
    row.update(failures=count, last_error=type(error).__name__)
    if count > len(RETRY_DELAYS):
        row.update(paused=True, status="paused_quota_error")
    else:
        row.update(next_at=now + RETRY_DELAYS[count - 1], status="quota_retry_wait")
    persist()
    fields = {"credential_id": row.get("credential_id"), "failures": count,
              "paused": bool(row.get("paused")), "error_type": type(error).__name__}
    if isinstance(error, httpx.HTTPStatusError):
        # Status and a fixed operation label diagnose local auth/route errors
        # without logging exception text, tokens, headers, bodies or URLs.
        fields.update(http_status=error.response.status_code, method=error.request.method)
        operation = error.request.url.path.rsplit('/', 1)[-1]
        if operation in ('quota-probe', 'quota', 'reveal', 'login', 'providers', 'credentials'):
            fields['operation'] = operation.replace('-', '_')
    emit("quota_failed", **fields)



class QuotaKeeper:
    def __init__(self, *, lookup: Callable, state_dir: str, base_url: str,
                 credentials_file: str, log: Callable[[str], None],
                 client_factory=None, clock=time.time, shutdown_timeout=100):
        self.lookup = lookup
        self.store = StateStore(state_dir)
        self.base_url = base_url
        self.credentials_file = credentials_file
        self.log = log
        self.client_factory = client_factory or (lambda: GproxyClient(base_url, credentials_file))
        self.clock = clock
        self.shutdown_timeout = shutdown_timeout
        self._task = None
        self._stop = asyncio.Event()
        self._mutation_lock = asyncio.Lock()
        self._last_error = None
        self._last_sweep_at = None
        self._lock_busy = False
        self._in_sweep = False
        self._last_summary = None

    def enabled(self) -> bool:
        return coerce_bool(self.lookup('quota_keeper', 'enabled'), False)

    def interval(self) -> float:
        try:
            value = float(self.lookup('quota_keeper', 'check_interval_sec') or 60)
        except (TypeError, ValueError):
            value = 60
        # Configuration cannot turn this feature into a request flood.
        return min(3600, max(60, value))

    def emit(self, event: str, **fields):
        # Do not pass exception messages, credentials, arbitrary reply text or URLs.
        fields.pop('reply', None)
        self.log('quota_keeper ' + json.dumps({'event': event, **fields}, ensure_ascii=False))

    async def start(self):
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name='antigravity-claude-quota-keeper')

    async def stop(self):
        self._stop.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task), self.shutdown_timeout)
        except asyncio.TimeoutError:
            # A POST may already have arrived. Its durable guard remains intact.
            self.emit('shutdown_timeout_guard_preserved')
            self._task.cancel()
        finally:
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self):
        next_sweep = 0.0
        previous_enabled = None
        while not self._stop.is_set():
            try:
                enabled = self.enabled()
                if enabled != previous_enabled:
                    self.emit('enabled' if enabled else 'disabled')
                    if enabled:
                        next_sweep = 0.0
                    previous_enabled = enabled
                if enabled and time.monotonic() >= next_sweep:
                    await self.sweep()
                    self._last_error = None
                    next_sweep = time.monotonic() + self.interval()
            except Exception as exc:
                self._last_error = type(exc).__name__
                self.emit('sweep_error', error_type=self._last_error)
                next_sweep = time.monotonic() + 60
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

    async def sweep(self):
        if not self.enabled() or self._stop.is_set():
            return
        async with self._mutation_lock:
            with self.store.locked() as acquired:
                self._lock_busy = not acquired
                if not acquired:
                    self.emit('lock_busy')
                    return
                state = self.store.load()
                persist = lambda: self.store.save(state)
                self._in_sweep = True
                try:
                    async with self.client_factory() as api:
                        providers = await api.api('providers')
                        provider_ids = {p['id'] for p in providers
                                        if p['channel'] == 'antigravity' and p.get('enabled', True)}
                        groups = account_groups(await api.api('credentials'), provider_ids)
                        for key, row in state['accounts'].items():
                            row['active'] = key in groups
                        for key, members in groups.items():
                            if self._stop.is_set() or not self.enabled():
                                break
                            row = state['accounts'].setdefault(key, {})
                            row.update(credential_ids=sorted(c['id'] for c in members), active=True)
                            try:
                                await handle_account(
                                    api, members, row, persist, self.emit, clock=self.clock,
                                    may_generate=lambda: self.enabled() and not self._stop.is_set(),
                                )
                            except (httpx.HTTPError, ProbeError, ValueError, KeyError) as exc:
                                quota_failure(row, self.clock(), exc, persist, self.emit)
                    state['last_run_at'] = self.clock()
                    state['schema_version'] = 1
                    persist()
                    self._last_sweep_at = state['last_run_at']
                    summary = (len(groups), sum(bool(r.get('paused')) for r in state['accounts'].values()
                                                if r.get('active')))
                    if summary != self._last_summary:
                        self.emit('account_summary_changed', unique_accounts=summary[0], paused=summary[1])
                        self._last_summary = summary
                finally:
                    self._in_sweep = False

    def status(self) -> dict:
        # Disk-only, including when the feature is disabled or unavailable.
        # Neither this method nor /health can wake a model or refresh upstream.
        enabled = self.enabled()
        error = self._last_error
        try:
            state = self.store.load()
        except StateError as exc:
            state = {'accounts': {}}
            error = type(exc).__name__
        accounts = []
        for row in state['accounts'].values():
            generation = row.get('last_generation') or {}
            accounts.append({
                'credential_ids': row.get('credential_ids', []),
                'credential_id': row.get('credential_id'),
                'active': bool(row.get('active')),
                'paused': bool(row.get('paused')),
                'status': row.get('status', 'uninitialized'),
                'next_at': row.get('next_at'),
                'guard_until': row.get('guard_until'),
                'last_trigger_at': row.get('last_trigger_at'),
                'last_probe_at': row.get('last_probe_at'),
                'cycle_confirmed': bool(row.get('cycle_confirmed')),
                'pending_verify': bool(row.get('pending_verify')),
                'attempts': row.get('triggers', 0),
                'quota_failures': row.get('failures', 0),
                'windows': row.get('windows', {}),
                'last_generation': {key: generation[key] for key in
                    ('http_status', 'ok', 'usage') if key in generation},
            })
        return {
            'enabled': enabled,
            'status': 'disabled' if not enabled else ('degraded' if error else
                      'standby' if self._lock_busy else 'running'),
            'model': MODEL,
            'check_interval_sec': self.interval(),
            'worker_running': self._task is not None and not self._task.done(),
            'in_sweep': self._in_sweep,
            'last_error': error,
            'last_run_at': state.get('last_run_at'),
            'unique_active_accounts': sum(a['active'] for a in accounts),
            'paused_accounts': sum(a['active'] and a['paused'] for a in accounts),
            'accounts': accounts,
        }

    async def resume(self, credential_id: int) -> str:
        # Schedule a re-check; never erase the guard or send a request here.
        if self._mutation_lock.locked():
            return 'busy'
        async with self._mutation_lock:
            with self.store.locked() as acquired:
                if not acquired:
                    return 'busy'
                state = self.store.load()
                for row in state['accounts'].values():
                    if credential_id in row.get('credential_ids', []):
                        row.update(paused=False, failures=0, next_at=0,
                                   status='resume_scheduled')
                        self.store.save(state)
                        self.emit('resume_scheduled', credential_id=credential_id)
                        return 'scheduled'
                return 'not_found'
