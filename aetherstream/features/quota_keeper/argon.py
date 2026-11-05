"""Independent, fail-closed Argon weekly cycle keeper; no work on import."""
from __future__ import annotations

import math

import httpx

from .client import ProbeError
from .policy import GRACE, RETRY_DELAYS

WEEK = 7 * 24 * 3600
UNAVAILABLE_RECHECK = 24 * 3600
VERIFY_DELAY = 60
BUCKET = 'gemini-argon-wtu-weekly'
MODELS = frozenset('gemini-4-argon-' + level for level in ('low', 'medium', 'high'))


def settings(value):
    if value is None:
        return {'enabled': False, 'model': 'gemini-4-argon-low'}
    if not isinstance(value, dict) or set(value) - {'enabled', 'model'}:
        raise ValueError('Invalid Argon weekly settings')
    enabled = value.get('enabled', False)
    model = value.get('model', 'gemini-4-argon-low')
    if not isinstance(enabled, bool) or not isinstance(model, str) or model not in MODELS:
        raise ValueError('Invalid Argon weekly settings')
    return {'enabled': enabled, 'model': model}


def safe_result(value):
    """Only known metrics survive; never reply text, account labels or errors."""
    if not isinstance(value, dict):
        raise ValueError('Invalid Argon generation result')
    status = value.get('http_status', 0)
    if isinstance(status, bool) or not isinstance(status, int):
        status = 0
    usage = value.get('usage') or {}
    return {'http_status': status, 'ok': value.get('ok') is True,
            'usage': {k: v for k, v in usage.items()
                      if k in ('inputTokens', 'outputTokens', 'reasoningTokens')
                      and not isinstance(v, bool) and isinstance(v, (int, float))
                      and math.isfinite(v) and v >= 0} if isinstance(usage, dict) else {}}


async def handle_argon_account(api, members, row, persist, emit, *, model,
                               clock, may_generate=lambda: True):
    now = clock()
    if row.get('paused') or now < row.get('next_at', 0):
        return
    member = next((c for c in members if c['id'] == row.get('credential_id')), None)
    if member is None:
        member = min(members, key=lambda c: (c.get('health') == 'dead', str(c['id'])))
    cid = member['id']
    row['credential_id'] = cid
    reading = await api.argon_probe(cid)
    now = clock()
    row.update(last_probe_at=now, observed_at_ms=reading['observed_at_ms'],
               available=reading['available'], window={k: reading[k] for k in
                   ('remaining', 'end') if k in reading})
    unconfirmed = bool(row.get('pending_verify') or row.get('uncertain_attempt')
                       or (row.get('last_trigger_at') and not row.get('cycle_confirmed')))
    if not reading['available'] or model not in reading.get('models', []):
        if unconfirmed:
            row.update(paused=True, pending_verify=False,
                       status='paused_unavailable_after_attempt')
        else:
            row.update(status='unavailable', next_at=now + UNAVAILABLE_RECHECK, failures=0)
        persist()
        return
    remaining, end = reading['remaining'], reading['end']
    if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
            or not math.isfinite(remaining) or not 0 <= remaining <= 1
            or isinstance(end, bool) or not isinstance(end, (int, float))
            or not math.isfinite(end) or end + GRACE <= now or end > now + WEEK + 300):
        raise ProbeError('Invalid or stale Argon weekly window')
    if remaining < 1:
        # Any existing real use starts this cycle. Exhaustion also waits for
        # the weekly reset; neither condition warrants another completion.
        checked = unconfirmed
        row.update(status='waiting', next_at=end + GRACE, cycle_end=end,
                   cycle_confirmed=True, pending_verify=False,
                   uncertain_attempt=False, failures=0)
        persist()
        if checked:
            emit('argon_cycle_checked', credential_id=cid, confirmed=True, next_at=end + GRACE)
        return
    if unconfirmed:
        # A sliding unused deadline is NOT proof that a submitted request
        # failed to arrive. This ambiguity survives a week, restart and resume.
        row.update(status='paused_unconfirmed_generation', paused=True,
                   pending_verify=False, uncertain_attempt=True, failures=0)
        persist()
        emit('argon_paused', credential_id=cid, reason='weekly_cycle_not_confirmed')
        return
    if now < row.get('guard_until', 0):
        row.update(status='request_guard', next_at=row['guard_until'], failures=0)
        persist()
        return
    if not may_generate():
        persist()
        return
    row.update(last_trigger_at=now, guard_until=now + WEEK + GRACE,
               pending_verify=True, uncertain_attempt=True, cycle_confirmed=False,
               next_at=now + VERIFY_DELAY, status='submitted', model=model,
               triggers=row.get('triggers', 0) + 1, failures=0)
    persist()  # Durable BEFORE the one paid POST; singleton lease stays held.
    try:
        result = safe_result(await api.generate_argon(member, model))
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        row.update(status='delivery_uncertain', generation_error=type(exc).__name__)
        persist()
        emit('argon_delivery_uncertain', credential_id=cid, error_type=type(exc).__name__)
        return
    row.update(last_generation=result,
               status='pending_verification' if result['ok'] else 'delivery_needs_verification')
    persist()
    emit('argon_triggered', credential_id=cid, **result)


def quota_failure(row, now, error, persist, emit):
    count = row.get('failures', 0) + 1
    row.update(failures=count, last_error=type(error).__name__)
    if count > len(RETRY_DELAYS):
        row.update(paused=True, status='paused_quota_error')
    else:
        row.update(next_at=now + RETRY_DELAYS[count - 1], status='quota_retry_wait')
    persist()
    emit('argon_quota_failed', credential_id=row.get('credential_id'), failures=count,
         paused=bool(row.get('paused')), error_type=type(error).__name__)


def status(row):
    """Disk-only metrics projection, deliberately not a raw state dump."""
    result = {k: row[k] for k in ('credential_id', 'status', 'paused', 'next_at',
              'guard_until', 'last_trigger_at', 'last_probe_at', 'available',
              'cycle_confirmed', 'pending_verify', 'uncertain_attempt',
              'model', 'failures') if k in row}
    result['attempts'] = row.get('triggers', 0)
    if isinstance(row.get('window'), dict):
        result['window'] = {k: v for k, v in row['window'].items()
                            if k in ('remaining', 'end') and not isinstance(v, bool)
                            and isinstance(v, (int, float)) and math.isfinite(v)}
    if isinstance(row.get('last_generation'), dict):
        result['last_generation'] = safe_result(row['last_generation'])
    return result
