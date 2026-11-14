"""Hydrogen uses the shared ordinary-Gemini five-hour AND weekly allowance.

A separate account schedule, never an independent budget or a quota reset API.
No work on import, no automatic retry of a potentially delivered generation.
"""
import math
import httpx
from .client import ProbeError
from .policy import FIVE_HOURS, GRACE, reset_clock_running
from . import argon

MODELS = frozenset('gemini-4-hydrogen-' + level for level in ('low', 'medium', 'high', 'xhigh'))
DEFAULT_MODEL = 'gemini-4-hydrogen-low'
SHORT, WEEKLY = 'gemini-5h', 'gemini-weekly'
DURATIONS = {SHORT: FIVE_HOURS, WEEKLY: argon.WEEK}


def settings(value):
    if value is None:
        return {'enabled': False, 'model': DEFAULT_MODEL}
    if not isinstance(value, dict) or set(value) - {'enabled', 'model'}:
        raise ValueError('Invalid Hydrogen settings')
    enabled, model = value.get('enabled', False), value.get('model', DEFAULT_MODEL)
    if not isinstance(enabled, bool) or not isinstance(model, str) or model not in MODELS:
        raise ValueError('Invalid Hydrogen settings')
    return {'enabled': enabled, 'model': model}


def permission_refusal(account):
    # Do not start a newly-added family on an account with known authentication
    # or permission refusal. A full quota response is not service authorization.
    return any((row.get('last_generation') or {}).get('http_status') in (401, 403)
               for row in (account, account.get('argon_weekly', {}), account.get('hydrogen', {})))


async def handle_account(api, members, row, persist, emit, *, model, clock,
                         may_generate=lambda: True):
    now = clock()
    if row.get('paused') or now < row.get('next_at', 0):
        return
    member = next((c for c in members if c['id'] == row.get('credential_id')), None)
    if member is None:
        member = min(members, key=lambda c: (c.get('health') == 'dead', str(c['id'])))
    cid = member['id']; row['credential_id'] = cid
    reading = await api.hydrogen_probe(cid)
    now = clock()
    prior = row.get('windows') or {}
    prior_observed = row.get('observed_at_ms', 0) / 1000
    prior_confirmed = row.get('cycle_confirmed') is True
    unconfirmed = bool(row.get('pending_verify') or row.get('uncertain_attempt')
                       or (row.get('last_trigger_at') and not prior_confirmed))
    row.update(last_probe_at=now, observed_at_ms=reading['observed_at_ms'],
               available=reading['available'])
    if not reading['available'] or model not in reading.get('models', []):
        row.update(status='paused_unavailable_after_attempt' if unconfirmed else 'unavailable',
                   paused=unconfirmed, pending_verify=False, failures=0,
                   next_at=now + argon.UNAVAILABLE_RECHECK)
        persist(); return
    quota = reading['windows']
    for name, duration in DURATIONS.items():
        value = quota.get(name, {})
        for field in ('remaining', 'end'):
            v = value.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ProbeError('Invalid Hydrogen quota')
        if not 0 <= value['remaining'] <= 1 or value['end'] <= now or value['end'] > now + duration + 300:
            raise ProbeError('Invalid or stale Hydrogen quota window')
    row['windows'] = quota
    short, week = quota[SHORT], quota[WEEKLY]
    if week['remaining'] == 0:
        row.update(status='weekly_exhausted', next_at=week['end'] + GRACE, failures=0)
        persist(); return
    if week.get('disabled') or short.get('disabled'):
        raise ProbeError('Disabled Hydrogen quota window')
    observed = reading['observed_at_ms'] / 1000
    running = reset_clock_running(short['end'], observed, FIVE_HOURS, now,
        previous_end=prior.get(SHORT, {}).get('end'), previous_observed=prior_observed,
        previous_confirmed=prior_confirmed)
    possible_clock = short['end'] < observed + FIVE_HOURS - 15
    if short['remaining'] < 1 or running or possible_clock:
        if unconfirmed and not running:
            checks = row.get('clock_checks', 0) + 1
            row.update(clock_checks=checks, cycle_confirmed=False, uncertain_attempt=True,
                       pending_verify=checks < 2, paused=checks >= 2, failures=0,
                       next_at=now + argon.VERIFY_DELAY,
                       status='awaiting_cycle_clock' if checks < 2 else 'paused_unconfirmed_generation')
        else:
            waiting = running or short['remaining'] < 1
            row.update(status='waiting' if waiting else 'awaiting_cycle_clock',
                       next_at=min(short['end'], week['end']) + GRACE if waiting else now + argon.VERIFY_DELAY,
                       cycle_end=short['end'], cycle_confirmed=running, failures=0,
                       pending_verify=False, uncertain_attempt=False, clock_checks=0)
            if unconfirmed and running:
                row['guard_until'] = short['end'] + GRACE
                emit('hydrogen_cycle_checked', credential_id=cid, confirmed=True)
        persist(); return
    if unconfirmed:
        row.update(status='paused_unconfirmed_generation', paused=True,
                   pending_verify=False, uncertain_attempt=True, cycle_confirmed=False, failures=0)
        persist(); return
    if now < row.get('guard_until', 0):
        row.update(status='request_guard', next_at=row['guard_until'], failures=0)
        persist(); return
    if not may_generate():
        persist(); return
    row.update(last_trigger_at=now, guard_until=now + FIVE_HOURS + GRACE,
               pending_verify=True, uncertain_attempt=True, cycle_confirmed=False, clock_checks=0,
               next_at=now + argon.VERIFY_DELAY, status='submitted', model=model,
               triggers=row.get('triggers', 0) + 1, failures=0)
    persist()  # Durable BEFORE exact-account one-attempt POST.
    try:
        result = argon.safe_result(await api.generate_hydrogen(member, model))
    except (httpx.HTTPError, ProbeError, ValueError, KeyError) as exc:
        row.update(status='delivery_uncertain', generation_error=type(exc).__name__)
        persist(); emit('hydrogen_delivery_uncertain', credential_id=cid, error_type=type(exc).__name__)
        return
    row.update(last_generation=result,
               status='pending_verification' if result['ok'] else 'delivery_needs_verification')
    if result['http_status'] in (401, 403):
        row.update(paused=True, status='paused_account_permission', pending_verify=False)
    persist(); emit('hydrogen_response_received', credential_id=cid, **result)


def quota_failure(row, now, error, persist, emit):
    argon.quota_failure(row, now, error, persist,
                        lambda event, **fields: emit(event.replace('argon_', 'hydrogen_'), **fields))


def status(row):
    result = argon.status(row)
    result['windows'] = {name: {k: v for k, v in value.items()
        if (k in ('remaining', 'end') and not isinstance(v, bool)
            and isinstance(v, (int, float)) and math.isfinite(v))
        or (k == 'disabled' and isinstance(v, bool))}
        for name, value in (row.get('windows') or {}).items()
        if name in DURATIONS and isinstance(value, dict)}
    return result
