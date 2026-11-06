"""Pure scheduling policy, preserved from the original systemd keeper.

An unused API deadline slides with query time. Fixed reset clocks establish
a live cycle even with zero reported use; a future deadline alone does not.
"""

import math
import re

GRACE = 30
FIVE_HOURS = 5 * 3600
RETRY_DELAYS = (300, 900)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def windows(snapshot):
    """Only Claude/3p buckets influence decisions, never Gemini buckets."""
    return {
        e["id"]: {"used": number(e.get("value", {}).get("used_percent")),
                  "end": number(e.get("value", {}).get("period_end")),
                  "disabled": e.get("label") == "antigravity_disabled"}
        for e in snapshot.get("entries", [])
        if e.get("source_id") == "subscription"
        and e.get("id") in ("3p-5h", "3p-weekly")
        and e.get("value", {}).get("kind") == "window"
    }


def reset_clock_running(end, observed_at, duration, now, *, previous_end=None,
                        previous_observed=None, previous_confirmed=False):
    """A fixed countdown, not the sliding unused now+duration placeholder.

    Compare against the snapshot's observation time, never current time alone:
    an old unused cached snapshot must not look like a running clock as it ages.
    """
    values = (end, observed_at, duration, now)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        return False
    if observed_at <= 0 or observed_at > now + 2 or end <= now:
        return False
    matching = (isinstance(previous_end, (int, float)) and not isinstance(previous_end, bool)
                and math.isfinite(previous_end) and abs(end - previous_end) <= 2)
    if previous_confirmed is True and matching:
        return True  # Retain durable proof for the same still-live cycle.
    return (isinstance(previous_end, (int, float)) and not isinstance(previous_end, bool)
            and isinstance(previous_observed, (int, float)) and not isinstance(previous_observed, bool)
            and math.isfinite(previous_end) and math.isfinite(previous_observed)
            and previous_observed > 0 and observed_at - previous_observed >= 15
            and abs(end - previous_end) <= 2)


def cycle_running(quota, kind, now, *, observed_at=0, previous=None,
                  previous_observed=0, previous_confirmed=False):
    window = quota.get(kind, {})
    end = window.get('end')
    if not isinstance(end, (int, float)) or end <= now or window.get('disabled'):
        return False
    # Usage may suppress a paid request, but only the clock confirms the cycle.
    return reset_clock_running(end, observed_at,
        FIVE_HOURS if kind == '3p-5h' else 7 * 24 * 3600, now,
        previous_end=(previous or {}).get(kind, {}).get('end'),
        previous_observed=previous_observed, previous_confirmed=previous_confirmed)


def decision(quota, now, *, observed_at=0, previous=None, previous_observed=0, previous_confirmed=False):
    """Return wait(deadline), due, unused, or unknown; never invent a reset."""
    week, short = quota.get("3p-weekly"), quota.get("3p-5h")
    if not week or week['used'] is None:
        if cycle_running(quota, '3p-5h', now, observed_at=observed_at,
                         previous=previous, previous_observed=previous_observed, previous_confirmed=previous_confirmed):
            return 'wait', quota['3p-5h']['end'] + GRACE
        return 'unknown', now
    if week["disabled"] or week["used"] >= 100:
        end = week["end"]
        return ("wait", end + GRACE) if end and end + GRACE > now else ("due", now)
    # A weekly boundary also warrants a fresh quota read.
    if week["end"] and week["end"] + GRACE <= now:
        return "due", now
    if not short or short['disabled']:
        return 'unknown', now
    if short["end"] and short["end"] + GRACE <= now:
        return "due", now
    running = cycle_running(quota, '3p-5h', now, observed_at=observed_at,
                            previous=previous, previous_observed=previous_observed, previous_confirmed=previous_confirmed)
    if not running and observed_at > 0 and short.get('end') and short['end'] < observed_at + FIVE_HOURS - 15:
        return 'observe', now + 60
    if short['used'] is None and not running:
        return 'unknown', now
    if short['used'] == 0 and not running:
        # For unused windows the API returns now+5h, not a fixed running clock.
        return "unused", now
    if not short["end"]:
        return "unknown", now
    ends = [q["end"] + GRACE for q in (week, short) if q["end"]]
    return "wait", min(ends)


def account_groups(credentials, provider_ids):
    groups = {}
    for c in credentials:
        if not c["enabled"] or c["provider_id"] not in provider_ids:
            continue
        email = re.search(r"[\w.+-]+@[\w.-]+", c.get("label") or "")
        key = email.group().lower() if email else "credential:" + str(c["id"])
        groups.setdefault(key, []).append(c)
    return groups


def observed(snapshot):
    return max((s.get("observed_at_ms") or 0 for s in snapshot.get("sources", [])
                if s.get("capability", {}).get("id") == "subscription"), default=0)

