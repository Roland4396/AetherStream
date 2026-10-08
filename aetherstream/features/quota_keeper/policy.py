"""Pure scheduling policy, preserved from the original systemd keeper.

An unused quota API deadline slides with query time; a nonzero used window
has a running clock. A future deadline alone is NOT evidence of a live cycle.
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


def decision(quota, now):
    """Return wait(deadline), due, unused, or unknown; never invent a reset."""
    week, short = quota.get("3p-weekly"), quota.get("3p-5h")
    if not week or week["used"] is None:
        return "unknown", now
    if week["disabled"] or week["used"] >= 100:
        end = week["end"]
        return ("wait", end + GRACE) if end and end + GRACE > now else ("due", now)
    # A weekly boundary also warrants a fresh quota read.
    if week["end"] and week["end"] + GRACE <= now:
        return "due", now
    if not short or short["disabled"] or short["used"] is None:
        return "unknown", now
    if short["end"] and short["end"] + GRACE <= now:
        return "due", now
    if short["used"] == 0:
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

