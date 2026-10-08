"""Sampling compatibility for Kimi models with fixed sampling parameters."""

import re
from typing import Any


def apply_kimi_sampling_compat(payload: dict[str, Any], model: Any) -> list[str]:
    # https://platform.kimi.ai/docs/api/models-overview
    # Omit fixed sampling fields so the model selects its supported defaults
    # (including K2.6's different thinking/non-thinking temperatures).
    name = str(model or '').strip().lower().rsplit('/', 1)[-1]
    name = re.sub(r'^\[[^\]]+\]\s*', '', name)
    if not re.match(r'^kimi[-_.]k?(?:3|2\.6|2\.7)(?:[-_.]|$)', name):
        return []
    removed = []
    for field in ('temperature', 'top_p'):
        if field in payload:
            payload.pop(field)
            removed.append(field)
    return removed
