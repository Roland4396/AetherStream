"""Small coercion helpers shared by config/runtime code."""

import json
import re
from typing import Any


def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    return default


def coerce_string_list(value: Any, default: list[str] | None = None) -> list[str]:
    fallback = list(default or [])

    if value is None:
        return fallback

    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return fallback
        try:
            loaded = json.loads(raw)
        except Exception:
            loaded = None
        if isinstance(loaded, list):
            items = loaded
        else:
            items = re.split(r'[\r\n,]+', raw)
    else:
        items = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def coerce_positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if result > 0 else default
