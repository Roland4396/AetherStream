"""GLM-5.2 family policy, enforced at the final Chat Completions send boundary.

Official wire format: https://docs.z.ai/guides/llm/glm-5.2
This intentionally does not depend on a route name or a channel's display prefix.
No scheduling, provider calls, prompt edits or shared state live here.
"""

import re
import unicodedata
from typing import Any, Callable


_GLM52 = re.compile(r"(?<![a-z0-9])glm[-_.\s]*5[._-]2(?![0-9])", re.IGNORECASE)


def is_glm52_model(model: Any) -> bool:
    return isinstance(model, str) and bool(_GLM52.search(unicodedata.normalize('NFKC', model)))


def apply_glm52_official_thinking(payload: dict[str, Any]) -> list[str]:
    """Set the approved family policy; touch only reasoning-related controls.

    A last-mile guard wins over both caller controls and route compatibility
    defaults, including the old generic template-thinking disable rule. Nested
    dictionaries are copied before editing; messages/tools/sampling are intact.
    """
    if not is_glm52_model(payload.get('model')):
        return []

    changed = []
    previous = payload.get('thinking')
    thinking: dict[str, Any] = {'type': 'enabled'}
    # Preserve the one other documented GLM thinking setting if explicitly set.
    if isinstance(previous, dict) and isinstance(previous.get('clear_thinking'), bool):
        thinking['clear_thinking'] = previous['clear_thinking']
    if previous != thinking:
        payload['thinking'] = thinking
        changed.append('thinking')
    if payload.get('reasoning_effort') != 'max':
        payload['reasoning_effort'] = 'max'
        changed.append('reasoning_effort')

    for key in ('enable_thinking', 'effort'):
        if key in payload:
            payload.pop(key)
            changed.append(key)
    for container, fields in (
        ('chat_template_kwargs', ('enable_thinking', 'thinking', 'reasoning_effort', 'effort')),
        ('reasoning', ('effort',)),
        ('output_config', ('effort',)),
    ):
        original = payload.get(container)
        if not isinstance(original, dict):
            continue
        cleaned = dict(original)
        for field in fields:
            if field in cleaned:
                cleaned.pop(field)
                changed.append(f'{container}.{field}')
        if cleaned:
            payload[container] = cleaned
        else:
            payload.pop(container, None)
    return changed


def enforce_glm52_official_thinking(
    payload: dict[str, Any], *, log: Callable[[str], None], trace_prefix: str,
) -> None:
    if not is_glm52_model(payload.get('model')):
        return
    changed = apply_glm52_official_thinking(payload)
    log(
        f"{trace_prefix}glm52_official_thinking "
        f"thinking=enabled reasoning_effort=max changed={','.join(changed) or '-'}"
    )
