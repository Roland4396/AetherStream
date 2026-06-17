"""Provider-specific drawing context filters."""

import re
from typing import Any


DS_DRAWING_CONTEXT_FILTER_ROUTE_NAME = 'deepseek'
DRAWING_CONTEXT_CLOSED_TAG_RE = re.compile(
    r'<(?P<tag>imgthink|image)\b[^>]*>.*?</(?P=tag)>',
    re.IGNORECASE | re.DOTALL,
)
DRAWING_CONTEXT_BARE_ANALYSIS_RE = re.compile(
    r'\*\*\[剧情节奏检查\]\*\*.*?image###Scene Composition:.*?;###',
    re.IGNORECASE | re.DOTALL,
)
DRAWING_CONTEXT_BARE_IMAGE_PROMPT_RE = re.compile(
    r'(?:(?:【[^】\r\n]{1,80}】)?image###Scene Composition:).*?;###',
    re.IGNORECASE | re.DOTALL,
)


def _strip_closed_drawing_blocks_from_text(text: str) -> tuple[str, int, int]:
    """Remove only fully closed drawing-only blocks.

    Deliberately does not try to recover from broken/unclosed tags. Losing a
    malformed <imgthink> block is acceptable; swallowing following story text is
    not.
    """
    if not isinstance(text, str) or not text:
        return text, 0, 0

    removed_blocks = 0
    removed_chars = 0

    def repl(match: re.Match) -> str:
        nonlocal removed_blocks, removed_chars
        removed_blocks += 1
        removed_chars += len(match.group(0))
        return ''

    cleaned = DRAWING_CONTEXT_CLOSED_TAG_RE.sub(repl, text)
    cleaned = DRAWING_CONTEXT_BARE_ANALYSIS_RE.sub(repl, cleaned)
    cleaned = DRAWING_CONTEXT_BARE_IMAGE_PROMPT_RE.sub(repl, cleaned)
    if cleaned != text:
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned, removed_blocks, removed_chars


def _strip_closed_drawing_blocks_from_content(content: Any) -> tuple[Any, int, int]:
    if isinstance(content, str):
        return _strip_closed_drawing_blocks_from_text(content)

    if not isinstance(content, list):
        return content, 0, 0

    removed_blocks = 0
    removed_chars = 0
    changed = False
    cleaned_items = []

    for item in content:
        if isinstance(item, str):
            cleaned, blocks, chars = _strip_closed_drawing_blocks_from_text(item)
            changed = changed or cleaned != item
            removed_blocks += blocks
            removed_chars += chars
            cleaned_items.append(cleaned)
            continue

        if isinstance(item, dict) and isinstance(item.get('text'), str):
            new_item = dict(item)
            cleaned, blocks, chars = _strip_closed_drawing_blocks_from_text(new_item['text'])
            changed = changed or cleaned != new_item['text']
            removed_blocks += blocks
            removed_chars += chars
            new_item['text'] = cleaned
            cleaned_items.append(new_item)
            continue

        cleaned_items.append(item)

    if not changed:
        return content, 0, 0
    return cleaned_items, removed_blocks, removed_chars


def apply_drawing_context_filter(request_data: dict) -> dict[str, int]:
    messages = request_data.get('messages')
    if not isinstance(messages, list):
        return {'messages': 0, 'blocks': 0, 'chars': 0}

    touched_messages = 0
    removed_blocks = 0
    removed_chars = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        cleaned, blocks, chars = _strip_closed_drawing_blocks_from_content(content)
        if blocks:
            message['content'] = cleaned
            touched_messages += 1
            removed_blocks += blocks
            removed_chars += chars

    return {'messages': touched_messages, 'blocks': removed_blocks, 'chars': removed_chars}


def should_apply_deepseek_drawing_context_filter(route_name: Any, base_url: Any) -> bool:
    return str(route_name or '').strip() == DS_DRAWING_CONTEXT_FILTER_ROUTE_NAME
