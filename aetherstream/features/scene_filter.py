"""Remove only the fixed, injected AI scene summaries from user messages."""

import re


SCENE_INTRO = '以下是scene，有时间戳和场景信息等：'
_INTRO_AT_END = re.compile(r'(?m)^[ \t]*' + re.escape(SCENE_INTRO) + r'[ \t\r\n]*\Z')
_TAG = re.compile(r'<(?P<close>/)?scene\b[^>]*>', re.I)
_PROTECTED_TAG = re.compile(r'</?(?:stage|ai_generated_stage|recall|latest_human_message)\b', re.I)
_EXPECTED_LINES = ('Day:', '【场景】', '【氛围】', '【类型】', '期限=')


def remove_scene_text(text: str) -> tuple[str, int, int]:
    """Return cleaned text, removed blocks, and malformed/ambiguous blocks.

    Require the exact standalone introduction, a complete nonnested <scene>,
    and the observed field labels. Do not infer missing boundaries or search
    for arbitrary dates. Leave everything outside accepted spans byte-for-byte
    unchanged, including stage/recall, prose, and local variable snapshots.
    """
    spans = []
    depth = 0
    intro_start = None
    body_start = 0
    malformed = False
    invalid = 0
    for tag in _TAG.finditer(text):
        if not tag.group('close'):
            if depth == 0:
                intro = _INTRO_AT_END.search(text[:tag.start()])
                intro_start = intro.start() if intro else None
                body_start = tag.end()
                malformed = tag.group(0) != '<scene>'
            else:
                malformed = True
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0 and intro_start is not None:
                body = text[body_start:tag.start()]
                lines = [line.strip() for line in body.splitlines() if line.strip()]
                fields_present = all(any(line.startswith(label) for line in lines)
                                     for label in _EXPECTED_LINES)
                if (malformed or tag.group(0) != '</scene>' or not fields_present
                        or not lines[0].startswith('Day:') or _PROTECTED_TAG.search(body)):
                    invalid += 1
                else:
                    spans.append((intro_start, tag.end()))
    if depth and intro_start is not None:
        invalid += 1
    if not spans:
        return text, 0, invalid
    result = []
    cursor = 0
    for start, end in spans:
        result.append(text[cursor:start])
        cursor = end
    result.append(text[cursor:])
    return ''.join(result), len(spans), invalid


def remove_scene_messages(messages) -> tuple[int, int]:
    """Clean every current/historical user text block; do not alter other data."""
    removed = invalid = 0
    if not isinstance(messages, list):
        return removed, invalid
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        content = message.get('content')
        if isinstance(content, str):
            targets = [(message, 'content')]
        elif isinstance(content, list):
            targets = [(part, 'text') for part in content
                       if isinstance(part, dict) and part.get('type') == 'text'
                       and isinstance(part.get('text'), str)]
        else:
            continue
        for target, key in targets:
            target[key], count, errors = remove_scene_text(target[key])
            removed += count
            invalid += errors
    return removed, invalid
