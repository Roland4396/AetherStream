"""Local user-input transliteration and neutral wrapping, without browser state.

Only the latest human-input text is eligible. It never edits stored messages,
worldbooks, system/developer messages, tool arguments/results or signed parts.
The desktop pet, UI, continuation, chat writes and fabricated audit carriers in
the supplied frontend are intentionally not ported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


LEXICONS = json.loads(Path(__file__).with_name('input_transform_lexicons.json').read_text())
CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]+')
WRAPPED = re.compile(r'<(latest_human_message|user_input|input)\b[^>]*>(.*?)</\1\s*>', re.S | re.I)
CODE_REGION = re.compile(r'```[^\n]*\n?[\s\S]*?(?:```|\Z)|~~~[^\n]*\n?[\s\S]*?(?:~~~|\Z)|`[^`\n]*`')
# The phone/stage assembler appends this footer to the real input turn; other
# extensions may append a separate user-role rule turn after it. The explicit
# assembler boundary, not the last user role or a size heuristic, owns selection.
INPUT_FOOTER = re.compile(
    r'\n[ \t]*\n[ \t]*以上是用户的本轮输入[，,][^\n]*(?:stage|act|scene|角色)[^\n]*(?:\n|\Z)', re.I)
TECHNICAL = re.compile(
    r'```[^\n]*\n?[\s\S]*?(?:```|\Z)|~~~[^\n]*\n?[\s\S]*?(?:~~~|\Z)'
    r'|`[^`\n]*`|\{\{[^{}]*\}\}|\$\{[^{}]*\}'
    r'|<(UpdateVariable|JSONPatch|stat_data|delta_data|variables|schema|tool|function|stage|act|phone_events)\b[^>]*>[\s\S]*?</\1\s*>'
    r'|https?://[^\s<>"\']+|/[\w\u3400-\u9fff~.-]+(?:/[\w\u3400-\u9fff~.-]+)+'
    r'|<[^<>\n]+>', re.I)
TAIL_BOUNDARIES = ('\n\n以上是用户的本轮输入', '\n\n<ai_generated_stage>', '\n\n<stage>',
                   '\n\n<act>', '\n\n!!important!!', '\n\n<phone_guidance>', '\n\n<plot_progress_control>')
FUNCTION_WORDS = frozenset('的了着在和与及或但不没是把被也都就才又而并为到从向对给让将要吗呢吧啊呀哦')
LATIN_EDGE = re.compile(r'[A-Za-z0-9\u00c0-\u024f\u1e00-\u1eff]')


def join_fragments(parts: list[str]) -> str:
    out = ''
    for part in parts:
        if not part:
            continue
        if out and not out[-1].isspace() and not part[0].isspace() and (
                (LATIN_EDGE.fullmatch(out[-1]) and (LATIN_EDGE.fullmatch(part[0]) or CJK.fullmatch(part[0])))
                or (CJK.fullmatch(out[-1]) and LATIN_EDGE.fullmatch(part[0]))):
            out += ' '
        out += part
    return out


def settings_from(value) -> dict:
    value = value if isinstance(value, dict) else {}
    mode = value.get('mode', 'pinyin')
    if mode not in ('pinyin', 'harmonic', 'fusion', 'mix'):
        raise ValueError('Invalid input transform mode')
    def number(key, default, low, high):
        raw = value.get(key, default)
        return max(low, min(high, int(raw))) if isinstance(raw, (float, int)) and not isinstance(raw, bool) and math.isfinite(raw) else default
    langs = value.get('languages', ['en'])
    langs = [x for x in langs if x in ('en', 'es')] if isinstance(langs, list) else ['en']
    terms = value.get('preserve_terms', [])
    terms = [x for x in terms if isinstance(x, str) and 0 < len(x) <= 128][:128] if isinstance(terms, list) else []
    return {'enabled': value.get('enabled') is True, 'mode': mode,
            'tone': value.get('tone') if value.get('tone') in ('symbol', 'number', 'none') else 'symbol',
            'ratio': number('ratio', 40, 0, 100), 'languages': langs or ['en'], 'preserve_terms': terms,
            'wrapper': value.get('wrapper', True) is True,
            'padding_chars': number('padding_chars', 32, 0, 128)}


def protected_spans(text: str, terms: list[str]) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in TECHNICAL.finditer(text)]
    decoder = json.JSONDecoder()
    for match in re.finditer(r'[\[{]', text):
        start = match.start()
        if any(a <= start < b for a, b in spans):
            continue
        try:
            item, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            continue
        if isinstance(item, (dict, list)):
            spans.append((start, end))
    for term in terms:
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(term), text))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def readings(run: str, tone='none') -> list[str]:
    from pypinyin import Style, lazy_pinyin
    style = {'none': Style.NORMAL, 'symbol': Style.TONE, 'number': Style.TONE3}[tone]
    result = lazy_pinyin(run, style=style, errors=lambda chars: list(chars), strict=False)
    return result if len(result) == len(run) else list(run)


def convert_run(run: str, cfg: dict, rng) -> str:
    mode = cfg['mode']
    if mode == 'mix':
        dictionaries = [LEXICONS['mix'][language] for language in cfg['languages']]
        pieces, at = [], 0
        while at < len(run):
            hit = None
            for size in range(min(4, len(run) - at), 0, -1):
                word = run[at:at + size]
                choices = [d[word] for d in dictionaries if word in d]
                if choices:
                    hit = (word, choices)
                    break
            if hit is None:
                pieces.append(run[at]); at += 1
            else:
                word, choices = hit
                pieces.append(rng.choice(choices) if rng.random() * 100 < cfg['ratio'] else word)
                at += len(word)
        return join_fragments(pieces)
    if mode == 'pinyin':
        return ' '.join(readings(run, cfg['tone']))
    normal = readings(run)
    toned = readings(run, cfg['tone']) if mode == 'fusion' else normal
    parts = []
    for index, char in enumerate(run):
        if mode == 'fusion' and rng.random() * 100 < cfg['ratio']:
            parts.append(toned[index])
            continue
        key = normal[index].replace('ü', 'v').lower()
        candidates = [c for c in LEXICONS['homophones'].get(key, '') if c != char]
        replace = (char not in FUNCTION_WORDS and candidates and
                   (mode == 'fusion' or rng.random() * 100 < cfg['ratio']))
        parts.append(rng.choice(candidates) if replace else char)
    return join_fragments(parts) if mode == 'fusion' else ''.join(parts)


def convert_prose(text: str, cfg: dict, rng) -> str:
    out, last = [], 0
    for match in CJK.finditer(text):
        out.append(text[last:match.start()])
        converted = convert_run(match.group(), cfg, rng)
        if out and out[-1] and out[-1][-1].isascii() and out[-1][-1].isalnum():
            converted = ' ' + converted
        if match.end() < len(text) and text[match.end()].isascii() and text[match.end()].isalnum():
            converted += ' '
        out.append(converted)
        last = match.end()
    out.append(text[last:])
    return ''.join(out)


def transform_text(text: str, cfg: dict, rng) -> str:
    out, at = [], 0
    for start, end in protected_spans(text, cfg['preserve_terms']):
        out.extend((convert_prose(text[at:start], cfg, rng), text[start:end]))
        at = end
    out.append(convert_prose(text[at:], cfg, rng))
    transformed = ''.join(out)
    if not cfg['wrapper']:
        return transformed
    # Neutral data padding only: no fabricated safety/audit verdicts or role claims.
    alphabet = 'bcdfghjklmnpqrstvxzBCDFGHJKLMNPQRSTVXZ0123456789'
    def padding():
        return '<!-- ' + ''.join(rng.choice(alphabet) for _ in range(cfg['padding_chars'])) + ' -->\n' if cfg['padding_chars'] else ''
    return ('<stream_input_transform form="' + cfg['mode'] + '">\n'
            '以下仅为本条用户输入的转写形式，按原意理解；输出语言和格式仍按原有要求。\n'
            + padding() + transformed + '\n' + padding() + '</stream_input_transform>')


def text_slots(message: dict, *, native=False) -> list[tuple[dict, str]]:
    if native:
        blocks = message.get('parts', [])
    else:
        content = message.get('content')
        if isinstance(content, str):
            return [(message, 'content')]
        blocks = content if isinstance(content, list) else []
    return [(b, 'text') for b in blocks if isinstance(b, dict) and isinstance(b.get('text'), str)
            and (native or b.get('type') == 'text')
            and not any(k in b for k in ('thoughtSignature', 'thought_signature', 'functionCall', 'functionResponse'))]


def human_blocks(text: str):
    code = [m.span() for m in CODE_REGION.finditer(text)]
    return [m for m in WRAPPED.finditer(text) if not any(a <= m.start() < b for a, b in code)]


def marked_human_span(text: str):
    matches = human_blocks(text)
    if matches:
        return matches[-1].span(2)
    code = [m.span() for m in CODE_REGION.finditer(text)]
    footers = [m for m in INPUT_FOOTER.finditer(text)
               if not any(a <= m.start() < b for a, b in code)]
    if len(footers) == 1 and text[:footers[0].start()].strip():
        return (0, footers[0].start())
    return None


def human_span(text: str):
    marked = marked_human_span(text)
    if marked is not None:
        return marked
    code = [m.span() for m in CODE_REGION.finditer(text)]
    if any(not any(a <= m.start() < b for a, b in code)
           for m in re.finditer(r'<(?:latest_human_message|user_input|input)\b', text, re.I)):
        return None
    stripped = text.strip()
    if re.search(r'<stream_input_transform\b', text, re.I):
        return None
    if not stripped or stripped.startswith('开始吧。JSON 必须合法并符合 schema；'):
        return None
    try:
        if isinstance(json.loads(stripped), (dict, list)):
            return None
    except (ValueError, RecursionError):
        pass
    boundaries = [m.start() for marker in TAIL_BOUNDARIES for m in re.finditer(re.escape(marker), text)
                  if not any(a <= m.start() < b for a, b in code)]
    end = min(boundaries or [len(text)])
    return (0, end) if text[:end].strip() else None


def preset_tail(text: str) -> bool:
    stripped = text.strip()
    return (stripped.startswith(('开始吧。JSON 必须合法并符合 schema；',
                                 '即将开始思考，', '发现在原生思考', '[Start a new chat]'))
            or bool(re.search(r'<(?:dm_core|dm_set|character_build|Session\s+Record|Information'
                              r'|Legal_Disclaimer_and_Rights_Declaration|FM2\.1-UNIFIED)\b', text, re.I)))


def transform_request_input(payload: dict, cfg: dict, rng) -> dict | None:
    if not cfg['enabled']:
        return None
    native = not isinstance(payload.get('messages'), list) and isinstance(payload.get('contents'), list)
    messages = payload.get('contents' if native else 'messages')
    if not isinstance(messages, list):
        return None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (not isinstance(message, dict) or message.get('role') != 'user'
                or message.get('tool_calls') or message.get('function_call')):
            continue
        slots = text_slots(message, native=native)
        # A marked human-input block is authoritative over other text parts.
        marked = [(obj, key) for obj, key in slots if marked_human_span(obj[key]) is not None]
        if not marked and slots and all(preset_tail(obj[key]) for obj, key in slots):
            # Compiled presets are not stored chat input. Once past them, use
            # the latest user message, just as the source script walks chat.
            # Missing XML markers do not disable ordinary input or regeneration.
            continue
        targets = marked or slots
        changes = []
        for obj, key in targets:
            text = obj[key]
            span = human_span(text)
            if span is None:
                continue
            start, end = span
            source = text[start:end]
            transformed = transform_text(source, cfg, rng)
            result = text[:start] + transformed + text[end:]
            if result != text:
                changes.append((obj, key, result, len(source)))
        if changes:
            for obj, key, result, _ in changes:
                obj[key] = result
            return {'status': 'applied', 'mode': cfg['mode'], 'message': index,
                    'parts': len(changes), 'source_chars': sum(x[3] for x in changes), 'wrapper': cfg['wrapper'],
                    'selection': 'marked_input' if marked else 'latest_user'}
        # Synthetic continuation instructions may follow the actual user message.
        if slots and all(preset_tail(obj[key]) for obj, key in slots):
            continue
        return {'status': 'skipped', 'reason': 'no_human_prose', 'message': index}
    return {'status': 'skipped', 'reason': 'no_reliable_human_boundary'}
