"""Opt-in request-local port of the user's v3.2 Gemini text perturbation script.

This changes outgoing text; it neither verifies nor guarantees provider routing.
Apply only inside the upstream transport, AFTER exact-request ownership/dedupe.
There is deliberately no MVU/variable-tag exemption. Tools and signed protocol
parts remain protocol data, not text to rewrite. No state outlives the request.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable

from aetherstream.features.input_transforms import settings_from as input_transform_settings, transform_request_input


LETTERS = 'bcdfghjklmnpqrstvxzBCDFGHJKLMNPQRSTVXZ'
DIGITS = '0123456789'
MARKER = re.compile(r'(?:\{\{|\[|<)?[ \t]*(?:izumi[ \t_-]*|gemini[ \t_-]*)?anti[ \t_-]*route[ \t]*(?:\}\}|\]|/?>)?', re.I)
VARIANTS = json.loads(Path(__file__).with_name('gemini_anti_route_variants.json').read_text())
DEFAULTS = {'enabled': False, 'mode': 'deep', 'headCommentMin': 18, 'headCommentMax': 38,
            'turnCommentMin': 8, 'turnCommentMax': 22, 'charMin': 16, 'charMax': 42}
BOUNDS = {'headCommentMin': (5, 100), 'headCommentMax': (10, 120),
          'turnCommentMin': (3, 60), 'turnCommentMax': (6, 80),
          'charMin': (12, 100), 'charMax': (18, 120)}


def config_from(value: Any) -> dict:
    value = value if isinstance(value, dict) else {}
    result = dict(DEFAULTS)
    result['enabled'] = value.get('enabled') is True
    result['mode'] = 'standard' if value.get('mode') == 'standard' else 'deep'
    for key, (low, high) in BOUNDS.items():
        raw = value.get(key, result[key])
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
            result[key] = max(low, min(high, int(raw)))
    for prefix in ('headComment', 'turnComment', 'char'):
        result[prefix + 'Max'] = max(result[prefix + 'Min'], result[prefix + 'Max'])
    try:
        result['input_transform'] = input_transform_settings(value.get('input_transform'))
    except ValueError:
        result['input_transform'] = input_transform_settings(None)
        result['input_transform_error'] = 'invalid_mode'
    return result


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


class GeminiAntiRouteSession:
    """One instance per upstream deps/request; retries reuse the same mutation."""
    def __init__(self, rng=None):
        self.rng = rng or random.SystemRandom()
        self._config = None
        self._results: dict[str, dict] = {}
        self._completed: set[str] = set()

    def token(self, size: int) -> str:
        chars = [self.rng.choice(DIGITS if self.rng.random() < .30 else LETTERS) for _ in range(size)]
        if not any(char in DIGITS for char in chars):
            chars[self.rng.randrange(size)] = self.rng.choice(DIGITS)
        if not any(char in LETTERS for char in chars):
            chars[self.rng.randrange(size)] = self.rng.choice(LETTERS)
        return ''.join(chars)

    def block(self, low: int, high: int, cfg: dict, markdown=True) -> str:
        html = ''.join('<!-- ' + self.token(self.rng.randint(cfg['charMin'], cfg['charMax'])) + ' -->'
                       for _ in range(self.rng.randint(low, high)))
        if not markdown:
            return html
        return html + '\n' + '\n'.join('[//]: # (' + self.token(self.rng.randint(18, 36)) + ')'
                                      for _ in range(self.rng.randint(1, 3)))

    def jitter(self, text: str) -> str:
        if len(text) < 16:
            return text
        for needle, choices in VARIANTS:
            if needle in text:
                text = text.replace(needle, self.rng.choice(choices), 1)
        lines = text.split('\n')
        for index, line in enumerate(lines[:-1]):
            trimmed = line.rstrip()
            if (len(trimmed) > 6 and not re.search(r'[>}%`~]$', trimmed)
                    and not re.match(r'^\s*(?:<|```|~~~|\{\{)', trimmed)
                    and self.rng.random() < .55):
                lines[index] = trimmed + ' ' * self.rng.randint(1, 3)
        return '\n'.join(lines)

    def text(self, text: str, prefix: str, suffix: str, cfg: dict, micro=True) -> str:
        text = MARKER.sub(lambda _: self.block(15, 45, cfg), text)
        if micro and cfg['mode'] == 'deep':
            text = self.jitter(text)
        return prefix + text + suffix

    def messages(self, messages: list, cfg: dict) -> int:
        eligible = [i for i, m in enumerate(messages) if isinstance(m, dict)
                    and isinstance(m.get('content'), str) and m['content'].strip()
                    and m.get('role') not in ('tool', 'function') and not m.get('tool_calls')
                    and not m.get('function_call')]
        tail = set(eligible[-2:])
        first = set()
        changed = 0
        for index, message in enumerate(messages):
            if (not isinstance(message, dict) or message.get('role') in ('tool', 'function')
                    or message.get('tool_calls') or message.get('function_call')):
                continue
            content, role = message.get('content'), message.get('role')
            if isinstance(content, str) and content.strip():
                advice = bool(re.search(r'<advice\b', content, re.I)) or '按照advice继续' in content
                prefix = suffix = ''
                if role in ('system', 'user', 'assistant') and role not in first:
                    first.add(role)
                    stem = 'headComment' if role == 'system' else 'turnComment'
                    prefix = self.block(cfg[stem + 'Min'], cfg[stem + 'Max'], cfg) + '\n'
                elif cfg['mode'] == 'deep':
                    if index in tail:
                        prefix = self.block(4, 10, cfg, False) + '\n'
                        suffix = '\n' + self.block(cfg['turnCommentMin'], cfg['turnCommentMax'], cfg)
                    elif len(content) >= 60 and role in ('system', 'user') and not advice:
                        prefix = self.block(4, 10, cfg) + '\n'
                result = self.text(content, prefix, suffix, cfg, not advice)
                if result != content:
                    message['content'] = result
                    changed += 1
            elif isinstance(content, list):
                touched = False
                for part_index, part in enumerate(content):
                    if (not isinstance(part, dict) or part.get('type') != 'text'
                            or not isinstance(part.get('text'), str) or not part['text'].strip()):
                        continue
                    prefix = ''
                    if role == 'system' and 'system' not in first and part_index == 0:
                        first.add('system')
                        prefix = self.block(cfg['headCommentMin'], cfg['headCommentMax'], cfg) + '\n'
                    result = self.text(part['text'], prefix, '', cfg)
                    if result != part['text']:
                        part['text'] = result
                        touched = True
                changed += int(touched)
        return changed

    @staticmethod
    def unsigned_text(part: Any) -> bool:
        return (isinstance(part, dict) and isinstance(part.get('text'), str) and bool(part['text'].strip())
                and not any(key in part for key in ('thoughtSignature', 'thought_signature', 'functionCall',
                                                    'functionResponse', 'inlineData', 'fileData')))

    def native(self, payload: dict, cfg: dict) -> int:
        changed = 0
        system = payload.get('systemInstruction') or payload.get('system_instruction')
        if isinstance(system, dict) and isinstance(system.get('parts'), list):
            for index, part in enumerate(system['parts']):
                if not self.unsigned_text(part):
                    continue
                prefix = (self.block(cfg['headCommentMin'], cfg['headCommentMax'], cfg) + '\n' if index == 0
                          else self.block(4, 10, cfg, False) + '\n' if cfg['mode'] == 'deep' else '')
                updated = self.text(part['text'], prefix, '', cfg)
                changed += int(updated != part['text'])
                part['text'] = updated
        turns = payload.get('contents')
        if isinstance(turns, list):
            for index, turn in enumerate(turns):
                if not isinstance(turn, dict) or not isinstance(turn.get('parts'), list):
                    continue
                for part_index, part in enumerate(turn['parts']):
                    if not self.unsigned_text(part):
                        continue
                    advice = bool(re.search(r'<advice\b', part['text'], re.I)) or '按照advice继续' in part['text']
                    prefix = suffix = ''
                    if index <= 1 and part_index == 0:
                        prefix = self.block(cfg['turnCommentMin'], cfg['turnCommentMax'], cfg) + '\n'
                    elif cfg['mode'] == 'deep':
                        if index >= len(turns) - 2 and part_index == len(turn['parts']) - 1:
                            prefix = self.block(4, 9, cfg, False) + '\n'
                            suffix = '\n' + self.block(cfg['turnCommentMin'], cfg['turnCommentMax'], cfg)
                        elif len(part['text']) >= 60 and not advice:
                            prefix = self.block(4, 9, cfg, False) + '\n'
                    updated = self.text(part['text'], prefix, suffix, cfg, not advice)
                    changed += int(updated != part['text'])
                    part['text'] = updated
        return changed

    def sampling(self, payload: dict) -> None:
        for key in ('openrouter_allow_fallbacks', 'openrouter_use_fallback'):
            if key in payload:
                payload[key] = False
        def alter(container, key, upper):
            value = container.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                # Preserve the supplied script's 0.1 floor, including temperature=0.
                container[key] = round(max(.1, min(upper, value + self.rng.random() * .004 - .002)), 4)
        alter(payload, 'temperature', 2.)
        alter(payload, 'top_p', 1.)
        if isinstance(payload.get('generationConfig'), dict):
            alter(payload['generationConfig'], 'temperature', 2.)
            alter(payload['generationConfig'], 'topP', 1.)

    def apply(self, payload: dict, *, model: str, lookup: Callable | None, log: Callable,
              trace_prefix='', native_transport=False) -> int:
        if not native_transport and 'gemini' not in str(model or payload.get('model') or '').lower():
            return 0
        if self._config is None:
            self._config = config_from(lookup('injections', 'gemini_anti_route') if callable(lookup) else None)
        cfg = self._config
        if not cfg['enabled']:
            return 0
        original = fingerprint(payload)
        if original in self._completed:
            return 0
        if original in self._results:
            payload.clear()
            payload.update(copy.deepcopy(self._results[original]))
            return 0
        # Work transactionally; malformed optional data must not partly rewrite a request.
        updated = copy.deepcopy(payload)
        before_chars = len(json.dumps(updated, ensure_ascii=False))
        if cfg.get('input_transform_error'):
            log(f'{trace_prefix}gemini_anti_route input_transform skipped reason=invalid_config')
        try:
            transform = transform_request_input(updated, cfg['input_transform'], self.rng)
        except (ImportError, ValueError, RuntimeError, UnicodeError) as exc:
            transform = None
            log(f'{trace_prefix}gemini_anti_route input_transform skipped error_type={type(exc).__name__}')
        if transform:
            details = ' '.join(f'{key}={value}' for key, value in transform.items())
            log(f'{trace_prefix}gemini_anti_route input_transform {details}')
        changed = self.messages(updated['messages'], cfg) if isinstance(updated.get('messages'), list) else 0
        if any(key in updated for key in ('contents', 'systemInstruction', 'system_instruction')):
            changed += self.native(updated, cfg)
        self.sampling(updated)
        result = fingerprint(updated)
        self._completed.add(result)
        self._results[original] = copy.deepcopy(updated)
        payload.clear()
        payload.update(updated)
        added = len(json.dumps(updated, ensure_ascii=False)) - before_chars
        log(f'{trace_prefix}gemini_anti_route applied mode={cfg["mode"]} model={model} '
            f'text_parts={changed} added_chars={added} routing_verified=false')
        return changed


def apply_gemini_anti_route(payload: dict, deps, *, model='', trace_prefix='', native_transport=False) -> int:
    session = getattr(deps, 'gemini_anti_route', None)
    if not isinstance(session, GeminiAntiRouteSession):
        return 0
    return session.apply(payload, model=model, lookup=getattr(deps, 'runtime_lookup', None),
                         log=deps.log, trace_prefix=trace_prefix, native_transport=native_transport)
