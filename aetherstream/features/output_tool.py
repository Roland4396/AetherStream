"""Request-owned output envelope; tool definition and wording belong to config.

Only our injected call is unwrapped. Ordinary tools, plain replies, provider
errors, request modes and cancellation keep their existing owners.
"""
from __future__ import annotations

import copy
import json
import re


RECORD_PREFIX = '[OUTPUT_TOOL_ENVELOPE '


def validate_settings(config):
    if not isinstance(config, dict) or set(config) - {
            'enabled', 'tool', 'field', 'control_prompt', 'anchors', 'match', 'transports'}:
        raise ValueError('Invalid output_tool settings')
    if 'enabled' in config and not isinstance(config['enabled'], bool):
        raise ValueError('output_tool.enabled must be boolean')
    match = config.get('match', {})
    if not isinstance(match, dict) or set(match) - {'regex'}:
        raise ValueError('output_tool match must be a model regex')
    if 'regex' in match:
        if not isinstance(match['regex'], str):
            raise ValueError('output_tool regex must be a string')
        try:
            re.compile(match['regex'], re.IGNORECASE)
        except re.error as exc:
            raise ValueError('Invalid output_tool model regex') from exc
    if 'transports' in config and config['transports'] != ['chat']:
        raise ValueError('output_tool currently supports chat upstreams')
    if not config.get('enabled'):
        return
    tool = config.get('tool', {})
    function = tool.get('function', {}) if isinstance(tool, dict) else {}
    name, field = function.get('name'), config.get('field')
    if not isinstance(tool, dict) or tool.get('type') != 'function' or not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('output_tool requires an existing function definition')
    params = function.get('parameters', {})
    if (not isinstance(field, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', field)
            or params.get('type') != 'object'
            or params.get('properties', {}).get(field, {}).get('type') != 'string'
            or field not in params.get('required', [])):
        raise ValueError('output_tool requires a required string output field')
    if not isinstance(config.get('control_prompt'), str) or not 0 < len(config['control_prompt']) <= 8000:
        raise ValueError('output_tool requires the existing control prompt')
    anchors = config.get('anchors', [])
    if not isinstance(anchors, list) or not all(isinstance(x, str) and 0 < len(x) <= 256 for x in anchors):
        raise ValueError('Invalid output_tool anchors')
    if len(json.dumps(config)) > 20000:
        raise ValueError('Output tool settings too large')


def selected_settings(config, model, transport):
    if config is None:
        return None
    validate_settings(config)
    pattern = config.get('match', {}).get('regex')
    if (config.get('enabled') and pattern and transport in config.get('transports', ['chat'])
            and re.search(pattern, model, re.IGNORECASE)):
        return copy.deepcopy(config)
    return None


def prepare_output_tool(payload, config, *, log, trace_prefix=''):
    """Mirror the existing S喵 request builder, without browser/UI side effects."""
    validate_settings(config)
    if not config.get('enabled'):
        return None
    choice = payload.get('tool_choice')
    if choice not in (None, 'auto'):
        log(f'{trace_prefix}output_tool skipped=caller_tool_choice')
        return None
    messages = payload.get('messages')
    tools = payload.get('tools', [])
    if (not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages)
            or not isinstance(tools, list)):
        return None
    name = config['tool']['function']['name']
    if any(isinstance(t, dict) and t.get('function', {}).get('name') == name for t in tools):
        # Never take ownership of a tool already supplied by the caller.
        log(f'{trace_prefix}output_tool skipped=caller_owned_tool name={name}')
        return None
    payload['tools'] = copy.deepcopy(tools) + [copy.deepcopy(config['tool'])]
    payload['tool_choice'] = 'auto'
    anchors = config.get('anchors', [])
    index = next((i for i, m in enumerate(messages) if isinstance(m, dict)
                  and isinstance(m.get('content'), str)
                  and any(a in m['content'] for a in anchors)), None)
    if index is None:
        role = 'user' if messages[-1].get('role') == 'assistant' else 'system'
        payload['messages'] = messages + [{'role': role, 'content': config['control_prompt']}]
    else:
        payload['messages'] = messages[:index] + [{'role': 'system', 'content': config['control_prompt']}] + messages[index:]
    log(f'{trace_prefix}output_tool injected name={name} field={config["field"]}')
    return copy.deepcopy(config)


def matches_name(name, config):
    return isinstance(name, str) and name.rsplit(':', 1)[-1] == config['tool']['function']['name']


def record_marker(config):
    if config is None:
        return ''
    return RECORD_PREFIX + json.dumps({'name': config['tool']['function']['name'], 'field': config['field']}, separators=(',', ':')) + ']\n'


def record_settings(markers):
    for marker in markers:
        if marker.startswith(RECORD_PREFIX) and marker.endswith(']'):
            value = json.loads(marker[len(RECORD_PREFIX):-1])
            if set(value) == {'name', 'field'} and all(isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', v) for v in value.values()):
                return {'tool': {'function': {'name': value['name']}}, 'field': value['field']}
    return None


def unwrap_completion(data, config):
    if config is None or not isinstance(data, dict):
        return data
    result = copy.deepcopy(data)
    choices = result.get('choices')
    if not isinstance(choices, list):
        return result
    for choice in choices:
        message = choice.get('message', {})
        calls = message.get('tool_calls') or []
        ours = [c for c in calls if matches_name(c.get('function', {}).get('name'), config)]
        if not ours:
            continue
        if len(ours) != 1:
            raise ValueError('Output envelope called more than once')
        args = ours[0]['function'].get('arguments', '')
        try:
            value = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            if choice.get('finish_reason') != 'length' or not isinstance(args, str):
                raise
            # A token limit remains a token limit, not a fabricated successful
            # completion or a new retry. Deliver the same visible partial text.
            decoder = StringFieldDecoder(config['field'])
            value = {config['field']: decoder.feed(args)}
        text = value.get(config['field']) if isinstance(value, dict) else None
        if not isinstance(text, str):
            raise ValueError('Output envelope is missing its string output')
        message['content'] = text
        remaining = [c for c in calls if c not in ours]
        if remaining:
            message['tool_calls'] = remaining
        else:
            message.pop('tool_calls', None)
            if choice.get('finish_reason') in ('tool_calls', 'function_call'):
                choice['finish_reason'] = 'stop'
    return result


class StringFieldDecoder:
    """Incremental JSON string decoder with bounded unconsumed input."""
    def __init__(self, field):
        self.field = field
        self.buffer = ''
        self.phase = 'object'
        self.key = None
        self.raw = []
        self.text = []

    def feed(self, fragment):
        self.raw.append(fragment)
        self.buffer += fragment
        out = []
        b, i = self.buffer, 0
        while i < len(b):
            if self.phase == 'tail':
                i = len(b)
                break
            if self.phase != 'string':
                while i < len(b) and b[i].isspace():
                    i += 1
                if i == len(b):
                    break
            if self.phase == 'object':
                if b[i] != '{':
                    raise ValueError('Output envelope arguments must be an object')
                i += 1
                self.phase = 'key'
            elif self.phase == 'key':
                if b[i] in ',}':
                    i += 1
                    continue
                try:
                    self.key, size = json.JSONDecoder().raw_decode(b[i:])
                except ValueError:
                    break
                if not isinstance(self.key, str):
                    raise ValueError('Invalid output envelope key')
                i += size
                self.phase = 'colon'
            elif self.phase == 'colon':
                if b[i] != ':':
                    raise ValueError('Invalid output envelope colon')
                i += 1
                self.phase = 'value'
            elif self.phase == 'value':
                if self.key != self.field:
                    try:
                        _, size = json.JSONDecoder().raw_decode(b[i:])
                    except ValueError:
                        break
                    # Numbers can be split inside a token (1 / 23 / e / +4).
                    # Wait for the enclosing object's separator before skipping.
                    end = i + size
                    while end < len(b) and b[end].isspace():
                        end += 1
                    if end == len(b) or b[end] not in ',}':
                        break
                    i += size
                    self.phase = 'key'
                else:
                    if b[i] != '"':
                        raise ValueError('Output envelope field must be a string')
                    i += 1
                    self.phase = 'string'
            else:
                ch = b[i]
                if ch == '"':
                    i += 1
                    self.phase = 'tail'
                elif ch != '\\':
                    if ord(ch) < 32:
                        raise ValueError('Unescaped control character in output envelope')
                    out.append(ch)
                    i += 1
                else:
                    if i + 1 >= len(b):
                        break
                    escaped = b[i + 1]
                    if escaped != 'u':
                        escapes = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}
                        if escaped not in escapes:
                            raise ValueError('Invalid JSON escape in output envelope')
                        out.append(escapes[escaped])
                        i += 2
                    else:
                        if i + 6 > len(b):
                            break
                        code = int(b[i + 2:i + 6], 16)
                        size = 6
                        if 0xD800 <= code <= 0xDBFF:
                            if i + 12 > len(b):
                                break
                            if b[i + 6:i + 8] != '\\u':
                                raise ValueError('Invalid UTF-16 pair in output envelope')
                            low = int(b[i + 8:i + 12], 16)
                            if not 0xDC00 <= low <= 0xDFFF:
                                raise ValueError('Invalid UTF-16 pair in output envelope')
                            code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00
                            size = 12
                        elif 0xDC00 <= code <= 0xDFFF:
                            raise ValueError('Unpaired UTF-16 surrogate in output envelope')
                        out.append(chr(code))
                        i += size
        self.buffer = b[i:]
        text = ''.join(out)
        self.text.append(text)
        return text

    def finish(self):
        try:
            value = json.loads(''.join(self.raw))
        except ValueError as exc:
            # Do not let the transport's malformed-SSE JSON fallback swallow a
            # malformed *argument* and forward our private call to the client.
            raise ValueError('Invalid output envelope arguments') from exc
        if not isinstance(value, dict) or not isinstance(value.get(self.field), str):
            raise ValueError('Output envelope is missing its string output')
        if value[self.field] != ''.join(self.text):
            raise ValueError('Output envelope decoding mismatch')


class OutputToolStreamDecoder:
    """State is private to one response, choice and tool-call index."""
    def __init__(self, config):
        self.config = config
        self.states = {}

    def feed(self, data):
        result = copy.deepcopy(data)
        choices = result.get('choices')
        if not isinstance(choices, list):
            return result
        for choice in choices:
            state = self.states.setdefault(choice.get('index', 0), {'calls': {}, 'own': False, 'client': False})
            delta = choice.get('delta')
            if not isinstance(delta, dict):
                continue
            kept, text = [], []
            for call in delta.get('tool_calls', []) or []:
                item = state['calls'].setdefault(call.get('index', 0), {'name': '', 'kind': None, 'pending': [], 'decoder': None})
                function = call.get('function', {})
                if item['kind'] == 'client':
                    kept.append(call)
                    continue
                name = function.get('name') or ''
                if name and name != item['name']:
                    item['name'] += name
                item['pending'].append(call)
                if item['kind'] is None and item['name']:
                    target = self.config['tool']['function']['name']
                    if matches_name(item['name'], self.config):
                        if state['own']:
                            raise ValueError('Output envelope called more than once')
                        item['kind'] = 'own'
                        item['decoder'] = StringFieldDecoder(self.config['field'])
                        state['own'] = True
                    elif not target.startswith(item['name'].rsplit(':', 1)[-1]):
                        item['kind'] = 'client'
                        state['client'] = True
                if item['kind'] == 'client':
                    kept.extend(item['pending'])
                    item['pending'] = []
                elif item['kind'] == 'own':
                    for pending in item['pending']:
                        fragment = pending.get('function', {}).get('arguments') or ''
                        if fragment:
                            text.append(item['decoder'].feed(fragment))
                    item['pending'] = []
            finish = choice.get('finish_reason')
            if finish is not None:
                for item in state['calls'].values():
                    if item['kind'] is None:
                        kept.extend(item['pending'])
                        state['client'] = state['client'] or bool(item['pending'])
                        item['pending'] = []
                    if item['kind'] == 'own' and finish in ('tool_calls', 'function_call', 'stop'):
                        item['decoder'].finish()
                if state['own'] and not state['client'] and finish in ('tool_calls', 'function_call'):
                    choice['finish_reason'] = 'stop'
            if 'tool_calls' in delta:
                if kept:
                    delta['tool_calls'] = kept
                else:
                    delta.pop('tool_calls')
            # Match the existing wrapper: once its output call starts, use its
            # field rather than simultaneously displaying another copy outside.
            if state['own']:
                delta.pop('content', None)
            emitted = ''.join(text)
            if emitted:
                delta['content'] = emitted
        return result
