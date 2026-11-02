"""Separate in-flight response sharing from replaying a completed inference.

Reasoning/usage alone is not a usable answer. Preserve the wire payload for the
current waiters, but never let an empty, refused or incomplete answer prevent a
later, intentional retry. Opaque non-inference values retain legacy semantics.
"""
from __future__ import annotations


_COMPLETE = {'stop', 'end_turn', 'stop_sequence', 'tool_calls', 'tool_use', 'function_call'}


def _text(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(isinstance(part, dict) and part.get('type') in {'text', 'output_text'}
                   and isinstance(part.get('text'), str) and bool(part['text'].strip())
                   for part in value)
    return False


def _tools(value) -> bool:
    if not isinstance(value, list):
        return False
    for call in value:
        if not isinstance(call, dict):
            continue
        function = call.get('function')
        if call.get('name') or (isinstance(function, dict) and function.get('name')):
            return True
    return False


def _complete(reason) -> bool:
    return reason is None or str(reason).lower() in _COMPLETE


def response_can_replay(result) -> bool:
    """Recognize current and schema-1 legacy response envelopes, without edits."""
    if not isinstance(result, dict):
        return True  # SharedRuntimeState also serves non-inference operations.
    if result.get('error') or result.get('refusal'):
        return False
    details = result.get('stop_details')
    if isinstance(details, dict) and details.get('type') == 'refusal':
        return False
    if result.get('status') in {'failed', 'incomplete', 'cancelled'}:
        return False
    if isinstance(result.get('response'), dict):
        return response_can_replay(result['response'])
    if 'collected' in result:
        collected = result['collected']
        return (isinstance(collected, (list, tuple)) and len(collected) >= 6
                and _complete(collected[3]) and (_text(collected[0]) or _tools(collected[5])))
    if 'choices' in result:
        choices = result['choices']
        if not isinstance(choices, list) or not choices:
            return False
        for choice in choices:
            if not isinstance(choice, dict) or not _complete(choice.get('finish_reason')):
                return False
            message = choice.get('message') or {}
            if not isinstance(message, dict) or message.get('refusal'):
                return False
            audio = message.get('audio')
            function = message.get('function_call')
            if not (_text(message.get('content')) or _text(choice.get('text'))
                    or _tools(message.get('tool_calls'))
                    or (isinstance(function, dict) and function.get('name'))
                    or (isinstance(audio, dict) and audio.get('data'))):
                return False
        return True
    if result.get('type') == 'message' or 'stop_reason' in result:
        content = result.get('content')
        return _complete(result.get('stop_reason')) and (_text(content) or _tools([
            part for part in content if isinstance(part, dict)
            and part.get('type') in {'tool_use', 'server_tool_use'}
        ] if isinstance(content, list) else []))
    return True
