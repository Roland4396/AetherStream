"""One stream-mode implementation for every channel and protocol adapter."""
from __future__ import annotations

import asyncio
import codecs
import copy
import json
import time
import uuid
from contextlib import suppress

import anyio

from aetherstream.routing.channel_policy import termination_reason
from aetherstream.upstreams.errors import public_stream_error


def upstream_stream_mode(policy: dict, downstream_stream: bool) -> bool:
    cfg = policy.get('streaming', {})
    forced = cfg.get('force_upstream', 'auto')
    if forced not in {'auto', 'stream', 'json'}:
        raise ValueError('Invalid request mode')
    return downstream_stream if forced == 'auto' else forced == 'stream'


def completion(model: str, content: str, usage: dict | None = None,
               finish_reason: str | None = 'stop', tool_calls: list | None = None) -> dict:
    message = {'role': 'assistant', 'content': content}
    if tool_calls:
        message['tool_calls'] = tool_calls
    return {
        'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
        'created': int(time.time()), 'model': model,
        'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
        'usage': usage or {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
    }


def pack_shared_result(result: dict, shape: str) -> dict:
    """Expand/contract cache envelopes used by an already-running old slot."""
    if shape == 'chat':
        return result
    choice = (result.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    content = message.get('content') or ''
    raw = json.dumps(result, ensure_ascii=False)
    if shape == 'claude-chat':
        return {'content': content, 'raw': raw, 'response': result}
    if shape == 'claude-messages':
        return {'collected': [content, result.get('model'), result.get('usage') or {},
                              choice.get('finish_reason'), raw, message.get('tool_calls') or []]}
    raise ValueError('Unknown shared response envelope')


def unpack_shared_result(result: dict, shape: str, model: str) -> dict:
    if shape == 'chat':
        return result
    if shape == 'claude-chat':
        return result['response']
    if shape == 'claude-messages':
        content, actual, usage, finish, _, calls = result['collected']
        return completion(actual or model, content, usage, finish, calls)
    raise ValueError('Unknown shared response envelope')


async def collect_chat_stream(iterator, *, model: str) -> dict:
    """Collect canonical Chat SSE once, including tool arguments and usage.

    Protocol adapters own wire decoding. This collector owns the client-JSON
    mode; there is no separate per-channel nonstream implementation.
    """
    decoder = codecs.getincrementaldecoder('utf-8')()
    buffer = ''
    result = completion(model, '', finish_reason=None)
    message = result['choices'][0]['message']
    calls: dict[int, dict] = {}
    terminal = False

    def consume(line: str):
        nonlocal terminal
        if not line.startswith('data:'):
            return
        raw = line[5:].strip()
        if raw == '[DONE]':
            terminal = True
            return
        try:
            data = json.loads(raw)
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        if isinstance(data.get('error'), dict):
            raise RuntimeError(str(data['error'].get('message') or 'Upstream stream failed'))
        for key in ('id', 'model', 'created'):
            if data.get(key):
                result[key] = data[key]
        if isinstance(data.get('usage'), dict):
            result['usage'] = copy.deepcopy(data['usage'])
        choices = data.get('choices') or []
        if not choices or not isinstance(choices[0], dict):
            return
        choice = choices[0]
        if choice.get('finish_reason'):
            terminal = True
            result['choices'][0]['finish_reason'] = choice['finish_reason']
        delta = choice.get('delta') or choice.get('message') or {}
        if not isinstance(delta, dict):
            return
        for key in ('content', 'reasoning_content', 'reasoning'):
            if isinstance(delta.get(key), str):
                message[key] = message.get(key, '') + delta[key]
        for item in delta.get('tool_calls', []) or []:
            if not isinstance(item, dict):
                continue
            index = int(item.get('index', 0))
            call = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
            if item.get('id'):
                call['id'] = item['id']
            function = item.get('function') or {}
            for key in ('name', 'arguments'):
                if isinstance(function.get(key), str):
                    call['function'][key] += function[key]

    try:
        async for chunk in iterator:
            buffer += decoder.decode(chunk if isinstance(chunk, bytes) else chunk.encode())
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                consume(line.rstrip('\r'))
        buffer += decoder.decode(b'', final=True)
        if buffer:
            consume(buffer.rstrip('\r'))
    finally:
        with anyio.CancelScope(shield=True):
            close = getattr(iterator, 'aclose', None)
            if callable(close):
                await close()
    if calls:
        message['tool_calls'] = [calls[index] for index in sorted(calls)]
    if not terminal:
        if termination_reason() == 'early_stop':
            result['choices'][0]['finish_reason'] = 'early_stop'
            result['proxy_termination'] = {'reason': 'early_stop', 'upstream_closed': True}
        else:
            raise RuntimeError('Upstream stream closed without a terminal event')
    elif result['choices'][0]['finish_reason'] is None:
        result['choices'][0]['finish_reason'] = 'tool_calls' if calls else 'stop'
    return result


async def replay_json_completion(operation, *, model: str, heartbeat_seconds: float = 3.0, on_error=None):
    """Client SSE over a true JSON upstream; one request-owned task, never retry."""
    stream_id = 'chatcmpl-json-' + uuid.uuid4().hex
    created = int(time.time())

    def error_chunk(error, exception_type=None):
        if on_error is not None:
            # Observability must not replace the original wire error.
            with suppress(Exception):
                on_error(error, exception_type)
        return ('data: ' + json.dumps({'error': error}, ensure_ascii=False) + '\n\n').encode()

    def chunk(delta=None, finish_reason=None, usage=None):
        data = {'id': stream_id, 'object': 'chat.completion.chunk', 'created': created,
                'model': model, 'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish_reason}]}
        if usage is not None:
            data['usage'] = usage
        return ('data: ' + json.dumps(data, ensure_ascii=False) + '\n\n').encode()

    task = None
    try:
        yield chunk()
        task = asyncio.create_task(operation())
        while True:
            try:
                data = await asyncio.wait_for(asyncio.shield(task), heartbeat_seconds)
                break
            except asyncio.TimeoutError:
                if task.done():
                    # An operation's own TimeoutError is not a heartbeat expiry.
                    data = task.result()
                    break
                yield chunk()
        if isinstance(data.get('error'), dict):
            yield error_chunk(data['error'])
            return
        choices = data.get('choices') or []
        choice = choices[0] if choices else {}
        message = choice.get('message') or {}
        model = str(data.get('model') or model)
        yield chunk({'role': 'assistant'})
        for field in ('reasoning_content', 'reasoning', 'content'):
            value = message.get(field)
            if isinstance(value, str):
                for start in range(0, len(value), 1200):
                    yield chunk({field: value[start:start + 1200]})
        for index, tool in enumerate(message.get('tool_calls', []) or []):
            yield chunk({'tool_calls': [{**tool, 'index': index}]})
        if choice.get('finish_reason') == 'early_stop':
            return  # Never fabricate a natural terminal event after a stop.
        yield chunk(finish_reason=choice.get('finish_reason') or 'stop', usage=data.get('usage'))
        yield b'data: [DONE]\n\n'
    except Exception as exc:
        # Headers/heartbeats have already committed HTTP 200. Report failure in
        # the SSE body and close, without a success finish or an automatic retry.
        # CancelledError and GeneratorExit intentionally bypass this handler.
        yield error_chunk(public_stream_error(exc), type(exc).__name__)
    finally:
        with anyio.CancelScope(shield=True):
            if task is not None:
                if not task.done():
                    task.cancel()
                # The operation error was already reported above. Re-awaiting
                # it during cleanup must not raise it a second time into ASGI,
                # nor replace a downstream cancellation/GeneratorExit.
                with suppress(asyncio.CancelledError, Exception):
                    await task
