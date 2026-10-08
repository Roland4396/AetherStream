import copy
import json
import time
import uuid
from typing import Any

import httpx

from aetherstream.observability.refusals import AnthropicRefusalDiagnostics

from aetherstream.upstreams.route_logging import format_account_pool_route, scope_stop_detection

from aetherstream.features.terminal_tool import TERMINAL_TOOL_NAME

from .protocol import (
    _append_raw_line,
    _extract_anthropic_sse_text,
    _map_stop_reason,
    _normalize_usage,
)
from .transport import _close_upstream_stream, _prime_code_cli_connection, _stream_with_oauth_recovery
from .types import AnthropicMessagesDeps
from .cache import _build_claude_cache_post_keepalive_key, _cancel_claude_cache_post_keepalive


class _ToolCallAccumulator:
    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def start(self, index: int, block: dict[str, Any]) -> None:
        initial_input = block.get("input")
        arguments = ""
        if initial_input not in (None, {}):
            arguments = json.dumps(initial_input, ensure_ascii=False, separators=(",", ":"))
        self._calls[index] = {
            "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex}"),
            "type": "function",
            "function": {
                "name": str(block.get("name") or ""),
                "arguments": arguments,
            },
        }

    def append_delta(self, index: int, delta: dict[str, Any]) -> bool:
        if delta.get("type") != "input_json_delta":
            return False
        call = self._calls.get(index)
        if call is None:
            return False
        partial = delta.get("partial_json")
        if isinstance(partial, str):
            call["function"]["arguments"] += partial
        return True

    def as_openai(self) -> list[dict[str, Any]]:
        calls = [copy.deepcopy(call) for _index, call in sorted(self._calls.items())]
        for call in calls:
            if not call["function"]["arguments"]:
                call["function"]["arguments"] = "{}"
        return calls


async def collect_anthropic_messages_as_chat_completion(
    *,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    messages: list,
    trace_id: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: AnthropicMessagesDeps,
    upstream_stream: bool = True,
) -> tuple[str, str, dict[str, Any], str, str, list[dict[str, Any]]]:
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    request_data = dict(request_data)
    request_data["stream"] = upstream_stream
    raw_sse_lines: list[str] = []
    raw_sse_size = 0
    raw_sse_truncated = False
    full_content = ""
    model_name = model
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    saw_message_stop = False
    first_data_time: float | None = None
    line_count = 0
    data_line_count = 0
    finish_status = "unknown"
    refusal_diagnostics = AnthropicRefusalDiagnostics()
    tool_calls = _ToolCallAccumulator()

    try:
        async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
            await _prime_code_cli_connection(
                client=client,
                url=url,
                deps=deps,
                trace_prefix=trace_prefix,
            )
            deps.log(f"{trace_prefix}claude_collect_request_start elapsed={deps.fmt_ms(request_t0)}")
            upstream_t0 = time.perf_counter()
            async with _stream_with_oauth_recovery(client=client, url=url, request_data=request_data,
                                                  headers=headers, deps=deps, trace_prefix=trace_prefix) as response:
                deps = scope_stop_detection(deps, response, url, trace_prefix)
                deps.log(
                    f"{trace_prefix}claude_collect_headers status={response.status_code} "
                    f"{format_account_pool_route(response)} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    error_text = (await response.aread()).decode(errors="replace")[:4000]
                    raise RuntimeError(f"Claude upstream HTTP {response.status_code}: {error_text}")

                if not upstream_stream:
                    await response.aread()
                    data = response.json()
                    refusal_diagnostics.observe(data)
                    if not isinstance(data, dict) or data.get('type') != 'message' or not isinstance(data.get('content'), list):
                        raise RuntimeError(f"Claude upstream non-stream error: {response.text[:4000]}")
                    model_name = str(data.get('model') or model)
                    usage = _normalize_usage(data.get('usage'))
                    stop_reason = data.get('stop_reason')
                    for index, block in enumerate(data['content']):
                        if not isinstance(block, dict):
                            continue
                        if block.get('type') == 'tool_use':
                            if block.get('name') == TERMINAL_TOOL_NAME:
                                stop_reason = 'tool_use' if tool_calls.as_openai() else 'end_turn'
                                break
                            tool_calls.start(index, block)
                        elif block.get('type') in ('text', 'thinking'):
                            value = block.get('text' if block['type'] == 'text' else 'thinking')
                            full_content += value if isinstance(value, str) else ''
                            if deps.has_stop_tag(full_content):
                                full_content = full_content[:deps.find_stop_tag(full_content)]
                                stop_reason = 'end_turn'
                                break
                    finish_status = 'nonstream_json'
                    if deps.retire_refused_session(data.get('stop_reason'), trace_prefix):
                        _cancel_claude_cache_post_keepalive(
                            post_key=_build_claude_cache_post_keepalive_key(model),
                            deps=deps, trace_prefix=trace_prefix, reason='upstream_refusal',
                        )
                    return (full_content, model_name, usage, _map_stop_reason(stop_reason),
                            response.text, tool_calls.as_openai())

                async for line in response.aiter_lines():
                    line_count += 1
                    raw_sse_size, raw_sse_truncated = _append_raw_line(
                        raw_sse_lines,
                        line,
                        raw_size=raw_sse_size,
                        raw_truncated=raw_sse_truncated,
                        max_raw_sse_bytes=max_raw_sse_bytes,
                    )

                    if not line or line.startswith("event:") or not line.startswith("data: "):
                        continue

                    data_line_count += 1
                    if first_data_time is None:
                        first_data_time = time.perf_counter()
                        deps.log(
                            f"{trace_prefix}claude_collect_first_data "
                            f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                        )

                    payload = line[6:].strip()
                    if not payload:
                        continue

                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    refusal_diagnostics.observe(data)

                    if data.get("type") == "error":
                        err = data.get("error") or {}
                        err_msg = err.get("message") or json.dumps(data, ensure_ascii=False)
                        finish_status = "upstream_error_event_seen_continue"
                        deps.log(
                            f"{trace_prefix}claude_collect_error_event_ignored_continue "
                            f"out_chars={len(full_content)} err={str(err_msg)[:500]}"
                        )
                        continue

                    event_type = data.get("type", "")

                    if event_type == "message_start":
                        message = data.get("message") or {}
                        if message.get("model"):
                            model_name = str(message["model"])
                        usage = _normalize_usage(message.get("usage"), usage)
                        continue

                    if event_type == "content_block_start":
                        block = data.get("content_block") or {}
                        if block.get("type") == "tool_use" and block.get("name") == TERMINAL_TOOL_NAME:
                            finish_status = "terminal_tool"
                            raw_response = "\n".join(raw_sse_lines)
                            collected_tool_calls = tool_calls.as_openai()
                            await _close_upstream_stream(
                                response=response,
                                client=client,
                                deps=deps,
                                trace_prefix=trace_prefix,
                                label="terminal_tool",
                                reason="claude_collect_terminal_tool",
                                started_at=request_t0,
                                line_count=line_count,
                                data_line_count=data_line_count,
                                out_chars=len(full_content),
                            )
                            return (
                                full_content,
                                model_name,
                                usage,
                                "tool_calls" if collected_tool_calls else "stop",
                                raw_response,
                                collected_tool_calls,
                            )
                        if block.get("type") == "tool_use":
                            tool_calls.start(int(data.get("index") or 0), block)
                        continue

                    if event_type == "content_block_delta":
                        raw_delta = data.get("delta") or {}
                        if isinstance(raw_delta, dict) and tool_calls.append_delta(
                            int(data.get("index") or 0),
                            raw_delta,
                        ):
                            continue
                        delta = _extract_anthropic_sse_text(data)
                        if not delta:
                            continue

                        full_content += delta
                        if deps.has_stop_tag(full_content):
                            full_content = full_content[:deps.find_stop_tag(full_content)]
                            raw_response = "\n".join(raw_sse_lines)
                            finish_status = "early_stop_tag"
                            await _close_upstream_stream(
                                response=response,
                                client=client,
                                deps=deps,
                                trace_prefix=trace_prefix,
                                label="early_stop",
                                reason="claude_collect_stop_tag",
                                started_at=request_t0,
                                line_count=line_count,
                                data_line_count=data_line_count,
                                out_chars=len(full_content),
                            )
                            return (
                                full_content,
                                model_name,
                                usage,
                                "stop",
                                raw_response,
                                tool_calls.as_openai(),
                            )
                        continue

                    if event_type == "message_delta":
                        delta = data.get("delta") or {}
                        if isinstance(delta, dict):
                            stop_reason = delta.get("stop_reason") or stop_reason
                        usage = _normalize_usage(data.get("usage"), usage)
                        continue

                    if event_type == "message_stop":
                        saw_message_stop = True
                        finish_status = f"message_stop:{stop_reason or 'stop'}"
                        if deps.retire_refused_session(stop_reason, trace_prefix):
                            _cancel_claude_cache_post_keepalive(
                                post_key=_build_claude_cache_post_keepalive_key(model),
                                deps=deps, trace_prefix=trace_prefix, reason='upstream_refusal',
                            )
                        break

        if not saw_message_stop:
            finish_status = "stream_end_without_message_stop"
            raise RuntimeError("Claude upstream stream closed without message_stop")

        raw_response = "\n".join(raw_sse_lines)
        return (
            full_content,
            model_name,
            usage,
            _map_stop_reason(stop_reason),
            raw_response,
            tool_calls.as_openai(),
        )
    finally:
        deps.log(
            f"{trace_prefix}claude_collect_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} "
            f"out_chars={len(full_content)} usage_total={usage.get('total_tokens', 0)}"
            f"{refusal_diagnostics.log_suffix()}"
        )


async def collect_anthropic_messages_response(
    *,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: AnthropicMessagesDeps,
    trace_id: str = "",
    upstream_stream: bool = True,
) -> tuple[dict[str, Any], str]:
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    request_data = dict(request_data)
    request_data["stream"] = upstream_stream
    raw_sse_lines: list[str] = []
    raw_sse_size = 0
    raw_sse_truncated = False
    full_content = ""
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    stop_sequence: str | None = None
    message_id = f"msg_{uuid.uuid4().hex}"
    role = "assistant"
    model_name = model
    saw_message_stop = False
    first_data_time: float | None = None
    line_count = 0
    data_line_count = 0
    finish_status = "unknown"
    refusal_diagnostics = AnthropicRefusalDiagnostics()

    try:
        async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
            await _prime_code_cli_connection(
                client=client,
                url=url,
                deps=deps,
                trace_prefix=trace_prefix,
            )
            deps.log(f"{trace_prefix}anthropic_collect_request_start elapsed={deps.fmt_ms(request_t0)}")
            upstream_t0 = time.perf_counter()
            async with _stream_with_oauth_recovery(client=client, url=url, request_data=request_data,
                                                  headers=headers, deps=deps, trace_prefix=trace_prefix) as response:
                deps = scope_stop_detection(deps, response, url, trace_prefix)
                deps.log(
                    f"{trace_prefix}anthropic_collect_headers status={response.status_code} "
                    f"{format_account_pool_route(response)} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    error_text = (await response.aread()).decode(errors="replace")[:4000]
                    raise RuntimeError(f"Anthropic upstream HTTP {response.status_code}: {error_text}")

                if not upstream_stream:
                    await response.aread()
                    data = response.json()
                    refusal_diagnostics.observe(data)
                    if not isinstance(data, dict) or data.get('type') != 'message' or not isinstance(data.get('content'), list):
                        raise RuntimeError(f"Anthropic upstream non-stream error: {response.text[:4000]}")
                    finish_status = 'nonstream_json'
                    return data, response.text

                async for line in response.aiter_lines():
                    line_count += 1
                    raw_sse_size, raw_sse_truncated = _append_raw_line(
                        raw_sse_lines,
                        line,
                        raw_size=raw_sse_size,
                        raw_truncated=raw_sse_truncated,
                        max_raw_sse_bytes=max_raw_sse_bytes,
                    )

                    if not line or line.startswith("event:") or not line.startswith("data: "):
                        continue

                    data_line_count += 1
                    if first_data_time is None:
                        first_data_time = time.perf_counter()
                        deps.log(
                            f"{trace_prefix}anthropic_collect_first_data "
                            f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                        )

                    payload = line[6:].strip()
                    if not payload:
                        continue

                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    refusal_diagnostics.observe(data)

                    if data.get("type") == "error":
                        err = data.get("error") or {}
                        err_msg = err.get("message") or json.dumps(data, ensure_ascii=False)
                        finish_status = "upstream_error_event_seen_continue"
                        deps.log(
                            f"{trace_prefix}claude_collect_error_event_ignored_continue "
                            f"out_chars={len(full_content)} err={str(err_msg)[:500]}"
                        )
                        continue

                    event_type = data.get("type", "")

                    if event_type == "message_start":
                        message = data.get("message") or {}
                        if message.get("id"):
                            message_id = str(message["id"])
                        if message.get("role"):
                            role = str(message["role"])
                        if message.get("model"):
                            model_name = str(message["model"])
                        usage = _normalize_usage(message.get("usage"), usage)
                        continue

                    if event_type == "content_block_start":
                        block = data.get("content_block") or {}
                        if block.get("type") == "tool_use" and block.get("name") == TERMINAL_TOOL_NAME:
                            finish_status = "terminal_tool"
                            response_payload = {
                                "id": message_id,
                                "type": "message",
                                "role": role,
                                "model": model_name,
                                "content": [{"type": "text", "text": full_content}],
                                "stop_reason": "end_turn",
                                "stop_sequence": None,
                                "usage": usage,
                            }
                            await _close_upstream_stream(
                                response=response,
                                client=client,
                                deps=deps,
                                trace_prefix=trace_prefix,
                                label="terminal_tool",
                                reason="anthropic_collect_terminal_tool",
                                started_at=request_t0,
                                line_count=line_count,
                                data_line_count=data_line_count,
                                out_chars=len(full_content),
                            )
                            return response_payload, "\n".join(raw_sse_lines)
                        continue

                    if event_type == "content_block_delta":
                        delta = _extract_anthropic_sse_text(data)
                        if not delta:
                            continue
                        full_content += delta
                        if deps.has_stop_tag(full_content):
                            full_content = full_content[:deps.find_stop_tag(full_content)]
                            response_payload = {
                                "id": message_id,
                                "type": "message",
                                "role": role,
                                "model": model_name,
                                "content": [{"type": "text", "text": full_content}],
                                "stop_reason": "stop_sequence",
                                "stop_sequence": None,
                                "usage": usage,
                            }
                            finish_status = "early_stop_tag"
                            await _close_upstream_stream(
                                response=response,
                                client=client,
                                deps=deps,
                                trace_prefix=trace_prefix,
                                label="early_stop",
                                reason="anthropic_collect_stop_tag",
                                started_at=request_t0,
                                line_count=line_count,
                                data_line_count=data_line_count,
                                out_chars=len(full_content),
                            )
                            return response_payload, "\n".join(raw_sse_lines)
                        continue

                    if event_type == "message_delta":
                        delta = data.get("delta") or {}
                        if isinstance(delta, dict):
                            stop_reason = delta.get("stop_reason") or stop_reason
                            stop_sequence = delta.get("stop_sequence") or stop_sequence
                        usage = _normalize_usage(data.get("usage"), usage)
                        continue

                    if event_type == "message_stop":
                        saw_message_stop = True
                        finish_status = f"message_stop:{stop_reason or 'stop'}"
                        break

        if not saw_message_stop:
            finish_status = "stream_end_without_message_stop"
            raise RuntimeError("Anthropic upstream stream closed without message_stop")

        response_payload = {
            "id": message_id,
            "type": "message",
            "role": role,
            "model": model_name,
            "content": [{"type": "text", "text": full_content}],
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": usage,
        }
        return response_payload, "\n".join(raw_sse_lines)
    finally:
        deps.log(
            f"{trace_prefix}anthropic_collect_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} "
            f"out_chars={len(full_content)} usage_total={usage.get('total_tokens', 0)}"
            f"{refusal_diagnostics.log_suffix()}"
        )
