import asyncio
import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable
from urllib.parse import urlsplit

import httpx


@dataclass
class AnthropicUpstreamDeps:
    log: Callable[[str], None]
    save_request_log: Callable[..., None]
    build_openai_sse_error: Callable[[int, str, str], bytes]
    has_stop_tag: Callable[[str], bool]
    find_stop_tag: Callable[[str], int]
    fmt_ms: Callable[[float, float | None], str]
    release_caller: Callable[[str, str], None]


@dataclass
class _ClaudeCacheKeepaliveLoopState:
    in_flight: bool = False
    stop_requested: bool = False


@dataclass
class _ClaudeCachePostKeepaliveState:
    task: asyncio.Task | None = None
    started: bool = False
    in_flight: bool = False
    stop_requested: bool = False


_CLAUDE_CACHE_POST_KEEPALIVE_TASKS: dict[str, _ClaudeCachePostKeepaliveState] = {}


async def _close_upstream_stream(
    *,
    response: httpx.Response | None,
    client: httpx.AsyncClient | None,
    deps: "AnthropicUpstreamDeps",
    trace_prefix: str,
    label: str,
    reason: str,
    started_at: float,
    line_count: int,
    data_line_count: int,
    out_chars: int,
    reader_task: asyncio.Task | None = None,
) -> None:
    close_t0 = time.perf_counter()
    if reader_task is not None and not reader_task.done():
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    response_closed = False
    client_closed = False
    if response is not None:
        try:
            await response.aclose()
            response_closed = True
        except Exception as e:
            deps.log(
                f"{trace_prefix}{label}_upstream_close_response_error "
                f"reason={reason} type={type(e).__name__} err={e}"
            )
    if client is not None:
        try:
            await client.aclose()
            client_closed = True
        except Exception as e:
            deps.log(
                f"{trace_prefix}{label}_upstream_close_client_error "
                f"reason={reason} type={type(e).__name__} err={e}"
            )
    deps.log(
        f"{trace_prefix}{label}_upstream_close "
        f"reason={reason} response_closed={str(response_closed).lower()} "
        f"client_closed={str(client_closed).lower()} close_elapsed={deps.fmt_ms(close_t0)} "
        f"elapsed={deps.fmt_ms(started_at)} lines={line_count} "
        f"data_lines={data_line_count} out_chars={out_chars}"
    )


def _append_raw_line(
    raw_lines: list[str],
    line: str,
    *,
    raw_size: int,
    raw_truncated: bool,
    max_raw_sse_bytes: int,
) -> tuple[int, bool]:
    line_size = len(line.encode("utf-8", errors="replace")) + 1
    raw_lines.append(line)
    return raw_size + line_size, False


def _extract_anthropic_sse_delta(data: dict[str, Any]) -> tuple[str, str]:
    if data.get("type") != "content_block_delta":
        return "", ""
    delta = data.get("delta")
    if not isinstance(delta, dict):
        return "", ""
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        text = delta.get("text")
        return (text if isinstance(text, str) else ""), "content"
    if delta_type == "thinking_delta":
        thinking = delta.get("thinking")
        return (thinking if isinstance(thinking, str) else ""), "reasoning"
    return "", ""


def _extract_anthropic_sse_text(data: dict[str, Any]) -> str:
    text, _ = _extract_anthropic_sse_delta(data)
    return text


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "".join(text_parts)


def _normalize_usage(usage: dict[str, Any] | None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base or {})
    if isinstance(usage, dict):
        merged.update(usage)

    prompt_tokens = int(merged.get("input_tokens") or 0)
    completion_tokens = int(merged.get("output_tokens") or 0)
    total_tokens = int(merged.get("total_tokens") or (prompt_tokens + completion_tokens))
    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    normalized.update(merged)
    return normalized


def _has_cache_control(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("cache_control"), dict):
            return True
        return any(_has_cache_control(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_cache_control(item) for item in value)
    return False


def _build_claude_cache_post_keepalive_key(model: str) -> str:
    return str(model or "").strip().lower() or "unknown"


def _estimate_anthropic_text_chars(value: Any) -> int:
    if isinstance(value, dict):
        total = 0
        for key, child in value.items():
            if key == "text" and isinstance(child, str):
                total += len(child)
            elif key != "cache_control":
                total += _estimate_anthropic_text_chars(child)
        return total
    if isinstance(value, list):
        return sum(_estimate_anthropic_text_chars(item) for item in value)
    if isinstance(value, str):
        return len(value)
    return 0


def _find_last_cache_control_location(request_data: dict[str, Any]) -> tuple[str, int | None, int | None] | None:
    last: tuple[str, int | None, int | None] | None = None

    system_value = request_data.get("system")
    if isinstance(system_value, list):
        for block_index, block in enumerate(system_value):
            if _has_cache_control(block):
                last = ("system", None, block_index)
    elif system_value is not None and _has_cache_control(system_value):
        last = ("system", None, None)

    messages = request_data.get("messages")
    if isinstance(messages, list):
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for block_index, block in enumerate(content):
                    if _has_cache_control(block):
                        last = ("message", message_index, block_index)
            elif _has_cache_control(content):
                last = ("message", message_index, None)

    return last


def _build_claude_cache_only_request(request_data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build a keepalive request containing only the prefix up to cache_control."""
    original_chars = _estimate_anthropic_text_chars(request_data)
    messages_before = len(request_data.get("messages") or [])
    location = _find_last_cache_control_location(request_data)
    if location is None:
        return None, {
            "original_chars": original_chars,
            "trimmed_chars": 0,
            "saved_chars": original_chars,
            "location": "none",
            "message_index": -1,
            "block_index": -1,
            "messages_before": messages_before,
            "messages_after": 0,
        }

    kind, message_index, block_index = location
    trimmed = copy.deepcopy(request_data)

    if kind == "system":
        system_value = trimmed.get("system")
        if isinstance(system_value, list) and block_index is not None:
            trimmed["system"] = system_value[:block_index + 1]
        # Anthropic requests need at least one message.  Use a tiny tail after
        # the cached system prefix instead of replaying the real volatile tail.
        trimmed["messages"] = [{
            "role": "user",
            "content": [{"type": "text", "text": "."}],
        }]
    elif kind == "message" and message_index is not None:
        messages = trimmed.get("messages")
        if not isinstance(messages, list) or message_index >= len(messages):
            return None, {
                "original_chars": original_chars,
                "trimmed_chars": 0,
                "saved_chars": original_chars,
                "location": "invalid_message",
                "message_index": message_index if message_index is not None else -1,
                "block_index": block_index if block_index is not None else -1,
                "messages_before": messages_before,
                "messages_after": 0,
            }
        kept_messages = messages[:message_index + 1]
        last_message = kept_messages[-1]
        if isinstance(last_message, dict) and isinstance(last_message.get("content"), list) and block_index is not None:
            last_message["content"] = last_message["content"][:block_index + 1]
        trimmed["messages"] = kept_messages

    trimmed_chars = _estimate_anthropic_text_chars(trimmed)
    return trimmed, {
        "original_chars": original_chars,
        "trimmed_chars": trimmed_chars,
        "saved_chars": max(0, original_chars - trimmed_chars),
        "location": kind,
        "message_index": message_index if message_index is not None else -1,
        "block_index": block_index if block_index is not None else -1,
        "messages_before": messages_before,
        "messages_after": len(trimmed.get("messages") or []),
    }


def _build_claude_cache_fingerprint(model: str, request_data: dict[str, Any]) -> str:
    units: list[Any] = []
    last_cache_unit = -1

    def append_unit(kind: str, role: str, value: Any) -> None:
        nonlocal last_cache_unit
        units.append([kind, role, value])
        if _has_cache_control(value):
            last_cache_unit = len(units) - 1

    system_value = request_data.get("system")
    if isinstance(system_value, list):
        for block in system_value:
            append_unit("system", "", block)
    elif system_value is not None:
        append_unit("system", "", system_value)

    for message in request_data.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                append_unit("message", role, block)
        else:
            append_unit("message", role, content)

    prefix_units = units[:last_cache_unit + 1] if last_cache_unit >= 0 else units
    raw = json.dumps(
        {"model": model, "prefix_units": prefix_units},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _cancel_claude_cache_post_keepalive(
    *,
    post_key: str,
    deps: AnthropicUpstreamDeps,
    trace_prefix: str,
    reason: str,
) -> None:
    state = _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.get(post_key)
    task = state.task if isinstance(state, _ClaudeCachePostKeepaliveState) else None
    if task is None or task.done():
        _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.pop(post_key, None)
        return
    if state.started and state.in_flight:
        state.stop_requested = True
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_post_cancel_skip "
            f"key={post_key} reason=in_flight_stop_after_current requested_reason={reason}"
        )
        return
    if state.started:
        _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.pop(post_key, None)
        task.cancel()
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_post_cancel "
            f"key={post_key} reason={reason}_between_runs"
        )
        return
    _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.pop(post_key, None)
    task.cancel()
    deps.log(
        f"{trace_prefix}claude_cache_keepalive_post_cancel "
        f"key={post_key} reason={reason}"
    )


async def _run_claude_cache_keepalive_once(
    *,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    trace_id: str,
    deps: AnthropicUpstreamDeps,
    max_tokens: int = 1,
    first_data_timeout_sec: float = 60.0,
    close_after_data_events: int = 1,
) -> bool:
    keepalive_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    keepalive_request, cache_only_stats = _build_claude_cache_only_request(request_data)
    if not isinstance(keepalive_request, dict):
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_skip "
            f"reason=no_cache_prefix model={model}"
        )
        return False
    keepalive_request["stream"] = True
    keepalive_request["max_tokens"] = max(1, int(max_tokens or 1))
    close_after_events = max(1, int(close_after_data_events or 1))

    request_timeout = httpx.Timeout(
        connect=30.0,
        read=max(1.0, float(first_data_timeout_sec or 60.0)),
        write=30.0,
        pool=30.0,
    )

    try:
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_start "
            f"model={model} max_tokens={keepalive_request['max_tokens']} "
            f"first_data_timeout={first_data_timeout_sec:.1f}s "
            f"cache_only_location={cache_only_stats.get('location')} "
            f"cache_only_message_index={cache_only_stats.get('message_index', -1)} "
            f"cache_only_block_index={cache_only_stats.get('block_index', -1)} "
            f"messages={cache_only_stats.get('messages_after')}/{cache_only_stats.get('messages_before')} "
            f"text_chars={cache_only_stats.get('trimmed_chars')}/{cache_only_stats.get('original_chars')} "
            f"saved_chars={cache_only_stats.get('saved_chars', 0)}"
        )
        async with httpx.AsyncClient(timeout=request_timeout, http2=False) as client:
            await _prime_code_cli_connection(
                client=client,
                url=url,
                deps=deps,
                trace_prefix=trace_prefix,
            )
            upstream_t0 = time.perf_counter()
            async with client.stream("POST", url, json=keepalive_request, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}claude_cache_keepalive_headers "
                    f"status={response.status_code} elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")[:1000]
                    deps.log(
                        f"{trace_prefix}claude_cache_keepalive_http_error "
                        f"status={response.status_code} elapsed={deps.fmt_ms(keepalive_t0)} "
                        f"body={body!r}"
                    )
                    return False

                line_count = 0
                data_line_count = 0
                non_ping_data_count = 0
                final_usage: dict[str, Any] = {}
                last_event_type = "-"
                async for line in response.aiter_lines():
                    line_count += 1
                    if not line.startswith("data: "):
                        continue
                    data_line_count += 1
                    payload = line[6:].strip()
                    event_type = "-"
                    if payload:
                        try:
                            data = json.loads(payload)
                            if isinstance(data, dict):
                                event_type = str(data.get("type") or "-")
                                usage_value = (
                                    data.get("usage")
                                    or (data.get("message") if isinstance(data.get("message"), dict) else {}).get("usage")
                                    or (data.get("delta") if isinstance(data.get("delta"), dict) else {}).get("usage")
                                )
                                if isinstance(usage_value, dict):
                                    final_usage = usage_value
                        except json.JSONDecodeError:
                            event_type = "json_error"
                    if event_type == "ping":
                        continue
                    non_ping_data_count += 1
                    last_event_type = event_type
                    if event_type in {"message_stop", "error"} or non_ping_data_count >= close_after_events:
                        deps.log(
                            f"{trace_prefix}claude_cache_keepalive_close "
                            f"elapsed={deps.fmt_ms(keepalive_t0)} "
                            f"lines={line_count} data_lines={data_line_count} "
                            f"non_ping_data={non_ping_data_count}/{close_after_events} "
                            f"event={event_type} action=close "
                            f"usage_read={final_usage.get('cache_read_input_tokens', 0)} "
                            f"usage_create={final_usage.get('cache_creation_input_tokens', 0)} "
                            f"usage_input={final_usage.get('input_tokens', 0)} "
                            f"usage_output={final_usage.get('output_tokens', 0)}"
                        )
                        return event_type != "error"

                deps.log(
                    f"{trace_prefix}claude_cache_keepalive_no_data "
                    f"elapsed={deps.fmt_ms(keepalive_t0)} "
                    f"lines={line_count} data_lines={data_line_count} "
                    f"non_ping_data={non_ping_data_count} last_event={last_event_type} "
                    f"usage_read={final_usage.get('cache_read_input_tokens', 0)} "
                    f"usage_create={final_usage.get('cache_creation_input_tokens', 0)} "
                    f"usage_input={final_usage.get('input_tokens', 0)} "
                    f"usage_output={final_usage.get('output_tokens', 0)}"
                )
                return non_ping_data_count > 0
    except asyncio.CancelledError:
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_cancelled "
            f"elapsed={deps.fmt_ms(keepalive_t0)}"
        )
        raise
    except Exception as e:
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_error "
            f"type={type(e).__name__} elapsed={deps.fmt_ms(keepalive_t0)} err={e}"
        )
        return False


async def _run_claude_cache_keepalive_loop(
    *,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    trace_id: str,
    deps: AnthropicUpstreamDeps,
    interval_sec: float,
    max_tokens: int,
    first_data_timeout_sec: float,
    close_after_data_events: int,
    max_runs: int,
    state: _ClaudeCacheKeepaliveLoopState | None = None,
) -> None:
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    interval = max(30.0, float(interval_sec or 240.0))
    run_index = 0
    run_limit = max(1, min(5, int(max_runs or 5)))
    deps.log(
        f"{trace_prefix}claude_cache_keepalive_scheduled "
        f"interval={interval:.1f}s max_tokens={max_tokens} "
        f"first_data_timeout={first_data_timeout_sec:.1f}s "
        f"close_after_data_events={close_after_data_events} "
        f"max_runs={run_limit}"
    )
    try:
        while run_index < run_limit:
            await asyncio.sleep(interval)
            if state is not None and state.stop_requested:
                break
            run_index += 1
            if state is not None:
                state.in_flight = True
            try:
                ok = await _run_claude_cache_keepalive_once(
                    url=url,
                    request_data=request_data,
                    headers=headers,
                    model=model,
                    trace_id=trace_id,
                    deps=deps,
                    max_tokens=max_tokens,
                    first_data_timeout_sec=first_data_timeout_sec,
                    close_after_data_events=close_after_data_events,
                )
            finally:
                if state is not None:
                    state.in_flight = False
            deps.log(
                f"{trace_prefix}claude_cache_keepalive_result "
                f"run={run_index} ok={str(ok).lower()}"
            )
            if state is not None and state.stop_requested:
                break
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_completed "
            f"runs={run_index} reason={'stop_requested' if state is not None and state.stop_requested else 'max_runs'}"
        )
    except asyncio.CancelledError:
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_stopped "
            f"runs={run_index}"
        )
        raise


async def _run_claude_cache_post_keepalive_after_delay(
    *,
    post_key: str,
    delay_sec: float,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    trace_id: str,
    deps: AnthropicUpstreamDeps,
    max_tokens: int,
    first_data_timeout_sec: float,
    close_after_data_events: int,
    max_runs: int,
    post_state: _ClaudeCachePostKeepaliveState,
) -> None:
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    delay = max(30.0, float(delay_sec or 240.0))
    run_limit = max(1, min(5, int(max_runs or 1))) if "sonnet" in str(model or "").lower() else 1
    fingerprint = _build_claude_cache_fingerprint(model, request_data)
    deps.log(
        f"{trace_prefix}claude_cache_keepalive_post_scheduled "
        f"key={post_key} fingerprint={fingerprint} delay={delay:.1f}s "
        f"max_runs={run_limit}"
    )
    try:
        await asyncio.sleep(delay)
        post_state.started = True
        run_index = 0
        while run_index < run_limit:
            if _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.get(post_key) is not post_state:
                deps.log(
                    f"{trace_prefix}claude_cache_keepalive_post_skip "
                    f"key={post_key} reason=replaced runs={run_index}"
                )
                return
            if post_state.stop_requested:
                break
            run_index += 1
            post_state.in_flight = True
            try:
                ok = await _run_claude_cache_keepalive_once(
                    url=url,
                    request_data=request_data,
                    headers=headers,
                    model=model,
                    trace_id=trace_id,
                    deps=deps,
                    max_tokens=max_tokens,
                    first_data_timeout_sec=first_data_timeout_sec,
                    close_after_data_events=close_after_data_events,
                )
            finally:
                post_state.in_flight = False
            deps.log(
                f"{trace_prefix}claude_cache_keepalive_post_result "
                f"key={post_key} fingerprint={fingerprint} run={run_index}/{run_limit} "
                f"ok={str(ok).lower()}"
            )
            if post_state.stop_requested or run_index >= run_limit:
                break
            await asyncio.sleep(delay)
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_post_completed "
            f"key={post_key} fingerprint={fingerprint} runs={run_index} "
            f"reason={'stop_requested' if post_state.stop_requested else 'max_runs'}"
        )
    except asyncio.CancelledError:
        deps.log(
            f"{trace_prefix}claude_cache_keepalive_post_cancelled "
            f"key={post_key}"
        )
        raise
    finally:
        if _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.get(post_key) is post_state:
            _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.pop(post_key, None)


def _build_code_cli_prime_url(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except Exception:
        return None

    host = (parts.hostname or "").lower()
    # Optional connection priming for selected hosts only.  Disabled by default
    # for public/open-source use; enable by setting CLAUDE_PRIME_HOSTS.
    hosts = {h.strip().lower() for h in os.environ.get("CLAUDE_PRIME_HOSTS", "").split(",") if h.strip()}
    if parts.scheme != "https" or not hosts or host not in hosts:
        return None

    return f"{parts.scheme}://{parts.netloc}/"


async def _prime_code_cli_connection(
    *,
    client: httpx.AsyncClient,
    url: str,
    deps: AnthropicUpstreamDeps,
    trace_prefix: str = "",
) -> None:
    prime_url = _build_code_cli_prime_url(url)
    if not prime_url:
        return

    try:
        prime_t0 = time.perf_counter()
        response = await client.head(
            prime_url,
            headers={
                "Connection": "keep-alive",
                "User-Agent": "Bun/1.3.13",
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br, zstd",
            },
        )
        deps.log(
            f"{trace_prefix}claude_prime_head status={response.status_code} "
            f"elapsed={deps.fmt_ms(prime_t0)}"
        )
    except Exception as e:
        deps.log(f"{trace_prefix}claude_prime_head_error {type(e).__name__}: {e}")


def _map_stop_reason(stop_reason: str | None) -> str:
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    return "stop"


def _build_openai_chunk(
    *,
    stream_id: str,
    created: int,
    model: str,
    delta: str = "",
    role: str | None = None,
    delta_field: str = "content",
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta_payload: dict[str, Any] = {}
    if role:
        delta_payload["role"] = role
    if delta:
        delta_payload[delta_field] = delta

    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta_payload,
            "finish_reason": finish_reason,
        }],
    }


def _build_openai_keepalive_chunk(*, stream_id: str, created: int, model: str) -> bytes:
    # OpenAI-compatible no-op chunk. SillyTavern accepts choices[].delta objects;
    # an empty delta produces no visible text, but it refreshes the downstream
    # stream while Claude Code Opus is emitting only thinking/signature/ping.
    chunk = _build_openai_chunk(stream_id=stream_id, created=created, model=model)
    return f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n".encode()


def _truncate_for_stop_tag(full_response: str, delta: str, find_stop_tag: Callable[[str], int]) -> tuple[str, str]:
    prefix_pos = find_stop_tag(full_response)
    if prefix_pos < 0:
        return full_response, delta

    prev_len = len(full_response) - len(delta)
    if prefix_pos >= prev_len:
        keep_len = prefix_pos - prev_len
        clean_delta = delta[:keep_len]
    else:
        clean_delta = ""

    truncated = full_response[:prefix_pos]
    return truncated, clean_delta


def _extract_saved_raw_sse_lines(raw_sse_text: str) -> list[str]:
    lines = raw_sse_text.splitlines()
    if (
        len(lines) >= 3
        and lines[0].startswith("Time: ")
        and lines[1].startswith("Model: ")
        and lines[2]
        and set(lines[2]) == {"="}
    ):
        return lines[3:]
    return lines


async def _iterate_saved_raw_sse_lines(raw_sse_text: str) -> AsyncGenerator[str, None]:
    for line in _extract_saved_raw_sse_lines(raw_sse_text):
        yield line
        await asyncio.sleep(0)


async def forward_anthropic_stream(
    *,
    url: str,
    request_data: dict,
    headers: dict,
    timeout: httpx.Timeout,
    deps: AnthropicUpstreamDeps,
    enable_early_stop: bool = True,
    trace_id: str = "",
) -> AsyncGenerator[bytes, None]:
    """Anthropic SSE 原样透传，保留现有 /v1/messages 路由能力。"""
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    content_buffer = ""
    first_data_time: float | None = None
    line_count = 0
    data_line_count = 0
    finish_status = "unknown"

    try:
        async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
            await _prime_code_cli_connection(client=client, url=url, deps=deps, trace_prefix=trace_prefix)
            deps.log(f"{trace_prefix}anthropic_stream_request_start elapsed={deps.fmt_ms(request_t0)}")
            upstream_t0 = time.perf_counter()
            async with client.stream("POST", url, json=request_data, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}anthropic_stream_headers status={response.status_code} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    content = await response.aread()
                    error_text = content.decode(errors="replace")[:4000]
                    payload = {
                        "type": "error",
                        "error": {
                            "type": "upstream_http_error",
                            "status": response.status_code,
                            "message": error_text,
                        },
                    }
                    yield b"event: error\n"
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                    yield b"event: message_stop\n"
                    yield b'data: {"type":"message_stop"}\n\n'
                    return

                async for line in response.aiter_lines():
                    line_count += 1
                    if not line:
                        yield b"\n"
                        continue

                    if line.startswith("data: "):
                        data_line_count += 1
                        if first_data_time is None:
                            first_data_time = time.perf_counter()
                            deps.log(
                                f"{trace_prefix}anthropic_stream_first_data "
                                f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                            )
                        payload = line[6:].strip()
                        if payload:
                            try:
                                data = json.loads(payload)
                            except json.JSONDecodeError:
                                data = None

                            if isinstance(data, dict):
                                content = _extract_anthropic_sse_text(data)
                                if content:
                                    content_buffer += content
                                    if enable_early_stop and deps.has_stop_tag(content_buffer):
                                        deps.log("Anthropic stream: detected STOP_TAG")
                                        finish_status = "early_stop_tag"
                                        await _close_upstream_stream(
                                            response=response,
                                            client=client,
                                            deps=deps,
                                            trace_prefix=trace_prefix,
                                            label="early_stop",
                                            reason="anthropic_raw_stop_tag",
                                            started_at=request_t0,
                                            line_count=line_count,
                                            data_line_count=data_line_count,
                                            out_chars=len(content_buffer),
                                        )
                                        yield b"event: message_stop\n"
                                        yield b'data: {"type":"message_stop"}\n\n'
                                        return

                                if data.get("type") == "message_stop":
                                    finish_status = "message_stop"
                                    yield f"{line}\n".encode()
                                    return

                    yield f"{line}\n".encode()

                finish_status = "stream_end_without_message_stop"

    except Exception as e:
        finish_status = f"exception:{type(e).__name__}"
        deps.log(f"Anthropic stream forward error: {e}")
        payload = {
            "type": "error",
            "error": {
                "type": "proxy_stream_error",
                "message": str(e),
            },
        }
        yield b"event: error\n"
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
        yield b"event: message_stop\n"
        yield b'data: {"type":"message_stop"}\n\n'
    finally:
        deps.log(
            f"{trace_prefix}anthropic_stream_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} out_chars={len(content_buffer)}"
        )


async def forward_anthropic_chat_stream(
    *,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    messages: list,
    trace_id: str,
    caller_key: str,
    caller_desc: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: AnthropicUpstreamDeps,
    cache_keepalive: dict[str, Any] | None = None,
) -> AsyncGenerator[bytes, None]:
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    full_response = ""
    first_data_time: float | None = None
    line_count = 0
    data_line_count = 0
    finish_status = "unknown"
    raw_sse_lines: list[str] = []
    raw_sse_size = 0
    raw_sse_truncated = False
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    sent_role_chunk = False
    keepalive_interval = 10.0
    progress_interval = 30.0
    last_downstream_emit = time.perf_counter()
    keepalive_count = 0
    last_progress_time = time.perf_counter()
    last_upstream_line_time = last_progress_time
    last_visible_content_time: float | None = None
    event_counts: dict[str, int] = {}
    ping_count = 0
    reasoning_chars = 0
    ignored_delta_chars = 0
    visible_delta_count = 0
    last_event_type = "-"
    upstream_error_event_msg: str | None = None
    upstream_error_event_count = 0
    cache_keepalive_task: asyncio.Task | None = None
    cache_keepalive_state: _ClaudeCacheKeepaliveLoopState | None = None
    cache_post_keepalive_key = _build_claude_cache_post_keepalive_key(model)
    cache_keepalive_enabled = (
        isinstance(cache_keepalive, dict)
        and bool(cache_keepalive.get("enabled"))
        and _has_cache_control(request_data)
    )
    upstream_closed_by_proxy = False

    def emit_done() -> bytes:
        return b"data: [DONE]\n\n"

    def log_progress(reason: str, *, force: bool = False) -> None:
        nonlocal last_progress_time
        now = time.perf_counter()
        if not force and now - last_progress_time < progress_interval:
            return
        last_progress_time = now
        event_summary = ",".join(
            f"{key}:{value}"
            for key, value in sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
        ) or "-"
        deps.log(
            f"{trace_prefix}claude_upstream_progress "
            f"reason={reason} elapsed={deps.fmt_ms(request_t0, now)} "
            f"since_upstream_line={deps.fmt_ms(last_upstream_line_time, now)} "
            f"since_visible={deps.fmt_ms(last_visible_content_time, now) if last_visible_content_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} "
            f"out_chars={len(full_response)} visible_deltas={visible_delta_count} "
            f"reasoning_chars={reasoning_chars} ignored_delta_chars={ignored_delta_chars} "
            f"pings={ping_count} keepalives={keepalive_count} "
            f"last_event={last_event_type} events={event_summary}"
        )

    def emit_keepalive_if_idle() -> bytes | None:
        nonlocal last_downstream_emit, keepalive_count
        now = time.perf_counter()
        if now - last_downstream_emit < keepalive_interval:
            return None
        previous_emit = last_downstream_emit
        last_downstream_emit = now
        keepalive_count += 1
        deps.log(
            f"{trace_prefix}claude_downstream_keepalive "
            f"count={keepalive_count} idle={deps.fmt_ms(previous_emit, now)} "
            f"out_chars={len(full_response)}"
        )
        return _build_openai_keepalive_chunk(
            stream_id=stream_id,
            created=created,
            model=model,
        )

    async def close_upstream_for_early_stop(
        *,
        response: httpx.Response,
        client: httpx.AsyncClient,
        reader_task: asyncio.Task | None,
        reason: str,
    ) -> None:
        nonlocal upstream_closed_by_proxy
        close_t0 = time.perf_counter()
        upstream_closed_by_proxy = True
        if reader_task is not None and not reader_task.done():
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        response_closed = False
        client_closed = False
        try:
            await response.aclose()
            response_closed = True
        except Exception as e:
            deps.log(
                f"{trace_prefix}early_stop_upstream_close_response_error "
                f"reason={reason} type={type(e).__name__} err={e}"
            )
        try:
            await client.aclose()
            client_closed = True
        except Exception as e:
            deps.log(
                f"{trace_prefix}early_stop_upstream_close_client_error "
                f"reason={reason} type={type(e).__name__} err={e}"
            )
        deps.log(
            f"{trace_prefix}early_stop_upstream_close "
            f"reason={reason} response_closed={str(response_closed).lower()} "
            f"client_closed={str(client_closed).lower()} elapsed={deps.fmt_ms(close_t0)} "
            f"lines={line_count} data_lines={data_line_count} out_chars={len(full_response)}"
        )

    try:
        if caller_key:
            deps.log(f"{trace_prefix}stream_owner caller={caller_key} {caller_desc}")
        if cache_keepalive_enabled:
            _cancel_claude_cache_post_keepalive(
                post_key=cache_post_keepalive_key,
                deps=deps,
                trace_prefix=trace_prefix,
                reason="new_request",
            )

        async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
            await _prime_code_cli_connection(
                client=client,
                url=url,
                deps=deps,
                trace_prefix=trace_prefix,
            )
            deps.log(f"{trace_prefix}claude_upstream_request_start elapsed={deps.fmt_ms(request_t0)}")
            upstream_t0 = time.perf_counter()
            async with client.stream("POST", url, json=request_data, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}claude_upstream_headers status={response.status_code} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    content = await response.aread()
                    error_text = content.decode(errors="replace")[:4000]
                    deps.save_request_log(
                        model,
                        messages,
                        f"[UPSTREAM_HTTP_ERROR] {error_text}",
                        stream=True,
                        raw_sse=f"[HTTP {response.status_code}]\n{error_text}",
                        request_payload=request_data,
                        error_type="claude_upstream_http_error",
                        trace_id=trace_id,
                    )
                    finish_status = f"upstream_http_{response.status_code}"
                    yield deps.build_openai_sse_error(
                        response.status_code,
                        error_text,
                        "claude_upstream_http_error",
                    )
                    yield emit_done()
                    return

                line_queue = asyncio.Queue()
                if cache_keepalive_enabled:
                    cache_keepalive_state = _ClaudeCacheKeepaliveLoopState()
                    cache_keepalive_task = asyncio.create_task(
                        _run_claude_cache_keepalive_loop(
                            url=url,
                            request_data=request_data,
                            headers=headers,
                            model=model,
                            trace_id=trace_id,
                            deps=deps,
                            interval_sec=float(cache_keepalive.get("interval_sec") or 240.0),
                            max_tokens=int(cache_keepalive.get("max_tokens") or 1),
                            first_data_timeout_sec=float(cache_keepalive.get("first_data_timeout_sec") or 60.0),
                            close_after_data_events=int(cache_keepalive.get("close_after_data_events") or 1),
                            max_runs=int(cache_keepalive.get("max_runs") or 5),
                            state=cache_keepalive_state,
                        )
                    )
                elif isinstance(cache_keepalive, dict) and cache_keepalive.get("enabled"):
                    deps.log(
                        f"{trace_prefix}claude_cache_keepalive_skip "
                        f"reason=no_cache_control"
                    )

                async def read_upstream_lines() -> None:
                    reader_t0 = time.perf_counter()
                    reader_last_progress = reader_t0
                    reader_lines = 0
                    reader_data_lines = 0
                    reader_events: dict[str, int] = {}
                    reader_text_chars = 0
                    reader_thinking_chars = 0
                    reader_last_event = "-"

                    def reader_log_progress(reason: str, *, force: bool = False) -> None:
                        nonlocal reader_last_progress
                        now = time.perf_counter()
                        if not force and now - reader_last_progress < progress_interval:
                            return
                        reader_last_progress = now
                        event_summary = ",".join(
                            f"{key}:{value}"
                            for key, value in sorted(reader_events.items(), key=lambda item: (-item[1], item[0]))[:6]
                        ) or "-"
                        deps.log(
                            f"{trace_prefix}claude_upstream_reader_progress "
                            f"reason={reason} elapsed={deps.fmt_ms(request_t0, now)} "
                            f"reader_elapsed={deps.fmt_ms(reader_t0, now)} "
                            f"reader_lines={reader_lines} reader_data_lines={reader_data_lines} "
                            f"reader_text_chars={reader_text_chars} reader_thinking_chars={reader_thinking_chars} "
                            f"queue_size={line_queue.qsize()} last_event={reader_last_event} "
                            f"events={event_summary}"
                        )

                    try:
                        async for upstream_line in response.aiter_lines():
                            reader_lines += 1
                            if upstream_line.startswith("event:"):
                                event_name = upstream_line.split(":", 1)[1].strip() or "-"
                                reader_last_event = event_name
                                reader_events[event_name] = reader_events.get(event_name, 0) + 1
                                if event_name == "ping":
                                    reader_log_progress("reader_ping")
                            elif upstream_line.startswith("data: "):
                                reader_data_lines += 1
                                payload = upstream_line[6:].strip()
                                if payload:
                                    try:
                                        reader_data = json.loads(payload)
                                    except json.JSONDecodeError:
                                        reader_data = None
                                    if isinstance(reader_data, dict):
                                        reader_type = str(reader_data.get("type") or reader_last_event or "-")
                                        reader_last_event = reader_type
                                        reader_events[reader_type] = reader_events.get(reader_type, 0) + 1
                                        delta = reader_data.get("delta")
                                        if isinstance(delta, dict):
                                            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                                                reader_text_chars += len(delta["text"])
                                            elif delta.get("type") == "thinking_delta" and isinstance(delta.get("thinking"), str):
                                                reader_thinking_chars += len(delta["thinking"])
                            reader_log_progress("reader_line")
                            await line_queue.put(upstream_line)
                    except Exception as upstream_error:
                        reader_log_progress(f"reader_exception:{type(upstream_error).__name__}", force=True)
                        await line_queue.put(upstream_error)
                    finally:
                        reader_log_progress("reader_done", force=True)
                        await line_queue.put(None)

                upstream_reader = asyncio.create_task(read_upstream_lines())
                try:
                    while True:
                        try:
                            queued_line = await asyncio.wait_for(
                                line_queue.get(),
                                timeout=keepalive_interval,
                            )
                        except asyncio.TimeoutError:
                            keepalive = emit_keepalive_if_idle()
                            if keepalive is not None:
                                yield keepalive
                            log_progress("idle_no_upstream_line")
                            continue

                        if queued_line is None:
                            break
                        if isinstance(queued_line, Exception):
                            raise queued_line

                        last_upstream_line_time = time.perf_counter()
                        keepalive = emit_keepalive_if_idle()
                        if keepalive is not None:
                            yield keepalive

                        line = queued_line
                        line_count += 1
                        raw_sse_size, raw_sse_truncated = _append_raw_line(
                            raw_sse_lines,
                            line,
                            raw_size=raw_sse_size,
                            raw_truncated=raw_sse_truncated,
                            max_raw_sse_bytes=max_raw_sse_bytes,
                        )

                        if not line or line.startswith("event:"):
                            continue
                        if not line.startswith("data: "):
                            continue

                        data_line_count += 1
                        if first_data_time is None:
                            first_data_time = time.perf_counter()
                            deps.log(
                                f"{trace_prefix}claude_upstream_first_data "
                                f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                            )

                        payload = line[6:].strip()
                        if not payload:
                            continue

                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        if data.get("type") == "error":
                            err = data.get("error") or {}
                            err_msg = err.get("message") or json.dumps(data, ensure_ascii=False)
                            upstream_error_event_count += 1
                            upstream_error_event_msg = str(err_msg)
                            finish_status = "upstream_error_event_seen_continue"
                            deps.log(
                                f"{trace_prefix}claude_upstream_error_event_ignored_continue "
                                f"count={upstream_error_event_count} out_chars={len(full_response)} "
                                f"err={str(err_msg)[:500]}"
                            )
                            # Do not actively close the upstream stream on event:error.  Some
                            # providers send terminal error events over an otherwise valid SSE
                            # channel; keep reading until the upstream naturally ends so we don't
                            # become the side that aborts the connection.
                            continue

                        event_type = data.get("type", "")
                        if event_type:
                            last_event_type = str(event_type)
                            event_counts[last_event_type] = event_counts.get(last_event_type, 0) + 1
                        if event_type == "ping":
                            ping_count += 1
                            log_progress("upstream_ping")
                            continue
                        log_progress("upstream_data")

                        if event_type == "message_start":
                            message = data.get("message") or {}
                            if message.get("id"):
                                stream_id = f"chatcmpl-{message['id']}"
                            if message.get("model"):
                                model = str(message["model"])
                            usage = _normalize_usage(message.get("usage"), usage)
                            if not sent_role_chunk:
                                role_chunk = _build_openai_chunk(
                                    stream_id=stream_id,
                                    created=created,
                                    model=model,
                                    role="assistant",
                                )
                                yield f"data: {json.dumps(role_chunk, ensure_ascii=True)}\n\n".encode()
                                last_downstream_emit = time.perf_counter()
                                sent_role_chunk = True
                            continue

                        if event_type == "content_block_delta":
                            delta, delta_field = _extract_anthropic_sse_delta(data)
                            if not delta:
                                continue
                            if delta_field != "content":
                                # SillyTavern's OpenAI stream consumer does not display
                                # reasoning chunks; sending only reasoning can make it
                                # abort the stream before visible text arrives.
                                ignored_delta_chars += len(delta)
                                if delta_field == "reasoning":
                                    reasoning_chars += len(delta)
                                log_progress(f"ignored_{delta_field or 'delta'}")
                                continue

                            if not sent_role_chunk:
                                role_chunk = _build_openai_chunk(
                                    stream_id=stream_id,
                                    created=created,
                                    model=model,
                                    role="assistant",
                                )
                                yield f"data: {json.dumps(role_chunk, ensure_ascii=True)}\n\n".encode()
                                last_downstream_emit = time.perf_counter()
                                sent_role_chunk = True

                            full_response += delta
                            visible_delta_count += 1
                            last_visible_content_time = time.perf_counter()
                            if deps.has_stop_tag(full_response):
                                full_response, clean_delta = _truncate_for_stop_tag(
                                    full_response,
                                    delta,
                                    deps.find_stop_tag,
                                )
                                if clean_delta:
                                    chunk = _build_openai_chunk(
                                        stream_id=stream_id,
                                        created=created,
                                        model=model,
                                        delta=clean_delta,
                                        delta_field=delta_field,
                                    )
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                                    last_downstream_emit = time.perf_counter()

                                stop_chunk = _build_openai_chunk(
                                    stream_id=stream_id,
                                    created=created,
                                    model=model,
                                    finish_reason="stop",
                                )
                                deps.save_request_log(
                                    model,
                                    messages,
                                    full_response or "[empty response]",
                                    stream=True,
                                    raw_sse="\n".join(raw_sse_lines),
                                    request_payload=request_data,
                                    trace_id=trace_id,
                                )
                                finish_status = "early_stop_tag"
                                await close_upstream_for_early_stop(
                                    response=response,
                                    client=client,
                                    reader_task=upstream_reader,
                                    reason="stop_tag",
                                )
                                yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                                last_downstream_emit = time.perf_counter()
                                yield emit_done()
                                return

                            chunk = _build_openai_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                                delta=delta,
                                delta_field=delta_field,
                            )
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                            last_downstream_emit = time.perf_counter()
                            continue

                        if event_type == "message_delta":
                            delta = data.get("delta") or {}
                            if isinstance(delta, dict):
                                stop_reason = delta.get("stop_reason") or stop_reason
                            usage = _normalize_usage(data.get("usage"), usage)
                            continue

                        if event_type == "message_stop":
                            stop_chunk = _build_openai_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                                finish_reason=_map_stop_reason(stop_reason),
                            )
                            deps.save_request_log(
                                model,
                                messages,
                                full_response or "[empty response]",
                                stream=True,
                                raw_sse="\n".join(raw_sse_lines),
                                request_payload=request_data,
                                trace_id=trace_id,
                            )
                            finish_status = f"message_stop:{stop_reason or 'stop'}"
                            yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                            last_downstream_emit = time.perf_counter()
                            yield emit_done()
                            return

                finally:
                    if not upstream_reader.done():
                        upstream_reader.cancel()
                        try:
                            await upstream_reader
                        except asyncio.CancelledError:
                            pass

                if upstream_error_event_count:
                    if full_response:
                        finish_status = "upstream_error_then_natural_end_content_kept"
                        deps.save_request_log(
                            model,
                            messages,
                            full_response,
                            stream=True,
                            raw_sse="\n".join(raw_sse_lines),
                            request_payload=request_data,
                            error_type="claude_upstream_error_event_ignored",
                            trace_id=trace_id,
                        )
                        yield emit_done()
                    else:
                        finish_status = "upstream_error_then_natural_end_no_content"
                        deps.save_request_log(
                            model,
                            messages,
                            f"[UPSTREAM_ERROR_EVENT] {upstream_error_event_msg or ''}",
                            stream=True,
                            raw_sse="\n".join(raw_sse_lines),
                            request_payload=request_data,
                            error_type="claude_upstream_error_event",
                            trace_id=trace_id,
                        )
                        yield deps.build_openai_sse_error(
                            502,
                            (upstream_error_event_msg or "Upstream provider error")[:4000],
                            "claude_upstream_error",
                        )
                        yield emit_done()
                else:
                    finish_status = "anthropic_incomplete_stream_end"
                    deps.save_request_log(
                        model,
                        messages,
                        full_response or "[empty response]",
                        stream=True,
                        raw_sse="\n".join(raw_sse_lines + ["[UPSTREAM_INCOMPLETE]"]),
                        request_payload=request_data,
                        error_type="claude_upstream_incomplete",
                        trace_id=trace_id,
                    )
                    yield deps.build_openai_sse_error(
                        502,
                        "Anthropic stream closed without message_stop",
                        "upstream_stream_incomplete",
                    )
                    yield emit_done()
    except asyncio.CancelledError:
        finish_status = "downstream_cancelled"
        deps.save_request_log(
            model,
            messages,
            full_response or "[cancelled before content]",
            stream=True,
            raw_sse="\n".join(raw_sse_lines + ["[CANCELLED: downstream client disconnected]"]),
            request_payload=request_data,
            error_type="downstream_cancelled",
            trace_id=trace_id,
        )
        raise
    except Exception as e:
        finish_status = f"exception:{type(e).__name__}"
        deps.save_request_log(
            model,
            messages,
            full_response + f"\n[ERROR] {e}" if full_response else f"[ERROR] {e}",
            stream=True,
            raw_sse="\n".join(raw_sse_lines + [f"[EXCEPTION] {e}"]),
            request_payload=request_data,
            error_type="claude_upstream_proxy_error",
            trace_id=trace_id,
        )
        yield deps.build_openai_sse_error(502, str(e), "claude_upstream_proxy_error")
        yield emit_done()
    finally:
        if cache_keepalive_task and not cache_keepalive_task.done():
            if cache_keepalive_state is not None and cache_keepalive_state.in_flight:
                cache_keepalive_state.stop_requested = True
                deps.log(
                    f"{trace_prefix}claude_cache_keepalive_stop_after_inflight "
                    f"reason=main_request_finished"
                )
            else:
                cache_keepalive_task.cancel()
                try:
                    await cache_keepalive_task
                except asyncio.CancelledError:
                    pass
        if cache_keepalive_enabled and isinstance(cache_keepalive, dict) and finish_status not in {"early_stop_tag", "downstream_cancelled"}:
            post_state = _ClaudeCachePostKeepaliveState()
            post_task = asyncio.create_task(
                _run_claude_cache_post_keepalive_after_delay(
                    post_key=cache_post_keepalive_key,
                    delay_sec=float(cache_keepalive.get("post_delay_sec") or cache_keepalive.get("interval_sec") or 240.0),
                    url=url,
                    request_data=request_data,
                    headers=headers,
                    model=model,
                    trace_id=trace_id,
                    deps=deps,
                    max_tokens=int(cache_keepalive.get("max_tokens") or 1),
                    first_data_timeout_sec=float(cache_keepalive.get("first_data_timeout_sec") or 60.0),
                    close_after_data_events=int(cache_keepalive.get("close_after_data_events") or 1),
                    max_runs=int(cache_keepalive.get("max_runs") or 5),
                    post_state=post_state,
                )
            )
            post_state.task = post_task
            previous_post_state = _CLAUDE_CACHE_POST_KEEPALIVE_TASKS.get(cache_post_keepalive_key)
            previous_post_task = (
                previous_post_state.task
                if isinstance(previous_post_state, _ClaudeCachePostKeepaliveState)
                else None
            )
            if previous_post_task and not previous_post_task.done():
                if previous_post_state and previous_post_state.started:
                    deps.log(
                        f"{trace_prefix}claude_cache_keepalive_post_replace_skip_cancel "
                        f"key={cache_post_keepalive_key} reason=already_started"
                    )
                else:
                    previous_post_task.cancel()
            _CLAUDE_CACHE_POST_KEEPALIVE_TASKS[cache_post_keepalive_key] = post_state
        if caller_key:
            deps.release_caller(caller_key, trace_id)
        deps.log(
            f"{trace_prefix}claude_upstream_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} "
            f"out_chars={len(full_response)} visible_deltas={visible_delta_count} "
            f"reasoning_chars={reasoning_chars} ignored_delta_chars={ignored_delta_chars} "
            f"pings={ping_count} keepalives={keepalive_count} "
            f"usage_total={usage.get('total_tokens', 0)}"
        )


async def collect_anthropic_chat_completion(
    *,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    messages: list,
    trace_id: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: AnthropicUpstreamDeps,
) -> tuple[str, str, dict[str, Any], str, str]:
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    request_data = dict(request_data)
    request_data["stream"] = True
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
            async with client.stream("POST", url, json=request_data, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}claude_collect_headers status={response.status_code} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    error_text = (await response.aread()).decode(errors="replace")[:4000]
                    raise RuntimeError(f"Claude upstream HTTP {response.status_code}: {error_text}")

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

                    if event_type == "content_block_delta":
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
                            return full_content, model_name, usage, "stop", raw_response
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
                        break

        if not saw_message_stop:
            finish_status = "stream_end_without_message_stop"
            raise RuntimeError("Claude upstream stream closed without message_stop")

        raw_response = "\n".join(raw_sse_lines)
        return full_content, model_name, usage, _map_stop_reason(stop_reason), raw_response
    finally:
        deps.log(
            f"{trace_prefix}claude_collect_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} "
            f"out_chars={len(full_content)} usage_total={usage.get('total_tokens', 0)}"
        )


async def collect_anthropic_message_response(
    *,
    url: str,
    request_data: dict[str, Any],
    headers: dict[str, str],
    model: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: AnthropicUpstreamDeps,
    trace_id: str = "",
) -> tuple[dict[str, Any], str]:
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    request_data = dict(request_data)
    request_data["stream"] = True
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
            async with client.stream("POST", url, json=request_data, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}anthropic_collect_headers status={response.status_code} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    error_text = (await response.aread()).decode(errors="replace")[:4000]
                    raise RuntimeError(f"Anthropic upstream HTTP {response.status_code}: {error_text}")

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
        )


async def replay_anthropic_chat_stream(
    *,
    raw_sse_text: str,
    request_data: dict[str, Any],
    model: str,
    messages: list,
    trace_id: str,
    caller_key: str,
    caller_desc: str,
    deps: AnthropicUpstreamDeps,
) -> AsyncGenerator[bytes, None]:
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    full_response = ""
    first_data_time: float | None = None
    line_count = 0
    data_line_count = 0
    finish_status = "unknown"
    raw_sse_lines = _extract_saved_raw_sse_lines(raw_sse_text)
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    sent_role_chunk = False

    def emit_done() -> bytes:
        return b"data: [DONE]\n\n"

    try:
        if caller_key:
            deps.log(f"{trace_prefix}replay_stream_owner caller={caller_key} {caller_desc}")

        async for line in _iterate_saved_raw_sse_lines(raw_sse_text):
            line_count += 1

            if not line or line.startswith("event:"):
                continue
            if not line.startswith("data: "):
                continue

            data_line_count += 1
            if first_data_time is None:
                first_data_time = time.perf_counter()
                deps.log(
                    f"{trace_prefix}claude_replay_first_data "
                    f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                )

            payload = line[6:].strip()
            if not payload:
                continue

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "error":
                err = data.get("error") or {}
                err_msg = err.get("message") or json.dumps(data, ensure_ascii=False)
                deps.save_request_log(
                    model,
                    messages,
                    f"[REPLAY_ERROR_EVENT] {err_msg}",
                    stream=True,
                    raw_sse="\n".join(raw_sse_lines),
                    request_payload=request_data,
                    error_type="claude_replay_error_event",
                    trace_id=trace_id,
                )
                finish_status = "replay_error_event"
                yield deps.build_openai_sse_error(
                    502,
                    err_msg[:4000],
                    err.get("type", "claude_replay_error"),
                )
                yield emit_done()
                return

            event_type = data.get("type", "")

            if event_type == "message_start":
                message = data.get("message") or {}
                if message.get("id"):
                    stream_id = f"chatcmpl-{message['id']}"
                if message.get("model"):
                    model = str(message["model"])
                usage = _normalize_usage(message.get("usage"), usage)
                if not sent_role_chunk:
                    role_chunk = _build_openai_chunk(
                        stream_id=stream_id,
                        created=created,
                        model=model,
                        role="assistant",
                    )
                    yield f"data: {json.dumps(role_chunk, ensure_ascii=True)}\n\n".encode()
                    sent_role_chunk = True
                continue

            if event_type == "content_block_delta":
                delta = _extract_anthropic_sse_text(data)
                if not delta:
                    continue

                if not sent_role_chunk:
                    role_chunk = _build_openai_chunk(
                        stream_id=stream_id,
                        created=created,
                        model=model,
                        role="assistant",
                    )
                    yield f"data: {json.dumps(role_chunk, ensure_ascii=True)}\n\n".encode()
                    sent_role_chunk = True

                full_response += delta
                if deps.has_stop_tag(full_response):
                    full_response, clean_delta = _truncate_for_stop_tag(
                        full_response,
                        delta,
                        deps.find_stop_tag,
                    )
                    if clean_delta:
                        chunk = _build_openai_chunk(
                            stream_id=stream_id,
                            created=created,
                            model=model,
                            delta=clean_delta,
                        )
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

                    stop_chunk = _build_openai_chunk(
                        stream_id=stream_id,
                        created=created,
                        model=model,
                        finish_reason="stop",
                    )
                    deps.save_request_log(
                        model,
                        messages,
                        full_response or "[empty replay response]",
                        stream=True,
                        raw_sse="\n".join(raw_sse_lines),
                        request_payload=request_data,
                        trace_id=trace_id,
                    )
                    finish_status = "replay_early_stop_tag"
                    yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                    yield emit_done()
                    return

                chunk = _build_openai_chunk(
                    stream_id=stream_id,
                    created=created,
                    model=model,
                    delta=delta,
                )
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                continue

            if event_type == "message_delta":
                delta = data.get("delta") or {}
                if isinstance(delta, dict):
                    stop_reason = delta.get("stop_reason") or stop_reason
                usage = _normalize_usage(data.get("usage"), usage)
                continue

            if event_type == "message_stop":
                stop_chunk = _build_openai_chunk(
                    stream_id=stream_id,
                    created=created,
                    model=model,
                    finish_reason=_map_stop_reason(stop_reason),
                )
                deps.save_request_log(
                    model,
                    messages,
                    full_response or "[empty replay response]",
                    stream=True,
                    raw_sse="\n".join(raw_sse_lines),
                    request_payload=request_data,
                    trace_id=trace_id,
                )
                finish_status = f"replay_message_stop:{stop_reason or 'stop'}"
                yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                yield emit_done()
                return

        finish_status = "replay_incomplete_stream_end"
        deps.save_request_log(
            model,
            messages,
            full_response or "[empty replay response]",
            stream=True,
            raw_sse="\n".join(raw_sse_lines + ["[REPLAY_INCOMPLETE]"]),
            request_payload=request_data,
            error_type="claude_replay_incomplete",
            trace_id=trace_id,
        )
        yield deps.build_openai_sse_error(
            502,
            "Replay stream closed without message_stop",
            "replay_stream_incomplete",
        )
        yield emit_done()
    except asyncio.CancelledError:
        finish_status = "replay_downstream_cancelled"
        deps.save_request_log(
            model,
            messages,
            full_response or "[cancelled before replay content]",
            stream=True,
            raw_sse="\n".join(raw_sse_lines + ["[CANCELLED: downstream client disconnected during replay]"]),
            request_payload=request_data,
            error_type="downstream_cancelled",
            trace_id=trace_id,
        )
        raise
    except Exception as e:
        finish_status = f"replay_exception:{type(e).__name__}"
        deps.save_request_log(
            model,
            messages,
            full_response + f"\n[ERROR] {e}" if full_response else f"[ERROR] {e}",
            stream=True,
            raw_sse="\n".join(raw_sse_lines + [f"[EXCEPTION] {e}"]),
            request_payload=request_data,
            error_type="claude_replay_proxy_error",
            trace_id=trace_id,
        )
        yield deps.build_openai_sse_error(502, str(e), "claude_replay_proxy_error")
        yield emit_done()
    finally:
        if caller_key:
            deps.release_caller(caller_key, trace_id)
        deps.log(
            f"{trace_prefix}claude_replay_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} "
            f"out_chars={len(full_response)} usage_total={usage.get('total_tokens', 0)}"
        )


async def collect_anthropic_chat_completion_from_raw_sse(
    *,
    raw_sse_text: str,
    request_data: dict[str, Any],
    model: str,
    messages: list,
    trace_id: str,
    deps: AnthropicUpstreamDeps,
) -> tuple[str, str, dict[str, Any], str, str]:
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    raw_sse_lines = _extract_saved_raw_sse_lines(raw_sse_text)
    full_content = ""
    model_name = model
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    saw_message_stop = False

    deps.log(f"{trace_prefix}claude_replay_collect_start lines={len(raw_sse_lines)}")

    async for line in _iterate_saved_raw_sse_lines(raw_sse_text):
        if not line or line.startswith("event:") or not line.startswith("data: "):
            continue

        payload = line[6:].strip()
        if not payload:
            continue

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "error":
            err = data.get("error") or {}
            err_msg = err.get("message") or json.dumps(data, ensure_ascii=False)
            raise RuntimeError(err_msg)

        event_type = data.get("type", "")

        if event_type == "message_start":
            message = data.get("message") or {}
            if message.get("model"):
                model_name = str(message["model"])
            usage = _normalize_usage(message.get("usage"), usage)
            continue

        if event_type == "content_block_delta":
            delta = _extract_anthropic_sse_text(data)
            if not delta:
                continue

            full_content += delta
            if deps.has_stop_tag(full_content):
                full_content = full_content[:deps.find_stop_tag(full_content)]
                return full_content, model_name, usage, "stop", "\n".join(raw_sse_lines)
            continue

        if event_type == "message_delta":
            delta = data.get("delta") or {}
            if isinstance(delta, dict):
                stop_reason = delta.get("stop_reason") or stop_reason
            usage = _normalize_usage(data.get("usage"), usage)
            continue

        if event_type == "message_stop":
            saw_message_stop = True
            break

    if not saw_message_stop:
        raise RuntimeError("Claude replay stream closed without message_stop")

    return full_content, model_name, usage, _map_stop_reason(stop_reason), "\n".join(raw_sse_lines)
