import asyncio
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .transport import _prime_code_cli_connection
from .types import AnthropicMessagesDeps


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
    deps: AnthropicMessagesDeps,
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
    deps: AnthropicMessagesDeps,
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
    deps: AnthropicMessagesDeps,
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
    deps: AnthropicMessagesDeps,
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
