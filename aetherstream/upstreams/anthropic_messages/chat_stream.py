import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator

import httpx

from aetherstream.observability.refusals import AnthropicRefusalDiagnostics

from aetherstream.runtime.tasks import spawn_detached

from aetherstream.features.terminal_tool import TERMINAL_TOOL_NAME
from aetherstream.upstreams.route_logging import format_account_pool_route, scope_stop_detection

from .cache import (
    _CLAUDE_CACHE_POST_KEEPALIVE_TASKS,
    _ClaudeCacheKeepaliveLoopState,
    _ClaudeCachePostKeepaliveState,
    _build_claude_cache_post_keepalive_key,
    _cancel_claude_cache_post_keepalive,
    _has_cache_control,
    _run_claude_cache_keepalive_loop,
    _run_claude_cache_post_keepalive_after_delay,
)
from .protocol import (
    _append_raw_line,
    _build_openai_chunk,
    _build_openai_keepalive_chunk,
    _extract_anthropic_sse_delta,
    _map_stop_reason,
    _normalize_usage,
    _truncate_for_stop_tag,
)
from .transport import _is_async_generator_close_race, _prime_code_cli_connection, _send_with_oauth_recovery
from .types import AnthropicMessagesDeps


async def forward_anthropic_messages_as_chat_stream(
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
    deps: AnthropicMessagesDeps,
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
    refusal_diagnostics = AnthropicRefusalDiagnostics()
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
    header_keepalive_enabled = bool(deps.header_keepalive_enabled)
    header_keepalive_interval_sec = max(0.1, float(deps.header_keepalive_interval_sec or 3.0))
    stream_idle_timeout_enabled = bool(deps.stream_idle_timeout_enabled)
    stream_idle_timeout_sec = max(0.1, float(deps.stream_idle_timeout_sec or 4.0))
    upstream_data_started = False
    last_upstream_data_time: float | None = None
    ordinary_tool_seen = False
    suppressed_terminal_blocks: set[int] = set()

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
            upstream_open_task = asyncio.create_task(_send_with_oauth_recovery(
                client=client, url=url, request_data=request_data, headers=headers,
                deps=deps, trace_prefix=trace_prefix))
            response: httpx.Response | None = None
            try:
                if header_keepalive_enabled:
                    while response is None:
                        try:
                            response = await asyncio.wait_for(
                                asyncio.shield(upstream_open_task),
                                timeout=header_keepalive_interval_sec,
                            )
                        except asyncio.TimeoutError:
                            keepalive_count += 1
                            last_downstream_emit = time.perf_counter()
                            deps.log(
                                f"{trace_prefix}claude_downstream_header_keepalive "
                                f"count={keepalive_count} interval={header_keepalive_interval_sec:.3f}s "
                                f"header_wait={deps.fmt_ms(upstream_t0)}"
                            )
                            yield _build_openai_keepalive_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                            )
                else:
                    response = await upstream_open_task

                deps = scope_stop_detection(deps, response, url, trace_prefix)
                deps.log(
                    f"{trace_prefix}claude_upstream_headers status={response.status_code} "
                    f"{format_account_pool_route(response)} "
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
                    cache_keepalive_task = spawn_detached(
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
                    nonlocal upstream_data_started, last_upstream_data_time
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
                                upstream_data_started = True
                                last_upstream_data_time = time.perf_counter()
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
                        queue_wait_timeout = keepalive_interval
                        if (
                            stream_idle_timeout_enabled
                            and upstream_data_started
                            and last_upstream_data_time is not None
                            and (visible_delta_count > 0 or ordinary_tool_seen)
                        ):
                            idle_remaining = stream_idle_timeout_sec - (
                                time.perf_counter() - last_upstream_data_time
                            )
                            queue_wait_timeout = max(0.001, min(keepalive_interval, idle_remaining))
                        try:
                            queued_line = await asyncio.wait_for(
                                line_queue.get(),
                                timeout=queue_wait_timeout,
                            )
                        except asyncio.TimeoutError:
                            now = time.perf_counter()
                            upstream_data_idle = (
                                stream_idle_timeout_enabled
                                and upstream_data_started
                                and last_upstream_data_time is not None
                                and (visible_delta_count > 0 or ordinary_tool_seen)
                                and now - last_upstream_data_time >= stream_idle_timeout_sec
                            )
                            if upstream_data_idle:
                                finish_status = "anthropic_upstream_data_idle_timeout"
                                idle_for = now - last_upstream_data_time
                                deps.log(
                                    f"{trace_prefix}claude_upstream_data_idle_timeout "
                                    f"idle={idle_for:.3f}s limit={stream_idle_timeout_sec:.3f}s "
                                    f"lines={line_count} data_lines={data_line_count} "
                                    f"out_chars={len(full_response)} last_event={last_event_type}"
                                )
                                deps.save_request_log(
                                    model,
                                    messages,
                                    full_response or "[empty response]",
                                    stream=True,
                                    raw_sse="\n".join(
                                        raw_sse_lines
                                        + [f"[UPSTREAM_DATA_IDLE_TIMEOUT: {idle_for:.3f}s]"]
                                    ),
                                    request_payload=request_data,
                                    error_type="claude_upstream_data_idle_timeout",
                                    trace_id=trace_id,
                                )
                                await close_upstream_for_early_stop(
                                    response=response,
                                    client=client,
                                    reader_task=upstream_reader,
                                    reason="upstream_data_idle_timeout",
                                )
                                yield deps.build_openai_sse_error(
                                    504,
                                    (
                                        "Anthropic upstream emitted no SSE data for "
                                        f"{stream_idle_timeout_sec:g} seconds"
                                    ),
                                    "upstream_stream_idle_timeout",
                                )
                                yield emit_done()
                                return
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

                        refusal_diagnostics.observe(data)

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

                        if event_type == "content_block_start":
                            block = data.get("content_block")
                            if not isinstance(block, dict) or block.get("type") != "tool_use":
                                continue
                            block_index = int(data.get("index") or 0)
                            tool_name = str(block.get("name") or "")
                            if tool_name == TERMINAL_TOOL_NAME:
                                suppressed_terminal_blocks.add(block_index)
                                finish_status = "terminal_tool"
                                deps.log(
                                    f"{trace_prefix}terminal_tool protocol=anthropic "
                                    f"after_ordinary={ordinary_tool_seen} index={block_index}"
                                )
                                deps.save_request_log(
                                    model,
                                    messages,
                                    full_response or "[empty response]",
                                    stream=True,
                                    raw_sse="\n".join(raw_sse_lines + ["[TERMINAL_TOOL by proxy]"]),
                                    request_payload=request_data,
                                    trace_id=trace_id,
                                )
                                await close_upstream_for_early_stop(
                                    response=response,
                                    client=client,
                                    reader_task=upstream_reader,
                                    reason="terminal_tool",
                                )
                                stop_chunk = _build_openai_chunk(
                                    stream_id=stream_id,
                                    created=created,
                                    model=model,
                                    finish_reason="tool_calls" if ordinary_tool_seen else "stop",
                                )
                                yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                                yield emit_done()
                                return

                            ordinary_tool_seen = True
                            if not sent_role_chunk:
                                role_chunk = _build_openai_chunk(
                                    stream_id=stream_id,
                                    created=created,
                                    model=model,
                                    role="assistant",
                                )
                                yield f"data: {json.dumps(role_chunk, ensure_ascii=True)}\n\n".encode()
                                sent_role_chunk = True
                            initial_input = block.get("input")
                            initial_arguments = (
                                json.dumps(initial_input, ensure_ascii=False, separators=(",", ":"))
                                if isinstance(initial_input, dict) and initial_input
                                else ""
                            )
                            tool_chunk = _build_openai_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                            )
                            tool_chunk["choices"][0]["delta"]["tool_calls"] = [{
                                "index": block_index,
                                "id": str(block.get("id") or f"toolu_{block_index}"),
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": initial_arguments,
                                },
                            }]
                            yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n".encode()
                            last_downstream_emit = time.perf_counter()
                            continue

                        if event_type == "content_block_delta":
                            block_index = int(data.get("index") or 0)
                            raw_delta = data.get("delta")
                            if isinstance(raw_delta, dict) and raw_delta.get("type") == "input_json_delta":
                                if block_index in suppressed_terminal_blocks:
                                    continue
                                partial_json = raw_delta.get("partial_json")
                                if isinstance(partial_json, str) and partial_json:
                                    ordinary_tool_seen = True
                                    tool_chunk = _build_openai_chunk(
                                        stream_id=stream_id,
                                        created=created,
                                        model=model,
                                    )
                                    tool_chunk["choices"][0]["delta"]["tool_calls"] = [{
                                        "index": block_index,
                                        "function": {"arguments": partial_json},
                                    }]
                                    yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n".encode()
                                    last_downstream_emit = time.perf_counter()
                                continue
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
                            if deps.retire_refused_session(stop_reason, trace_prefix):
                                _cancel_claude_cache_post_keepalive(
                                    post_key=cache_post_keepalive_key,
                                    deps=deps,
                                    trace_prefix=trace_prefix,
                                    reason="upstream_refusal",
                                )
                            stop_chunk = _build_openai_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                                finish_reason=(
                                    "tool_calls"
                                    if ordinary_tool_seen
                                    else _map_stop_reason(stop_reason)
                                ),
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
                        except RuntimeError as e:
                            if _is_async_generator_close_race(e):
                                deps.log(
                                    f"{trace_prefix}claude_upstream_reader_cancel_close_race_ignored "
                                    f"err={e}"
                                )
                            else:
                                raise

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
            finally:
                if not upstream_open_task.done():
                    upstream_open_task.cancel()
                    try:
                        await upstream_open_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                if response is not None:
                    try:
                        await response.aclose()
                    except Exception:
                        pass
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
        if _is_async_generator_close_race(e):
            finish_status = "downstream_cancelled"
            deps.save_request_log(
                model,
                messages,
                full_response or "[cancelled during async generator close]",
                stream=True,
                raw_sse="\n".join(
                    raw_sse_lines
                    + ["[CANCELLED: downstream client disconnected / async generator close race]"]
                ),
                request_payload=request_data,
                error_type="downstream_cancelled",
                trace_id=trace_id,
            )
            deps.log(
                f"{trace_prefix}claude_stream_close_race_treated_as_downstream_cancelled "
                f"type={type(e).__name__} err={e}"
            )
            return
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
        if cache_keepalive_enabled and isinstance(cache_keepalive, dict) and finish_status not in {
            "early_stop_tag",
            "terminal_tool",
            "downstream_cancelled",
            "anthropic_upstream_data_idle_timeout",
            "message_stop:refusal",
        }:
            post_state = _ClaudeCachePostKeepaliveState(token=uuid.uuid4().hex)
            shared = getattr(deps, 'shared_runtime_state', None)
            if shared is not None:
                shared.replace_token('claude-post:'+cache_post_keepalive_key, post_state.token)
            post_task = spawn_detached(
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
            f"usage_total={usage.get('total_tokens', 0)}{refusal_diagnostics.log_suffix()}"
        )
