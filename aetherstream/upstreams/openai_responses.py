import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable

import httpx

from aetherstream.features.terminal_tool import TERMINAL_TOOL_NAME
from aetherstream.upstreams.route_logging import format_account_pool_route, scope_stop_detection


@dataclass
class ResponsesUpstreamDeps:
    log: Callable[[str], None]
    save_request_log: Callable[..., None]
    build_openai_sse_error: Callable[[int, str, str], bytes]
    has_stop_tag: Callable[[str], bool]
    find_stop_tag: Callable[[str], int]
    fmt_ms: Callable[[float, float | None], str]
    release_caller: Callable[[str, str], None]


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


def _build_openai_chunk(
    *,
    stream_id: str,
    created: int,
    model: str,
    delta: str = "",
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta_payload: dict[str, Any] = {}
    if delta:
        delta_payload["content"] = delta
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


def _build_openai_tool_chunk(
    *,
    stream_id: str,
    created: int,
    model: str,
    index: int,
    call_id: str = "",
    name: str = "",
    arguments: str = "",
) -> dict[str, Any]:
    function: dict[str, Any] = {}
    if name:
        function["name"] = name
    if arguments:
        function["arguments"] = arguments
    call: dict[str, Any] = {"index": index, "function": function}
    if call_id:
        call["id"] = call_id
        call["type"] = "function"
    chunk = _build_openai_chunk(
        stream_id=stream_id,
        created=created,
        model=model,
    )
    chunk["choices"][0]["delta"]["tool_calls"] = [call]
    return chunk


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    normalized.update(usage)
    return normalized


def _extract_completed_text(response_obj: dict[str, Any]) -> str:
    output = response_obj.get("output")
    if not isinstance(output, list):
        return ""
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                text_parts.append(content["text"])
    return "".join(text_parts)


def _extract_completed_function_calls(response_obj: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in response_obj.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(item)
    return calls


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


async def _close_upstream_stream(
    *,
    response: httpx.Response | None,
    client: httpx.AsyncClient | None,
    deps: ResponsesUpstreamDeps,
    trace_prefix: str,
    label: str,
    reason: str,
    started_at: float,
    line_count: int,
    data_line_count: int,
    out_chars: int,
) -> None:
    close_t0 = time.perf_counter()
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


async def forward_responses_as_chat_stream(
    *,
    url: str,
    api_key: str,
    request_data: dict[str, Any],
    model: str,
    messages: list,
    trace_id: str,
    caller_key: str,
    caller_desc: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: ResponsesUpstreamDeps,
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
    saw_completed = False
    final_usage: dict[str, Any] = {}
    saved_log = False
    keepalive_interval = 10.0
    keepalive_count = 0
    last_downstream_emit = time.perf_counter()
    ordinary_tool_seen = False
    tool_index_by_item_id: dict[str, int] = {}
    tool_id_by_item_id: dict[str, str] = {}
    tool_arguments_by_item_id: dict[str, str] = {}
    suppressed_terminal_item_ids: set[str] = set()
    next_tool_index = 0

    def persist_log(response_text: str, *, raw_sse: str = "", error_type: str | None = None) -> None:
        nonlocal saved_log
        if saved_log:
            return
        try:
            deps.save_request_log(
                model,
                messages,
                response_text,
                stream=True,
                raw_sse=raw_sse,
                request_payload=request_data,
                error_type=error_type,
                trace_id=trace_id,
            )
            saved_log = True
        except Exception as save_err:
            deps.log(f"{trace_prefix}responses_stream_log_error err={save_err}")

    def emit_done() -> bytes:
        return b"data: [DONE]\n\n"

    def emit_keepalive(reason: str = "idle", *, force: bool = False) -> bytes | None:
        nonlocal keepalive_count, last_downstream_emit
        now = time.perf_counter()
        if not force and now - last_downstream_emit < keepalive_interval:
            return None
        keepalive_count += 1
        idle_ms = (now - last_downstream_emit) * 1000
        last_downstream_emit = now
        deps.log(
            f"{trace_prefix}responses_downstream_keepalive "
            f"count={keepalive_count} reason={reason} idle_ms={idle_ms:.1f}"
        )
        chunk = _build_openai_chunk(
            stream_id=stream_id,
            created=created,
            model=model,
        )
        return f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n".encode()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": f"StreamProxy/1.0 OpenAI-Responses/{model}",
    }

    async def open_upstream() -> tuple[httpx.AsyncClient, Any, httpx.Response]:
        client = httpx.AsyncClient(timeout=timeout, http2=True)
        try:
            upstream_t0 = time.perf_counter()
            cm = client.stream("POST", url, json=request_data, headers=headers)
            response = await cm.__aenter__()
            deps.log(
                f"{trace_prefix}responses_upstream_headers status={response.status_code} "
                f"{format_account_pool_route(response)} "
                f"elapsed={deps.fmt_ms(upstream_t0)}"
            )
            return client, cm, response
        except Exception:
            await client.aclose()
            raise

    upstream_task: asyncio.Task | None = None

    try:
        initial_keepalive = emit_keepalive("stream_open", force=True)
        if initial_keepalive:
            yield initial_keepalive

        upstream_task = asyncio.create_task(open_upstream())
        while not upstream_task.done():
            try:
                client, cm, response = await asyncio.wait_for(
                    asyncio.shield(upstream_task),
                    timeout=keepalive_interval,
                )
                break
            except asyncio.TimeoutError:
                keepalive = emit_keepalive("waiting_headers")
                if keepalive:
                    yield keepalive
        else:
            client, cm, response = await upstream_task

        deps = scope_stop_detection(deps, response, url, trace_prefix)

        try:
            if response.status_code != 200:
                content = await response.aread()
                error_text = content.decode(errors="replace")[:4000]
                persist_log(
                    f"[UPSTREAM_HTTP_ERROR] {error_text}",
                    raw_sse=f"[HTTP {response.status_code}]\n{error_text}",
                    error_type="responses_upstream_http_error",
                )
                finish_status = f"upstream_http_{response.status_code}"
                yield deps.build_openai_sse_error(
                    response.status_code,
                    error_text,
                    "responses_upstream_http_error",
                )
                yield emit_done()
                return

            async for line in response.aiter_lines():
                line_count += 1
                raw_sse_size, raw_sse_truncated = _append_raw_line(
                    raw_sse_lines,
                    line,
                    raw_size=raw_sse_size,
                    raw_truncated=raw_sse_truncated,
                    max_raw_sse_bytes=max_raw_sse_bytes,
                )

                if not line or line.startswith("event:"):
                    keepalive = emit_keepalive()
                    if keepalive:
                        yield keepalive
                    continue
                if not line.startswith("data: "):
                    keepalive = emit_keepalive()
                    if keepalive:
                        yield keepalive
                    continue

                data_line_count += 1
                if first_data_time is None:
                    first_data_time = time.perf_counter()
                    deps.log(
                        f"{trace_prefix}responses_upstream_first_data "
                        f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                    )

                payload = line[6:].strip()
                if payload == "[DONE]":
                    keepalive = emit_keepalive()
                    if keepalive:
                        yield keepalive
                    continue

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if isinstance(data.get("error"), dict):
                    err = data["error"]
                    err_msg = err.get("message") or json.dumps(err, ensure_ascii=False)
                    persist_log(
                        f"[UPSTREAM_ERROR_EVENT] {err_msg}",
                        raw_sse="\n".join(raw_sse_lines),
                        error_type="responses_upstream_error_event",
                    )
                    finish_status = "upstream_error_event"
                    yield deps.build_openai_sse_error(
                        502,
                        err_msg[:4000],
                        err.get("type", "responses_upstream_error"),
                    )
                    yield emit_done()
                    return

                event_type = data.get("type", "")
                response_obj = data.get("response") if isinstance(data.get("response"), dict) else {}
                if response_obj.get("id"):
                    stream_id = f"chatcmpl-{response_obj['id']}"
                if isinstance(response_obj.get("created_at"), int):
                    created = response_obj["created_at"]
                if response_obj.get("model"):
                    model = response_obj["model"]

                if event_type in {"response.output_item.added", "response.output_item.done"}:
                    item = data.get("item")
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        name = str(item.get("name") or "")
                        item_id = str(item.get("id") or item.get("call_id") or f"tool-{next_tool_index}")
                        call_id = str(item.get("call_id") or item.get("id") or item_id)
                        if name == TERMINAL_TOOL_NAME:
                            suppressed_terminal_item_ids.add(item_id)
                            suppressed_terminal_item_ids.add(call_id)
                            if ordinary_tool_seen:
                                deps.log(
                                    f"{trace_prefix}terminal_tool_invalid protocol=responses "
                                    "reason=ordinary_tool_already_seen"
                                )
                                continue
                            finish_status = "terminal_tool"
                            persist_log(
                                full_response or "[empty response]",
                                raw_sse="\n".join(raw_sse_lines + ["[TERMINAL_TOOL by proxy]"]),
                            )
                            await _close_upstream_stream(
                                response=response,
                                client=client,
                                deps=deps,
                                trace_prefix=trace_prefix,
                                label="terminal_tool",
                                reason="responses_terminal_tool",
                                started_at=request_t0,
                                line_count=line_count,
                                data_line_count=data_line_count,
                                out_chars=len(full_response),
                            )
                            stop_chunk = _build_openai_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                                finish_reason="stop",
                            )
                            yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                            yield emit_done()
                            return

                        if item_id not in tool_index_by_item_id:
                            tool_index_by_item_id[item_id] = next_tool_index
                            tool_id_by_item_id[item_id] = call_id
                            if call_id != item_id:
                                tool_index_by_item_id[call_id] = next_tool_index
                                tool_id_by_item_id[call_id] = call_id
                            next_tool_index += 1
                            ordinary_tool_seen = True
                            arguments = item.get("arguments")
                            if not isinstance(arguments, str):
                                arguments = ""
                            tool_arguments_by_item_id[item_id] = arguments
                            if call_id != item_id:
                                tool_arguments_by_item_id[call_id] = arguments
                            tool_chunk = _build_openai_tool_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                                index=tool_index_by_item_id[item_id],
                                call_id=call_id,
                                name=name,
                                arguments=arguments,
                            )
                            last_downstream_emit = time.perf_counter()
                            yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n".encode()
                        elif event_type == "response.output_item.done":
                            arguments = item.get("arguments")
                            if isinstance(arguments, str):
                                previous_arguments = tool_arguments_by_item_id.get(item_id, "")
                                if arguments.startswith(previous_arguments):
                                    remaining_arguments = arguments[len(previous_arguments):]
                                elif arguments != previous_arguments:
                                    remaining_arguments = arguments
                                else:
                                    remaining_arguments = ""
                                if remaining_arguments:
                                    tool_arguments_by_item_id[item_id] = arguments
                                    tool_chunk = _build_openai_tool_chunk(
                                        stream_id=stream_id,
                                        created=created,
                                        model=model,
                                        index=tool_index_by_item_id[item_id],
                                        arguments=remaining_arguments,
                                    )
                                    last_downstream_emit = time.perf_counter()
                                    yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n".encode()
                        continue

                if event_type == "response.function_call_arguments.delta":
                    item_id = str(data.get("item_id") or data.get("call_id") or "")
                    if item_id in suppressed_terminal_item_ids:
                        continue
                    if item_id not in tool_index_by_item_id:
                        tool_index_by_item_id[item_id] = next_tool_index
                        tool_id_by_item_id[item_id] = str(data.get("call_id") or item_id)
                        next_tool_index += 1
                    ordinary_tool_seen = True
                    arguments = data.get("delta")
                    if isinstance(arguments, str) and arguments:
                        tool_arguments_by_item_id[item_id] = (
                            tool_arguments_by_item_id.get(item_id, "") + arguments
                        )
                        tool_chunk = _build_openai_tool_chunk(
                            stream_id=stream_id,
                            created=created,
                            model=model,
                            index=tool_index_by_item_id[item_id],
                            arguments=arguments,
                        )
                        last_downstream_emit = time.perf_counter()
                        yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n".encode()
                    continue

                if event_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if not isinstance(delta, str) or not delta:
                        continue
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
                        persist_log(
                            full_response or "[empty response]",
                            raw_sse="\n".join(raw_sse_lines),
                        )
                        finish_status = "early_stop_tag"
                        await _close_upstream_stream(
                            response=response,
                            client=client,
                            deps=deps,
                            trace_prefix=trace_prefix,
                            label="early_stop",
                            reason="responses_stream_stop_tag",
                            started_at=request_t0,
                            line_count=line_count,
                            data_line_count=data_line_count,
                            out_chars=len(full_response),
                        )
                        yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                        yield emit_done()
                        return

                    chunk = _build_openai_chunk(
                        stream_id=stream_id,
                        created=created,
                        model=model,
                        delta=delta,
                    )
                    last_downstream_emit = time.perf_counter()
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                    continue

                keepalive = emit_keepalive()
                if keepalive:
                    yield keepalive

                if event_type == "response.completed":
                    saw_completed = True
                    if response_obj:
                        final_usage = _normalize_usage(response_obj.get("usage"))
                        completed_calls = _extract_completed_function_calls(response_obj)
                        completed_terminal = any(
                            call.get("name") == TERMINAL_TOOL_NAME
                            for call in completed_calls
                        )
                        completed_ordinary = [
                            call
                            for call in completed_calls
                            if call.get("name") != TERMINAL_TOOL_NAME
                        ]
                        if completed_terminal and not ordinary_tool_seen and not completed_ordinary:
                            stop_chunk = _build_openai_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                                finish_reason="stop",
                            )
                            persist_log(
                                full_response or "[empty response]",
                                raw_sse="\n".join(raw_sse_lines + ["[TERMINAL_TOOL at response.completed]"]),
                            )
                            finish_status = "terminal_tool_at_response_completed"
                            yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                            yield emit_done()
                            return
                        for call in completed_ordinary:
                            item_id = str(call.get("id") or call.get("call_id") or "")
                            call_id = str(call.get("call_id") or call.get("id") or item_id)
                            if item_id in tool_index_by_item_id or call_id in tool_index_by_item_id:
                                continue
                            arguments = call.get("arguments")
                            if not isinstance(arguments, str):
                                arguments = json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":"))
                            tool_index = next_tool_index
                            next_tool_index += 1
                            ordinary_tool_seen = True
                            tool_index_by_item_id[item_id or call_id] = tool_index
                            tool_chunk = _build_openai_tool_chunk(
                                stream_id=stream_id,
                                created=created,
                                model=model,
                                index=tool_index,
                                call_id=call_id,
                                name=str(call.get("name") or ""),
                                arguments=arguments,
                            )
                            yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n".encode()
                        completed_text = _extract_completed_text(response_obj)
                        if not full_response and completed_text:
                            full_response = completed_text
                            if deps.has_stop_tag(full_response):
                                full_response = full_response[:deps.find_stop_tag(full_response)]
                            if full_response:
                                chunk = _build_openai_chunk(
                                    stream_id=stream_id,
                                    created=created,
                                    model=model,
                                    delta=full_response,
                                )
                                last_downstream_emit = time.perf_counter()
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

                    stop_chunk = _build_openai_chunk(
                        stream_id=stream_id,
                        created=created,
                        model=model,
                        finish_reason="tool_calls" if ordinary_tool_seen else "stop",
                    )
                    persist_log(
                        full_response or "[empty response]",
                        raw_sse="\n".join(raw_sse_lines),
                    )
                    finish_status = "responses_response_completed"
                    last_downstream_emit = time.perf_counter()
                    yield f"data: {json.dumps(stop_chunk, ensure_ascii=True)}\n\n".encode()
                    yield emit_done()
                    return

            finish_status = "responses_incomplete_stream_end"
            persist_log(
                full_response or "[empty response]",
                raw_sse="\n".join(raw_sse_lines + ["[UPSTREAM_INCOMPLETE]"]),
                error_type="responses_upstream_incomplete",
            )
            yield deps.build_openai_sse_error(
                502,
                "OpenAI Responses stream closed without response.completed",
                "upstream_stream_incomplete",
            )
            yield emit_done()
        except (asyncio.CancelledError, GeneratorExit) as close_exc:
            if isinstance(close_exc, asyncio.CancelledError):
                close_reason = "downstream_cancelled"
                close_marker = "[CANCELLED: downstream client disconnected]"
            else:
                close_reason = "downstream_closed"
                close_marker = "[GENERATOR_CLOSED: downstream stopped consuming stream]"
            if finish_status == "unknown":
                finish_status = close_reason
            persist_log(
                full_response or "[cancelled before content]",
                raw_sse="\n".join(raw_sse_lines + [close_marker]),
                error_type=close_reason,
            )
            raise
        finally:
            await cm.__aexit__(None, None, None)
            await client.aclose()
    except asyncio.CancelledError:
        finish_status = "downstream_cancelled"
        if upstream_task is not None and not upstream_task.done():
            upstream_task.cancel()
        persist_log(
            full_response or "[cancelled before content]",
            raw_sse="\n".join(raw_sse_lines + ["[CANCELLED: downstream client disconnected]"]),
            error_type="downstream_cancelled",
        )
        raise
    except GeneratorExit:
        if finish_status == "unknown":
            finish_status = "downstream_closed"
        if upstream_task is not None and not upstream_task.done():
            upstream_task.cancel()
        persist_log(
            full_response or "[cancelled before content]",
            raw_sse="\n".join(raw_sse_lines + ["[GENERATOR_CLOSED: downstream stopped consuming stream]"]),
            error_type="downstream_closed",
        )
        raise
    except Exception as e:
        finish_status = f"exception:{type(e).__name__}"
        persist_log(
            full_response + f"\n[ERROR] {e}" if full_response else f"[ERROR] {e}",
            raw_sse="\n".join(raw_sse_lines + [f"[EXCEPTION] {e}"]),
            error_type="responses_upstream_proxy_error",
        )
        yield deps.build_openai_sse_error(502, str(e), "responses_upstream_proxy_error")
        yield emit_done()
    finally:
        if not saved_log:
            if finish_status == "unknown":
                finish_status = "generator_finalized_without_terminal_event"
            persist_log(
                full_response or "[stream finalized before content]",
                raw_sse="\n".join(raw_sse_lines + ["[FINALIZED_WITHOUT_TERMINAL_EVENT]"]),
                error_type="responses_stream_finalized",
            )
        if caller_key:
            deps.release_caller(caller_key, trace_id)
        deps.log(
            f"{trace_prefix}responses_upstream_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} "
            f"out_chars={len(full_response)} usage_total={final_usage.get('total_tokens', 0)} "
            f"keepalives={keepalive_count}"
        )


async def collect_responses_as_chat_completion(
    *,
    url: str,
    api_key: str,
    request_data: dict[str, Any],
    model: str,
    messages: list,
    trace_id: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: ResponsesUpstreamDeps,
    save_log: bool = True,
) -> tuple[str, dict[str, Any], str]:
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    full_response = ""
    finish_reason = "stop"
    usage: dict[str, Any] = {}
    raw_sse_lines: list[str] = []
    raw_sse_size = 0
    raw_sse_truncated = False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": f"StreamProxy/1.0 OpenAI-Responses/{model}",
    }

    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        upstream_t0 = time.perf_counter()
        async with client.stream("POST", url, json=request_data, headers=headers) as response:
            deps = scope_stop_detection(deps, response, url, trace_prefix)
            deps.log(
                f"{trace_prefix}responses_collect_headers status={response.status_code} "
                f"{format_account_pool_route(response)} "
                f"elapsed={deps.fmt_ms(upstream_t0)}"
            )
            if response.status_code != 200:
                error_text = (await response.aread()).decode(errors="replace")[:4000]
                deps.save_request_log(
                    model,
                    messages,
                    f"[UPSTREAM_HTTP_ERROR] {error_text}",
                    stream=False,
                    raw_sse=f"[HTTP {response.status_code}]\n{error_text}",
                    request_payload=request_data,
                    error_type="responses_upstream_http_error",
                    trace_id=trace_id,
                )
                raise RuntimeError(f"OpenAI Responses upstream HTTP {response.status_code}: {error_text}")

            async for line in response.aiter_lines():
                raw_sse_size, raw_sse_truncated = _append_raw_line(
                    raw_sse_lines,
                    line,
                    raw_size=raw_sse_size,
                    raw_truncated=raw_sse_truncated,
                    max_raw_sse_bytes=max_raw_sse_bytes,
                )

                if not line or line.startswith("event:") or not line.startswith("data: "):
                    continue

                payload = line[6:].strip()
                if payload == "[DONE]":
                    continue

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if isinstance(data.get("error"), dict):
                    err = data["error"]
                    err_msg = err.get("message") or json.dumps(err, ensure_ascii=False)
                    deps.save_request_log(
                        model,
                        messages,
                        f"[UPSTREAM_ERROR_EVENT] {err_msg}",
                        stream=False,
                        raw_sse="\n".join(raw_sse_lines),
                        request_payload=request_data,
                        error_type="responses_upstream_error_event",
                        trace_id=trace_id,
                    )
                    raise RuntimeError(err_msg)

                event_type = data.get("type", "")
                response_obj = data.get("response") if isinstance(data.get("response"), dict) else {}
                if response_obj.get("model"):
                    model = response_obj["model"]

                if event_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if isinstance(delta, str) and delta:
                        full_response += delta
                        if deps.has_stop_tag(full_response):
                            full_response = full_response[:deps.find_stop_tag(full_response)]
                            await _close_upstream_stream(
                                response=response,
                                client=client,
                                deps=deps,
                                trace_prefix=trace_prefix,
                                label="early_stop",
                                reason="responses_collect_stop_tag",
                                started_at=request_t0,
                                line_count=0,
                                data_line_count=0,
                                out_chars=len(full_response),
                            )
                            break
                    continue

                if event_type == "response.completed":
                    usage = _normalize_usage(response_obj.get("usage"))
                    completed_text = _extract_completed_text(response_obj)
                    if not full_response and completed_text:
                        full_response = completed_text
                        if deps.has_stop_tag(full_response):
                            full_response = full_response[:deps.find_stop_tag(full_response)]
                    break

    if save_log:
        deps.save_request_log(
            model,
            messages,
            full_response or "[empty response]",
            stream=False,
            raw_sse="\n".join(raw_sse_lines),
            request_payload=request_data,
            trace_id=trace_id,
        )
    deps.log(
        f"{trace_prefix}responses_collect_done elapsed={deps.fmt_ms(request_t0)} "
        f"out_chars={len(full_response)} usage_total={usage.get('total_tokens', 0)}"
    )
    return full_response, usage, finish_reason


async def collect_responses_nonstream(
    *,
    url: str,
    api_key: str,
    request_data: dict[str, Any],
    model: str,
    messages: list,
    trace_id: str,
    timeout: httpx.Timeout,
    deps: ResponsesUpstreamDeps,
    save_log: bool = True,
) -> tuple[str, dict[str, Any], str, str, list[dict[str, Any]]]:
    """Call OpenAI Responses upstream as true non-stream JSON and extract assistant text."""
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    payload = dict(request_data)
    payload["stream"] = False
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": f"StreamProxy/1.0 OpenAI-Responses/{model}",
    }
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        upstream_t0 = time.perf_counter()
        deps.log(
            f"{trace_prefix}responses_nonstream_request_start "
            f"elapsed={deps.fmt_ms(request_t0)} model={model}"
        )
        response = await client.post(url, json=payload, headers=headers)
        deps = scope_stop_detection(deps, response, url, trace_prefix)
        raw_text = response.text
        deps.log(
            f"{trace_prefix}responses_nonstream_response status={response.status_code} "
            f"{format_account_pool_route(response)} "
            f"elapsed={deps.fmt_ms(upstream_t0)} bytes={len(raw_text.encode(errors='replace'))}"
        )
        if response.status_code != 200:
            error_text = raw_text[:4000]
            if save_log:
                deps.save_request_log(
                    model,
                    messages,
                    f"[UPSTREAM_HTTP_ERROR] {error_text}",
                    stream=False,
                    raw_sse=f"[HTTP {response.status_code}]\n{error_text}",
                    request_payload=payload,
                    error_type="responses_nonstream_http_error",
                    trace_id=trace_id,
                )
            raise RuntimeError(f"OpenAI Responses upstream HTTP {response.status_code}: {error_text}")
        try:
            data = response.json()
        except Exception as exc:
            if save_log:
                deps.save_request_log(
                    model,
                    messages,
                    f"[UPSTREAM_NON_JSON] {raw_text[:4000]}",
                    stream=False,
                    raw_sse=raw_text[:4000],
                    request_payload=payload,
                    error_type="responses_nonstream_non_json",
                    trace_id=trace_id,
                )
            raise RuntimeError(f"OpenAI Responses upstream returned non-JSON response: {raw_text[:4000]}") from exc

    if isinstance(data.get("error"), dict):
        err = data["error"]
        err_msg = err.get("message") or json.dumps(err, ensure_ascii=False)
        if save_log:
            deps.save_request_log(
                model,
                messages,
                f"[UPSTREAM_ERROR] {err_msg}",
                stream=False,
                raw_sse=raw_text[:4000],
                request_payload=payload,
                error_type="responses_nonstream_error",
                trace_id=trace_id,
            )
        raise RuntimeError(err_msg)

    full_content = _extract_completed_text(data)
    if not full_content and isinstance(data.get("output_text"), str):
        full_content = data["output_text"]
    if deps.has_stop_tag(full_content):
        full_content = full_content[:deps.find_stop_tag(full_content)]
    usage = _normalize_usage(data.get("usage") if isinstance(data.get("usage"), dict) else {})
    finish_reason = "stop"
    status = str(data.get("status") or "")
    if status and status not in {"completed", "incomplete"}:
        finish_reason = status
    out_model = str(data.get("model") or model)
    function_calls = _extract_completed_function_calls(data)

    if save_log:
        deps.save_request_log(
            out_model,
            messages,
            full_content or "[empty response]",
            stream=False,
            raw_sse=raw_text,
            request_payload=payload,
            trace_id=trace_id,
        )
    deps.log(
        f"{trace_prefix}responses_nonstream_collect_done elapsed={deps.fmt_ms(request_t0)} "
        f"out_chars={len(full_content)} usage_total={usage.get('total_tokens', 0)} status={status or '-'}"
    )
    return full_content, usage, finish_reason, raw_text, function_calls


async def replay_responses_as_chat_stream(
    *,
    url: str,
    api_key: str,
    request_data: dict[str, Any],
    model: str,
    messages: list,
    trace_id: str,
    caller_key: str,
    caller_desc: str,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: ResponsesUpstreamDeps,
    chunk_size: int = 1200,
) -> AsyncGenerator[bytes, None]:
    """Call OpenAI Responses upstream as true non-stream, replay assistant text as OpenAI SSE."""
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    stream_id = f"chatcmpl-responses-replay-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    keepalive_interval = 10.0
    keepalive_count = 0
    last_downstream_emit = time.perf_counter()
    full_content = ""
    usage: dict[str, Any] = {}
    finish_reason = "stop"
    raw_text = ""
    finish_status = "unknown"
    saved_log = False
    collect_task: asyncio.Task | None = None

    def make_chunk(*, delta: str = "", finish_reason_value: str | None = None) -> dict[str, Any]:
        return _build_openai_chunk(
            stream_id=stream_id,
            created=created,
            model=model,
            delta=delta,
            finish_reason=finish_reason_value,
        )

    def emit_data(obj: dict[str, Any]) -> bytes:
        nonlocal last_downstream_emit
        last_downstream_emit = time.perf_counter()
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    def emit_done() -> bytes:
        nonlocal last_downstream_emit
        last_downstream_emit = time.perf_counter()
        return b"data: [DONE]\n\n"

    def emit_keepalive(reason: str = "idle", *, force: bool = False) -> bytes | None:
        nonlocal keepalive_count, last_downstream_emit
        now = time.perf_counter()
        if not force and now - last_downstream_emit < keepalive_interval:
            return None
        idle_ms = (now - last_downstream_emit) * 1000
        last_downstream_emit = now
        keepalive_count += 1
        deps.log(
            f"{trace_prefix}responses_replay_keepalive "
            f"count={keepalive_count} reason={reason} idle_ms={idle_ms:.1f}"
        )
        return f"data: {json.dumps(make_chunk(), ensure_ascii=True)}\n\n".encode("utf-8")

    def persist_log(marker: str = "", response_override: str | None = None, error_type: str | None = None) -> None:
        nonlocal saved_log
        if saved_log:
            return
        raw = raw_text or marker or "[RESPONSES_NONSTREAM_REPLAY]"
        if marker and raw_text:
            raw = f"{raw_text}\n{marker}"
        try:
            kwargs = {
                "stream": True,
                "raw_sse": raw,
                "request_payload": dict(request_data, stream=False),
                "trace_id": trace_id,
            }
            if error_type:
                kwargs["error_type"] = error_type
            deps.save_request_log(
                model,
                messages,
                response_override if response_override is not None else (full_content or "[empty response]"),
                **kwargs,
            )
            saved_log = True
        except Exception as save_err:
            deps.log(f"{trace_prefix}responses_replay_log_error err={save_err}")

    if caller_key:
        deps.log(f"{trace_prefix}responses_replay_owner caller={caller_key} {caller_desc}")

    try:
        initial = emit_keepalive("stream_open", force=True)
        if initial:
            yield initial

        collect_task = asyncio.create_task(collect_responses_nonstream(
            url=url,
            api_key=api_key,
            request_data=request_data,
            model=model,
            messages=messages,
            trace_id=trace_id,
            timeout=timeout,
            deps=deps,
            save_log=False,
        ))

        while not collect_task.done():
            try:
                full_content, usage, finish_reason, raw_text, function_calls = await asyncio.wait_for(
                    asyncio.shield(collect_task),
                    timeout=keepalive_interval,
                )
                break
            except asyncio.TimeoutError:
                keepalive = emit_keepalive("waiting_nonstream_body")
                if keepalive:
                    yield keepalive
        else:
            full_content, usage, finish_reason, raw_text, function_calls = await collect_task

        step = max(1, int(chunk_size) or 1200)
        for start in range(0, len(full_content), step):
            piece = full_content[start:start + step]
            if piece:
                yield emit_data(make_chunk(delta=piece))

        ordinary_calls: list[dict[str, Any]] = []
        terminal_tool_seen = False
        for call in function_calls:
            if call.get("name") == TERMINAL_TOOL_NAME:
                terminal_tool_seen = True
            else:
                ordinary_calls.append(call)
        for index, call in enumerate(ordinary_calls):
            arguments = call.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":"))
            yield emit_data(_build_openai_tool_chunk(
                stream_id=stream_id,
                created=created,
                model=model,
                index=index,
                call_id=str(call.get("call_id") or call.get("id") or f"call_{index}"),
                name=str(call.get("name") or ""),
                arguments=arguments,
            ))
        if ordinary_calls:
            finish_reason = "tool_calls"
        elif terminal_tool_seen:
            finish_reason = "stop"
            finish_status = "terminal_tool_nonstream_replay"

        yield emit_data(make_chunk(finish_reason_value=finish_reason or "stop"))
        yield emit_done()
        if finish_status != "terminal_tool_nonstream_replay":
            finish_status = "responses_nonstream_replay_completed"
        persist_log("[RESPONSES_NONSTREAM_REPLAY_COMPLETED]", full_content or "[empty response]")
    except asyncio.CancelledError:
        finish_status = "downstream_cancelled"
        if collect_task is not None and not collect_task.done():
            collect_task.cancel()
        persist_log("[CANCELLED: downstream client disconnected during Responses nonstream replay]", full_content or "[cancelled before content]", "downstream_cancelled")
        raise
    except Exception as e:
        finish_status = f"exception:{type(e).__name__}"
        persist_log(f"[RESPONSES_NONSTREAM_REPLAY_EXCEPTION] {e}", full_content + f"\n[ERROR] {e}" if full_content else f"[ERROR] {e}", "responses_nonstream_replay_error")
        yield deps.build_openai_sse_error(502, str(e), "responses_nonstream_replay_error")
        yield emit_done()
    finally:
        if caller_key:
            deps.release_caller(caller_key, trace_id)
        deps.log(
            f"{trace_prefix}responses_nonstream_replay_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} out_chars={len(full_content)} "
            f"usage_total={usage.get('total_tokens', 0)} keepalives={keepalive_count}"
        )
