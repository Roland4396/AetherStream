import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable

import httpx

from aetherstream.features.early_stop import is_himodels_upstream
from aetherstream.features.glm_thinking import enforce_glm52_official_thinking
from aetherstream.features.terminal_tool import TERMINAL_TOOL_NAME, OpenAIChatTerminalDetector
from aetherstream.upstreams.route_logging import format_account_pool_route, scope_stop_detection


@dataclass
class ChatCompletionsUpstreamDeps:
    log: Callable[[str], None]
    save_request_log: Callable[..., None]
    build_openai_sse_error: Callable[[int, str, str], bytes]
    has_stop_tag: Callable[[str], bool]
    find_stop_tag: Callable[[str], int]
    fmt_ms: Callable[[float, float | None], str]
    release_caller: Callable[[str, str], None]


def _extract_openai_content(data: dict) -> str:
    if "choices" in data and data["choices"]:
        choice = data["choices"][0]
        delta = choice.get("delta", {})
        return delta.get("content", "")
    return ""


def _append_raw_sse_line(
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


def _build_openai_chunk(
    *,
    stream_id: str,
    created: int,
    model: str,
    delta: str = "",
    role: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta_payload: dict[str, Any] = {}
    if role:
        delta_payload["role"] = role
    if delta:
        delta_payload["content"] = delta

    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [{
            "delta": delta_payload,
            "logprobs": None,
            "finish_reason": finish_reason,
            "index": 0,
        }],
        "usage": usage,
    }


def _coerce_openai_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _extract_openai_message_content(data: dict[str, Any]) -> str:
    if "choices" not in data or not data["choices"]:
        return ""
    choice = data["choices"][0]
    message = choice.get("message", {})
    if isinstance(message, dict):
        return _coerce_openai_text_content(message.get("content"))
    return ""


def _extract_openai_message_reasoning(data: dict[str, Any]) -> str:
    if "choices" not in data or not data["choices"]:
        return ""
    choice = data["choices"][0]
    message = choice.get("message", {})
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_openai_finish_reason(data: dict[str, Any]) -> str | None:
    if "choices" not in data or not data["choices"]:
        return None
    finish_reason = data["choices"][0].get("finish_reason")
    return str(finish_reason) if finish_reason else None


def _truncate_for_stop_tag(
    full_response: str,
    delta: str,
    find_stop_tag: Callable[[str], int],
) -> tuple[str, str]:
    prefix_pos = find_stop_tag(full_response)
    if prefix_pos < 0:
        return full_response, delta

    prev_len = len(full_response) - len(delta)
    if prefix_pos >= prev_len:
        keep_len = prefix_pos - prev_len
        clean_delta = delta[:keep_len]
    else:
        clean_delta = ""

    return full_response[:prefix_pos], clean_delta


async def _close_upstream_stream(
    *,
    response: httpx.Response | None,
    client: httpx.AsyncClient | None,
    deps: ChatCompletionsUpstreamDeps,
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


async def replay_chat_completions_stream(
    *,
    raw_sse_text: str,
    request_data: dict,
    model: str,
    messages: list | None,
    trace_id: str = "",
    caller_key: str = "",
    caller_desc: str = "",
    deps: ChatCompletionsUpstreamDeps,
    enable_early_stop: bool = True,
) -> AsyncGenerator[bytes, None]:
    """Replay a saved OpenAI-compatible chat.completion.chunk SSE log."""
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    raw_sse_lines = _extract_saved_raw_sse_lines(raw_sse_text)
    stream_id = f"chatcmpl-replay-{uuid.uuid4().hex[:18]}"
    created = int(time.time())
    full_response = ""
    saved_log = False
    sent_role_chunk = False
    finish_status = "unknown"
    line_count = 0
    data_line_count = 0
    first_data_time: float | None = None
    downstream_data_count = 0
    downstream_done_count = 0
    downstream_content_chars = 0
    downstream_tail = deque(maxlen=12)

    def persist_stream_log(marker: str = "", response_override=None, error_type: str | None = None):
        nonlocal saved_log
        if saved_log or messages is None:
            return

        lines = list(raw_sse_lines)
        if marker:
            lines.append(marker)

        response_text = response_override
        if response_text is None:
            response_text = full_response or "[empty replay response]"

        try:
            kwargs = {
                "stream": True,
                "raw_sse": "\n".join(lines),
                "request_payload": request_data,
            }
            if error_type:
                kwargs["error_type"] = error_type
            deps.save_request_log(model, messages, response_text, trace_id=trace_id, **kwargs)
            saved_log = True
        except Exception as save_err:
            deps.log(f"{trace_prefix}OpenAI replay log save failed: {save_err}")

    def append_downstream_data(data_obj: dict[str, Any]):
        nonlocal downstream_data_count, downstream_content_chars
        downstream_data_count += 1
        content = _extract_openai_content(data_obj)
        if content:
            downstream_content_chars += len(content)
            preview = content[:40].replace("\n", "\\n")
            downstream_tail.append(f"c={len(content)}:{preview}")
            return
        finish_reason = _extract_openai_finish_reason(data_obj)
        if finish_reason:
            downstream_tail.append(f"finish={finish_reason}|c=0")
        else:
            downstream_tail.append("data_meta")

    def emit_data(data_obj: dict[str, Any], ensure_ascii: bool = True) -> bytes:
        append_downstream_data(data_obj)
        return f"data: {json.dumps(data_obj, ensure_ascii=ensure_ascii)}\n\n".encode()

    def emit_done() -> bytes:
        nonlocal downstream_done_count
        downstream_done_count += 1
        downstream_tail.append("DONE")
        return b"data: [DONE]\n\n"

    def normalize_chunk(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal stream_id, created, model
        if data.get("id"):
            stream_id = str(data["id"])
        if isinstance(data.get("created"), int):
            created = data["created"]
        if data.get("model"):
            model = str(data["model"])
        if data.get("object") == "chat.completion.chunk":
            return data

        content = _extract_openai_message_content(data)
        finish_reason = _extract_openai_finish_reason(data)
        return _build_openai_chunk(
            stream_id=stream_id,
            created=created,
            model=model,
            delta=content,
            finish_reason=finish_reason,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
        )

    if caller_key:
        deps.log(f"{trace_prefix}openai_replay_stream_owner caller={caller_key} {caller_desc}")

    try:
        async for line in _iterate_saved_raw_sse_lines(raw_sse_text):
            line_count += 1
            if not line:
                continue
            if line.startswith("[FINISH_REASON:") or line.startswith("[STREAM_END:"):
                continue
            if not line.startswith("data: "):
                continue

            data_line_count += 1
            if first_data_time is None:
                first_data_time = time.perf_counter()
                deps.log(
                    f"{trace_prefix}openai_replay_first_data "
                    f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                )

            payload = line[6:].strip()
            if not payload:
                continue

            if payload == "[DONE]":
                if not sent_role_chunk:
                    yield emit_data(
                        _build_openai_chunk(
                            stream_id=stream_id,
                            created=created,
                            model=model,
                            role="assistant",
                        )
                    )
                    sent_role_chunk = True
                finish_status = "replay_done"
                persist_stream_log("[REPLAY_DONE]")
                yield emit_done()
                return

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(data.get("error"), dict):
                err = data["error"]
                err_msg = err.get("message") or json.dumps(err, ensure_ascii=False)
                finish_status = "replay_error_event"
                persist_stream_log(
                    "[REPLAY_ERROR_EVENT]",
                    response_override=f"[ERROR] {err_msg}",
                    error_type="openai_replay_error_event",
                )
                yield deps.build_openai_sse_error(
                    502,
                    err_msg[:4000],
                    error_type=err.get("type", "openai_replay_error"),
                )
                yield emit_done()
                return

            if data.get("type") == "error" and data.get("error") == "aborted":
                deps.log(f"{trace_prefix}openai_replay_aborted_event skipped")
                continue

            out = normalize_chunk(data)
            content = _extract_openai_content(out)
            finish_reason = _extract_openai_finish_reason(out)

            if not sent_role_chunk:
                if not (isinstance(out.get("choices"), list) and out["choices"] and out["choices"][0].get("delta", {}).get("role")):
                    yield emit_data(
                        _build_openai_chunk(
                            stream_id=stream_id,
                            created=created,
                            model=model,
                            role="assistant",
                        )
                    )
                sent_role_chunk = True

            if content:
                full_response += content
                if enable_early_stop and deps.has_stop_tag(full_response):
                    full_response, clean_delta = _truncate_for_stop_tag(
                        full_response,
                        content,
                        deps.find_stop_tag,
                    )
                    if clean_delta:
                        out = json.loads(json.dumps(out))
                        out["choices"][0].setdefault("delta", {})["content"] = clean_delta
                        out["choices"][0]["finish_reason"] = None
                        yield emit_data(out, ensure_ascii=False)
                    stop_chunk = _build_openai_chunk(
                        stream_id=stream_id,
                        created=created,
                        model=model,
                        finish_reason="stop",
                    )
                    finish_status = "replay_early_stop_tag"
                    persist_stream_log("[REPLAY_EARLY_STOP_TAG]")
                    yield emit_data(stop_chunk)
                    yield emit_done()
                    return

            if finish_reason:
                finish_status = f"replay_finish_reason:{finish_reason}"
                persist_stream_log(f"[REPLAY_FINISH_REASON: {finish_reason}]")
                yield emit_data(out, ensure_ascii=True)
                yield emit_done()
                return

            yield emit_data(out, ensure_ascii=False)

        finish_status = "replay_incomplete_stream_end"
        persist_stream_log(
            "[REPLAY_INCOMPLETE]",
            error_type="openai_replay_incomplete",
        )
        yield deps.build_openai_sse_error(
            502,
            "OpenAI replay stream closed without finish_reason or [DONE]",
            "openai_replay_incomplete",
        )
        yield emit_done()
    except asyncio.CancelledError:
        finish_status = "replay_downstream_cancelled"
        persist_stream_log(
            "[CANCELLED: downstream client disconnected during OpenAI replay]",
            error_type="downstream_cancelled",
        )
        raise
    except Exception as e:
        finish_status = f"replay_exception:{type(e).__name__}"
        persist_stream_log(
            f"[EXCEPTION] {e}",
            response_override=full_response + f"\n[ERROR] {e}" if full_response else f"[ERROR] {e}",
            error_type="openai_replay_proxy_error",
        )
        yield deps.build_openai_sse_error(502, str(e), "openai_replay_proxy_error")
        yield emit_done()
    finally:
        if caller_key:
            deps.release_caller(caller_key, trace_id)
        if not saved_log:
            persist_stream_log("[FINALIZER: replay closed before completion]")
            if finish_status == "unknown":
                finish_status = "finalizer_saved_partial"
        deps.log(
            f"{trace_prefix}openai_replay_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} out_chars={len(full_response)} "
            f"down_data_lines={downstream_data_count} down_out_chars={downstream_content_chars} "
            f"down_done={downstream_done_count} down_tail={list(downstream_tail)[-4:]}"
        )


async def forward_chat_completions_stream(
    *,
    url: str,
    request_data: dict,
    headers: dict,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: ChatCompletionsUpstreamDeps,
    enable_early_stop: bool = True,
    model: str = "",
    messages: list | None = None,
    trace_id: str = "",
    caller_key: str = "",
    caller_desc: str = "",
    supersede_event: asyncio.Event | None = None,
) -> AsyncGenerator[bytes, None]:
    """OpenAI-compatible streaming forwarder with downstream keepalive.

    Important: model-directory/pro upstreams can take tens of seconds before
    headers or visible content. We therefore emit OpenAI-compatible no-op chunks
    to the downstream client while waiting for upstream headers and while waiting
    between upstream SSE lines. This mirrors the Claude Code keepalive behavior
    and prevents SillyTavern/client-side timeout aborts without polluting text.
    """
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    enforce_glm52_official_thinking(request_data, log=deps.log, trace_prefix=trace_prefix)
    content_buffer = ""
    full_response = ""
    raw_sse_lines: list[str] = []
    raw_sse_size = 0
    raw_sse_truncated = False
    saved_log = False
    upstream_aborted_event = False
    saw_finish_reason = False
    saw_done = False
    last_chunk_id = f"chatcmpl-keepalive-{uuid.uuid4().hex[:18]}"
    last_chunk_created = int(time.time())
    last_chunk_model = model or "unknown"
    finish_status = "unknown"
    line_count = 0
    data_line_count = 0
    first_data_time: float | None = None
    downstream_line_count = 0
    downstream_data_count = 0
    downstream_done_count = 0
    downstream_content_chars = 0
    downstream_tail = deque(maxlen=12)
    keepalive_interval = 10.0
    last_downstream_emit = time.perf_counter()
    keepalive_count = 0
    terminal_detector = OpenAIChatTerminalDetector()
    pending_tool_lines: list[str] = []

    def persist_stream_log(marker: str = "", response_override=None):
        nonlocal saved_log
        if saved_log or messages is None:
            return

        if marker:
            raw_sse_lines.append(marker)

        response_text = response_override
        if response_text is None:
            response_text = full_response or "[empty response]"

        try:
            deps.save_request_log(
                model,
                messages,
                response_text,
                stream=True,
                raw_sse="\n".join(raw_sse_lines),
                request_payload=request_data,
                trace_id=trace_id,
            )
            saved_log = True
        except Exception as save_err:
            deps.log(f"Failed to save stream log: {save_err}")

    def append_downstream_line(line: str):
        nonlocal downstream_line_count, downstream_data_count
        nonlocal downstream_done_count, downstream_content_chars
        downstream_line_count += 1
        if not line:
            return

        if line.startswith("data: "):
            downstream_data_count += 1
            payload = line[6:].strip()
            if payload == "[DONE]":
                downstream_done_count += 1
                downstream_tail.append("DONE")
                return
            try:
                data = json.loads(payload)
                content = _extract_openai_content(data)
                if isinstance(content, str) and content:
                    downstream_content_chars += len(content)
                finish_reason = None
                if "choices" in data and data["choices"]:
                    finish_reason = data["choices"][0].get("finish_reason")
                if finish_reason:
                    downstream_tail.append(
                        f"finish={finish_reason}|c={len(content) if isinstance(content, str) else 0}"
                    )
                elif isinstance(content, str) and content:
                    preview = content[:40].replace("\n", "\\n")
                    downstream_tail.append(f"c={len(content)}:{preview}")
                else:
                    downstream_tail.append("data_meta")
            except Exception:
                downstream_tail.append(f"data_nonjson:{payload[:60]}")
            return

        downstream_tail.append(f"line:{line[:60]}")

    def emit_line(line: str) -> bytes:
        nonlocal last_downstream_emit
        append_downstream_line(line)
        last_downstream_emit = time.perf_counter()
        return f"{line}\n".encode()

    def emit_data(data_obj: dict, ensure_ascii: bool = True) -> bytes:
        nonlocal last_downstream_emit
        line = f"data: {json.dumps(data_obj, ensure_ascii=ensure_ascii)}"
        append_downstream_line(line)
        last_downstream_emit = time.perf_counter()
        return f"{line}\n\n".encode()

    def emit_done() -> bytes:
        nonlocal last_downstream_emit
        append_downstream_line("data: [DONE]")
        last_downstream_emit = time.perf_counter()
        return b"data: [DONE]\n\n"

    def make_keepalive_chunk() -> dict[str, Any]:
        return _build_openai_chunk(
            stream_id=last_chunk_id,
            created=last_chunk_created,
            model=last_chunk_model,
        )

    def emit_keepalive(reason: str) -> bytes | None:
        nonlocal last_downstream_emit, keepalive_count
        now = time.perf_counter()
        if now - last_downstream_emit < keepalive_interval:
            return None
        idle_ms = (now - last_downstream_emit) * 1000
        last_downstream_emit = now
        keepalive_count += 1
        deps.log(
            f"{trace_prefix}openai_downstream_keepalive "
            f"count={keepalive_count} reason={reason} idle={idle_ms:.1f}ms "
            f"out_chars={len(full_response)}"
        )
        return emit_data(make_keepalive_chunk(), ensure_ascii=True)

    if caller_key:
        deps.log(f"{trace_prefix}stream_owner caller={caller_key} {caller_desc}")

    upstream_task: asyncio.Task | None = None

    class StreamSuperseded(Exception):
        pass

    def raise_if_superseded() -> None:
        if supersede_event is not None and supersede_event.is_set():
            raise StreamSuperseded

    async def open_upstream() -> tuple[httpx.AsyncClient, Any, httpx.Response]:
        client = httpx.AsyncClient(timeout=timeout)
        try:
            deps.log(
                f"{trace_prefix}upstream_request_start "
                f"elapsed={deps.fmt_ms(request_t0)} model={model or '-'}"
            )
            upstream_t0 = time.perf_counter()
            cm = client.stream("POST", url, json=request_data, headers=headers)
            response = await cm.__aenter__()
            deps.log(
                f"{trace_prefix}upstream_headers status={response.status_code} "
                f"{format_account_pool_route(response)} "
                f"elapsed={deps.fmt_ms(upstream_t0)}"
            )
            return client, cm, response
        except Exception:
            await client.aclose()
            raise

    try:
        raise_if_superseded()
        # Start the downstream stream immediately, before upstream headers arrive.
        # Empty delta is a no-op for visible text but proves liveness to the client.
        yield emit_data(make_keepalive_chunk(), ensure_ascii=True)

        upstream_task = asyncio.create_task(open_upstream())
        while not upstream_task.done():
            raise_if_superseded()
            try:
                client, cm, response = await asyncio.wait_for(
                    asyncio.shield(upstream_task),
                    timeout=1.0 if supersede_event is not None else keepalive_interval,
                )
                break
            except asyncio.TimeoutError:
                raise_if_superseded()
                keepalive = emit_keepalive("waiting_headers")
                if keepalive is not None:
                    yield keepalive
        else:
            client, cm, response = await upstream_task

        deps = scope_stop_detection(deps, response, url, trace_prefix)

        try:
            if is_himodels_upstream(response, url):
                enable_early_stop = True
                deps.log(f"{trace_prefix}himodels_early_stop=runtime_config")
            raise_if_superseded()
            if response.status_code != 200:
                content = await response.aread()
                error_text = content.decode(errors="replace")
                nonlocal_marker = f"[HTTP {response.status_code}]"
                raw_sse_size, raw_sse_truncated = _append_raw_sse_line(
                    raw_sse_lines,
                    nonlocal_marker,
                    raw_size=raw_sse_size,
                    raw_truncated=raw_sse_truncated,
                    max_raw_sse_bytes=max_raw_sse_bytes,
                )
                raw_sse_size, raw_sse_truncated = _append_raw_sse_line(
                    raw_sse_lines,
                    error_text,
                    raw_size=raw_sse_size,
                    raw_truncated=raw_sse_truncated,
                    max_raw_sse_bytes=max_raw_sse_bytes,
                )
                persist_stream_log("[UPSTREAM_HTTP_ERROR]")
                finish_status = f"upstream_http_{response.status_code}"
                err_bytes = deps.build_openai_sse_error(
                    response.status_code,
                    error_text[:4000],
                    error_type="upstream_http_error",
                )
                for out_line in err_bytes.decode(errors="replace").splitlines():
                    append_downstream_line(out_line)
                last_downstream_emit = time.perf_counter()
                yield err_bytes
                yield emit_done()
                return

            line_queue: asyncio.Queue = asyncio.Queue()

            async def read_upstream_lines() -> None:
                try:
                    async for upstream_line in response.aiter_lines():
                        await line_queue.put(upstream_line)
                except Exception as upstream_error:
                    await line_queue.put(upstream_error)
                finally:
                    await line_queue.put(None)

            reader_task = asyncio.create_task(read_upstream_lines())
            try:
                while True:
                    raise_if_superseded()
                    try:
                        queued_line = await asyncio.wait_for(
                            line_queue.get(),
                            timeout=1.0 if supersede_event is not None else keepalive_interval,
                        )
                    except asyncio.TimeoutError:
                        raise_if_superseded()
                        keepalive = emit_keepalive("waiting_line")
                        if keepalive is not None:
                            yield keepalive
                        continue

                    keepalive = emit_keepalive("before_line")
                    if keepalive is not None:
                        yield keepalive

                    if queued_line is None:
                        break
                    if isinstance(queued_line, Exception):
                        raise queued_line

                    line = queued_line
                    line_count += 1
                    raw_sse_size, raw_sse_truncated = _append_raw_sse_line(
                        raw_sse_lines,
                        line,
                        raw_size=raw_sse_size,
                        raw_truncated=raw_sse_truncated,
                        max_raw_sse_bytes=max_raw_sse_bytes,
                    )
                    if not line:
                        if pending_tool_lines:
                            pending_tool_lines.append(line)
                            continue
                        append_downstream_line("")
                        last_downstream_emit = time.perf_counter()
                        yield b"\n"
                        continue

                    if line.startswith("data: "):
                        data_line_count += 1
                        if first_data_time is None:
                            first_data_time = time.perf_counter()
                            deps.log(
                                f"{trace_prefix}upstream_first_data "
                                f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                            )

                        json_str = line[6:].strip()
                        if json_str == "[DONE]":
                            for pending_line in pending_tool_lines:
                                yield emit_line(pending_line)
                            pending_tool_lines.clear()
                            saw_done = True
                            finish_status = "upstream_done_local_finalized"
                            persist_stream_log("[UPSTREAM_DONE_LOCAL_FINALIZED]")
                            stop_chunk = _build_openai_chunk(
                                stream_id=last_chunk_id,
                                created=last_chunk_created,
                                model=last_chunk_model,
                                finish_reason="stop",
                            )
                            yield emit_data(stop_chunk)
                            yield emit_done()
                            return

                        if json_str:
                            try:
                                data = json.loads(json_str)

                                if isinstance(data.get("error"), dict):
                                    err = data["error"]
                                    err_msg = err.get("message") or json.dumps(err, ensure_ascii=False)
                                    finish_status = "upstream_error_event"
                                    err_bytes = deps.build_openai_sse_error(
                                        502,
                                        err_msg[:4000],
                                        error_type=err.get("type", "upstream_error"),
                                    )
                                    for out_line in err_bytes.decode(errors="replace").splitlines():
                                        append_downstream_line(out_line)
                                    last_downstream_emit = time.perf_counter()
                                    yield err_bytes
                                    yield emit_done()
                                    persist_stream_log(
                                        "[UPSTREAM_ERROR_EVENT]",
                                        response_override=f"[ERROR] {err_msg}",
                                    )
                                    return

                                if data.get("type") == "error" and data.get("error") == "aborted":
                                    upstream_aborted_event = True
                                    deps.log(f"{trace_prefix}upstream_aborted_event swallowed")
                                    continue

                                if data.get("id"):
                                    last_chunk_id = data.get("id")
                                if isinstance(data.get("created"), int):
                                    last_chunk_created = data.get("created")
                                if data.get("model"):
                                    last_chunk_model = data.get("model")

                                tool_decision = terminal_detector.feed(data)
                                if tool_decision.has_tool_data:
                                    pending_tool_lines.append(line)
                                    if tool_decision.terminal:
                                        terminal_remainder = json.loads(json.dumps(data))
                                        remainder_present = False
                                        for remainder_choice in terminal_remainder.get("choices", []) or []:
                                            if not isinstance(remainder_choice, dict):
                                                continue
                                            remainder_choice["finish_reason"] = None
                                            remainder_delta = remainder_choice.get("delta")
                                            if isinstance(remainder_delta, dict):
                                                remainder_delta.pop("tool_calls", None)
                                                remainder_delta.pop("function_call", None)
                                                remainder_present = remainder_present or bool(remainder_delta)
                                        if remainder_present:
                                            remainder_content = _extract_openai_content(terminal_remainder)
                                            if remainder_content:
                                                content_buffer += remainder_content
                                                full_response += remainder_content
                                            yield emit_data(terminal_remainder, ensure_ascii=False)
                                        finish_status = "terminal_tool"
                                        await _close_upstream_stream(
                                            response=response,
                                            client=client,
                                            deps=deps,
                                            trace_prefix=trace_prefix,
                                            label="terminal_tool",
                                            reason="openai_chat_terminal_tool",
                                            started_at=request_t0,
                                            line_count=line_count,
                                            data_line_count=data_line_count,
                                            out_chars=len(full_response),
                                            reader_task=reader_task,
                                        )
                                        stop_chunk = _build_openai_chunk(
                                            stream_id=last_chunk_id,
                                            created=last_chunk_created,
                                            model=last_chunk_model,
                                            finish_reason="stop",
                                        )
                                        yield emit_data(stop_chunk)
                                        yield emit_done()
                                        persist_stream_log("[TERMINAL_TOOL by proxy]")
                                        return
                                    if tool_decision.defer:
                                        continue
                                    for pending_line in pending_tool_lines:
                                        yield emit_line(pending_line)
                                    pending_tool_lines.clear()
                                    continue
                                if pending_tool_lines:
                                    for pending_line in pending_tool_lines:
                                        yield emit_line(pending_line)
                                    pending_tool_lines.clear()

                                content = _extract_openai_content(data)

                                if content:
                                    content_buffer += content
                                    full_response += content

                                    if enable_early_stop and deps.has_stop_tag(content_buffer):
                                        deps.log("Stream: detected STOP_TAG")
                                        prefix_pos = deps.find_stop_tag(content_buffer)
                                        prev_len = len(content_buffer) - len(content)

                                        if prefix_pos >= prev_len:
                                            keep_len = prefix_pos - prev_len
                                            clean = content[:keep_len]
                                        else:
                                            clean = ""

                                        content_buffer = content_buffer[:prefix_pos]
                                        full_response = content_buffer
                                        if clean:
                                            data["choices"][0]["delta"]["content"] = clean
                                            yield emit_data(data, ensure_ascii=True)

                                        finish_status = "early_stop_tag"
                                        await _close_upstream_stream(
                                            response=response,
                                            client=client,
                                            deps=deps,
                                            trace_prefix=trace_prefix,
                                            label="early_stop",
                                            reason="openai_stream_stop_tag",
                                            started_at=request_t0,
                                            line_count=line_count,
                                            data_line_count=data_line_count,
                                            out_chars=len(full_response),
                                            reader_task=reader_task,
                                        )
                                        stop_chunk = _build_openai_chunk(
                                            stream_id=last_chunk_id,
                                            created=last_chunk_created,
                                            model=last_chunk_model,
                                            finish_reason="stop",
                                        )
                                        yield emit_data(stop_chunk)
                                        yield emit_done()
                                        persist_stream_log("[EARLY_STOP by proxy]")
                                        return

                                if "choices" in data and data["choices"]:
                                    finish_reason = data["choices"][0].get("finish_reason")
                                    if finish_reason:
                                        saw_finish_reason = True
                                        finish_status = f"finish_reason_local:{finish_reason}"
                                        choice = data["choices"][0]
                                        delta = choice.get("delta")
                                        if isinstance(delta, dict) and delta:
                                            content_chunk = json.loads(json.dumps(data))
                                            content_chunk["choices"][0]["finish_reason"] = None
                                            yield emit_data(content_chunk, ensure_ascii=False)
                                        stop_chunk = _build_openai_chunk(
                                            stream_id=last_chunk_id,
                                            created=last_chunk_created,
                                            model=last_chunk_model,
                                            finish_reason=str(finish_reason),
                                            usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
                                        )
                                        yield emit_data(stop_chunk)
                                        yield emit_done()
                                        deps.log(
                                            f"{trace_prefix}finish_reason_local_emit reason={finish_reason} "
                                            f"up_data_lines={data_line_count} up_out_chars={len(full_response)} "
                                            f"down_data_lines={downstream_data_count} "
                                            f"down_out_chars={downstream_content_chars} "
                                            f"down_done={downstream_done_count} "
                                            f"tail={list(downstream_tail)[-4:]}"
                                        )
                                        persist_stream_log(f"[FINISH_REASON: {finish_reason}]")
                                        return

                            except json.JSONDecodeError:
                                pass

                    yield emit_line(line)

                for pending_line in pending_tool_lines:
                    yield emit_line(pending_line)
                pending_tool_lines.clear()
                persist_stream_log("[STREAM_END: no finish_reason detected]")
                if (upstream_aborted_event or full_response) and not saw_finish_reason and not saw_done:
                    stop_chunk = _build_openai_chunk(
                        stream_id=last_chunk_id,
                        created=last_chunk_created,
                        model=last_chunk_model,
                        finish_reason="stop",
                    )
                    yield emit_data(stop_chunk)
                    yield emit_done()
                    finish_status = "stream_end_without_finish_reason"
                else:
                    finish_status = (
                        "upstream_aborted_stream_end"
                        if upstream_aborted_event else "stream_end_no_finish_reason"
                    )
                    yield deps.build_openai_sse_error(
                        502,
                        "OpenAI upstream stream closed without a terminal event",
                        "upstream_stream_incomplete",
                    )
                    yield emit_done()
                deps.log(f"Stream ended without finish_reason, response_len={len(full_response)}")
            finally:
                if not reader_task.done():
                    reader_task.cancel()
                    try:
                        await reader_task
                    except asyncio.CancelledError:
                        pass
        finally:
            try:
                await cm.__aexit__(None, None, None)
            finally:
                await client.aclose()

    except StreamSuperseded:
        if upstream_task is not None and not upstream_task.done():
            upstream_task.cancel()
            with suppress(asyncio.CancelledError):
                await upstream_task
        deps.log(
            f"{trace_prefix}local_stream_superseded model={model or '-'} "
            f"caller={caller_key or '-'}"
        )
        persist_stream_log("[SUPERSEDED: newer local stream replaced this request]")
        finish_status = "local_stream_superseded"
        return
    except asyncio.CancelledError:
        if upstream_task is not None and not upstream_task.done():
            upstream_task.cancel()
            try:
                await upstream_task
            except asyncio.CancelledError:
                pass
        deps.log(
            "Stream forward cancelled by downstream client "
            f"(down_data_lines={downstream_data_count}, down_out_chars={downstream_content_chars}, "
            f"down_done={downstream_done_count}, keepalives={keepalive_count}, "
            f"tail={list(downstream_tail)[-4:]}, caller={caller_key or '-'})"
        )
        persist_stream_log("[CANCELLED: downstream client disconnected]")
        finish_status = "downstream_cancelled"
        raise
    except Exception as e:
        deps.log(f"Stream forward error: {e}")
        error_response = full_response + f"\n[ERROR: {e}]" if full_response else f"[ERROR: {e}]"
        persist_stream_log(f"[EXCEPTION: {e}]", response_override=error_response)
        finish_status = f"exception:{type(e).__name__}"
        err_bytes = deps.build_openai_sse_error(
            502,
            str(e),
            error_type="proxy_stream_error",
        )
        for out_line in err_bytes.decode(errors="replace").splitlines():
            append_downstream_line(out_line)
        last_downstream_emit = time.perf_counter()
        yield err_bytes
        yield emit_done()
    finally:
        if caller_key:
            deps.release_caller(caller_key, trace_id)
        if not saved_log:
            persist_stream_log("[FINALIZER: stream closed before completion]")
            if finish_status == "unknown":
                finish_status = "finalizer_saved_partial"
        deps.log(
            f"{trace_prefix}forward_stream_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} out_chars={len(full_response)} "
            f"down_lines={downstream_line_count} down_data_lines={downstream_data_count} "
            f"down_out_chars={downstream_content_chars} down_done={downstream_done_count} "
            f"keepalives={keepalive_count} down_tail={list(downstream_tail)[-4:]}"
        )


async def collect_chat_completions_nonstream(
    *,
    url: str,
    request_data: dict,
    headers: dict,
    timeout: httpx.Timeout,
    deps: ChatCompletionsUpstreamDeps,
    trace_id: str = "",
) -> tuple[str, str, dict, str, str, dict[str, Any]]:
    """Collect a plain OpenAI-compatible non-stream chat completion."""
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    request_t0 = time.perf_counter()
    enforce_glm52_official_thinking(request_data, log=deps.log, trace_prefix=trace_prefix)
    payload = request_data.copy()
    payload["stream"] = False

    async with httpx.AsyncClient(timeout=timeout) as client:
        deps.log(f"{trace_prefix}nonstream_request_start elapsed={deps.fmt_ms(request_t0)}")
        upstream_t0 = time.perf_counter()
        response = await client.post(url, json=payload, headers=headers)
        deps.log(
            f"{trace_prefix}nonstream_upstream_response status={response.status_code} "
            f"{format_account_pool_route(response)} "
            f"elapsed={deps.fmt_ms(upstream_t0)}"
        )
        raw_text = response.text
        if response.status_code != 200:
            raise RuntimeError(f"Upstream error: {response.status_code} - {raw_text}")
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Upstream returned non-JSON response: {raw_text[:4000]}") from exc

    full_content = _extract_openai_message_content(data)
    model_name = str(data.get("model") or "") if isinstance(data, dict) else ""
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    finish_reason = _extract_openai_finish_reason(data) or "stop"
    deps.log(
        f"{trace_prefix}nonstream_collect_done elapsed={deps.fmt_ms(request_t0)} "
        f"out_chars={len(full_content)}"
    )
    return full_content, model_name, usage, finish_reason, raw_text, data


async def replay_chat_completions_nonstream_as_stream(
    *,
    url: str,
    request_data: dict,
    headers: dict,
    timeout: httpx.Timeout,
    deps: ChatCompletionsUpstreamDeps,
    model: str = "",
    messages: list | None = None,
    trace_id: str = "",
    caller_key: str = "",
    caller_desc: str = "",
    chunk_size: int = 1200,
) -> AsyncGenerator[bytes, None]:
    """Call an OpenAI-compatible upstream as true non-stream, replay as SSE.

    This is for pro Gemini only: SillyTavern uses stream=true downstream, while
    the upstream receives stream=false. While waiting for the upstream JSON body
    we emit empty OpenAI chunks as keepalive; after the body arrives we split
    message.content into normal delta chunks and finish with [DONE].
    """
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    enforce_glm52_official_thinking(request_data, log=deps.log, trace_prefix=trace_prefix)
    payload = request_data.copy()
    payload["stream"] = False
    stream_id = f"chatcmpl-nonstream-replay-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    out_model = model or str(payload.get("model") or "unknown")
    keepalive_interval = 10.0
    keepalive_count = 0
    saved_log = False
    finish_status = "unknown"
    full_content = ""
    raw_text = ""
    usage: dict[str, Any] = {}

    def persist_log(marker: str = "", response_override: str | None = None):
        nonlocal saved_log
        if saved_log or messages is None:
            return
        response_text = response_override if response_override is not None else (full_content or "[empty response]")
        raw_for_log = raw_text
        if marker:
            raw_for_log = f"{raw_for_log}\n{marker}" if raw_for_log else marker
        try:
            deps.save_request_log(
                model or str(payload.get("model") or ""),
                messages,
                response_text,
                stream=True,
                raw_sse=raw_for_log,
                request_payload=payload,
                trace_id=trace_id,
            )
            saved_log = True
        except Exception as save_err:
            deps.log(f"{trace_prefix}nonstream_replay_log_error err={save_err}")

    def make_chunk(
        *,
        delta: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        usage_obj: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": out_model,
            "system_fingerprint": None,
            "choices": [{
                "delta": delta or {},
                "logprobs": None,
                "finish_reason": finish_reason,
                "index": 0,
            }],
            "usage": usage_obj,
        }

    def emit_data(obj: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

    def emit_done() -> bytes:
        return b"data: [DONE]\n\n"

    async def call_upstream() -> tuple[str, dict[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            deps.log(f"{trace_prefix}nonstream_replay_upstream_request_start elapsed={deps.fmt_ms(request_t0)}")
            upstream_t0 = time.perf_counter()
            response = await client.post(url, json=payload, headers=headers)
            deps.log(
                f"{trace_prefix}nonstream_replay_upstream_response "
                f"status={response.status_code} {format_account_pool_route(response)} "
                f"elapsed={deps.fmt_ms(upstream_t0)}"
            )
            text = response.text
            if response.status_code != 200:
                raise RuntimeError(f"Upstream error: {response.status_code} - {text}")
            try:
                data = response.json()
            except Exception as exc:
                raise RuntimeError(f"Upstream returned non-JSON response: {text[:4000]}") from exc
            return text, data

    if caller_key:
        deps.log(f"{trace_prefix}nonstream_replay_stream_owner caller={caller_key} {caller_desc}")

    upstream_task: asyncio.Task | None = None
    try:
        # Start the downstream stream immediately; role-only chunk is valid and
        # visible-text empty, so it acts as a liveness proof for SillyTavern.
        yield emit_data(make_chunk(delta={"role": "assistant"}))

        upstream_task = asyncio.create_task(call_upstream())
        while not upstream_task.done():
            try:
                raw_text, data = await asyncio.wait_for(
                    asyncio.shield(upstream_task),
                    timeout=keepalive_interval,
                )
                break
            except asyncio.TimeoutError:
                keepalive_count += 1
                deps.log(
                    f"{trace_prefix}nonstream_replay_keepalive "
                    f"count={keepalive_count} elapsed={deps.fmt_ms(request_t0)}"
                )
                yield emit_data(make_chunk())
        else:
            raw_text, data = await upstream_task

        if isinstance(data.get("model"), str) and data.get("model"):
            out_model = data["model"]
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        finish_reason = _extract_openai_finish_reason(data) or "stop"
        reasoning = _extract_openai_message_reasoning(data)
        full_content = _extract_openai_message_content(data)
        choices = data.get("choices") if isinstance(data, dict) else None
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        raw_tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
        ordinary_tool_calls: list[dict[str, Any]] = []
        terminal_tool_seen = False
        for tool_call in raw_tool_calls or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if name == TERMINAL_TOOL_NAME:
                terminal_tool_seen = True
            else:
                ordinary_tool_calls.append(tool_call)

        if reasoning:
            yield emit_data(make_chunk(delta={"reasoning_content": reasoning}))

        if full_content:
            step = max(1, int(chunk_size or 1200))
            for idx in range(0, len(full_content), step):
                yield emit_data(make_chunk(delta={"content": full_content[idx:idx + step]}))

        if ordinary_tool_calls:
            yield emit_data(make_chunk(delta={"tool_calls": ordinary_tool_calls}))
            finish_reason = "tool_calls"
        elif terminal_tool_seen:
            finish_reason = "stop"
            finish_status = "terminal_tool_nonstream_replay"

        yield emit_data(make_chunk(finish_reason=finish_reason, usage_obj=usage or None))
        yield emit_done()
        if finish_status != "terminal_tool_nonstream_replay":
            finish_status = f"finish_reason:{finish_reason}"
        persist_log(f"[NONSTREAM_REPLAY_FINISH_REASON: {finish_reason}]")
    except asyncio.CancelledError:
        if upstream_task is not None and not upstream_task.done():
            upstream_task.cancel()
        finish_status = "downstream_cancelled"
        deps.log(f"{trace_prefix}nonstream_replay_cancelled keepalives={keepalive_count} out_chars={len(full_content)}")
        persist_log("[CANCELLED: downstream client disconnected]")
        raise
    except Exception as e:
        finish_status = f"exception:{type(e).__name__}"
        deps.log(f"{trace_prefix}nonstream_replay_error err={e}")
        persist_log(f"[EXCEPTION: {e}]", response_override=full_content + f"\n[ERROR: {e}]" if full_content else f"[ERROR: {e}]")
        yield deps.build_openai_sse_error(502, str(e), error_type="proxy_nonstream_replay_error")
        yield emit_done()
    finally:
        if caller_key:
            deps.release_caller(caller_key, trace_id)
        if not saved_log:
            persist_log("[FINALIZER: nonstream replay closed before completion]")
        deps.log(
            f"{trace_prefix}nonstream_replay_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} out_chars={len(full_content)} "
            f"keepalives={keepalive_count}"
        )


async def collect_chat_completions_stream(
    *,
    url: str,
    request_data: dict,
    headers: dict,
    timeout: httpx.Timeout,
    max_raw_sse_bytes: int,
    deps: ChatCompletionsUpstreamDeps,
    enable_early_stop: bool = True,
    trace_id: str = "",
) -> tuple[str, str, dict, str, str]:
    """收集 OpenAI/NewAPI 流式响应，供非流请求复用。"""
    full_content = ""
    model_name = ""
    usage = {}
    finish_reason = "stop"
    raw_sse_lines: list[str] = []
    raw_sse_size = 0
    raw_sse_truncated = False
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    request_t0 = time.perf_counter()
    line_count = 0
    data_line_count = 0
    first_data_time: float | None = None

    enforce_glm52_official_thinking(request_data, log=deps.log, trace_prefix=trace_prefix)
    request_data = request_data.copy()
    request_data["stream"] = True

    async with httpx.AsyncClient(timeout=timeout) as client:
        deps.log(f"{trace_prefix}collect_request_start elapsed={deps.fmt_ms(request_t0)}")
        upstream_t0 = time.perf_counter()
        async with client.stream("POST", url, json=request_data, headers=headers) as response:
            deps = scope_stop_detection(deps, response, url, trace_prefix)
            deps.log(
                f"{trace_prefix}collect_upstream_headers status={response.status_code} "
                f"{format_account_pool_route(response)} "
                f"elapsed={deps.fmt_ms(upstream_t0)}"
            )
            if is_himodels_upstream(response, url):
                enable_early_stop = True
                deps.log(f"{trace_prefix}himodels_early_stop=runtime_config")
            if response.status_code != 200:
                content = await response.aread()
                error_text = content.decode(errors="replace")
                raw_sse_size, raw_sse_truncated = _append_raw_sse_line(
                    raw_sse_lines,
                    f"[HTTP {response.status_code}]",
                    raw_size=raw_sse_size,
                    raw_truncated=raw_sse_truncated,
                    max_raw_sse_bytes=max_raw_sse_bytes,
                )
                raw_sse_size, raw_sse_truncated = _append_raw_sse_line(
                    raw_sse_lines,
                    error_text,
                    raw_size=raw_sse_size,
                    raw_truncated=raw_sse_truncated,
                    max_raw_sse_bytes=max_raw_sse_bytes,
                )
                raise RuntimeError(f"Upstream error: {response.status_code} - {error_text}")

            async for line in response.aiter_lines():
                line_count += 1
                raw_sse_size, raw_sse_truncated = _append_raw_sse_line(
                    raw_sse_lines,
                    line,
                    raw_size=raw_sse_size,
                    raw_truncated=raw_sse_truncated,
                    max_raw_sse_bytes=max_raw_sse_bytes,
                )
                if not line:
                    continue

                payload = line
                if payload.startswith("data: "):
                    data_line_count += 1
                    if first_data_time is None:
                        first_data_time = time.perf_counter()
                        deps.log(
                            f"{trace_prefix}collect_first_data "
                            f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                        )
                    payload = payload[6:]

                if payload == "[DONE]":
                    break

                try:
                    data = json.loads(payload)

                    if isinstance(data.get("error"), dict):
                        err = data["error"]
                        err_msg = err.get("message") or json.dumps(err, ensure_ascii=False)
                        raise RuntimeError(f"Upstream error event: {err_msg}")

                    if data.get("type") == "error" and data.get("error") == "aborted":
                        continue

                    if data.get("object") == "chat.completion.chunk" and isinstance(data.get("usage"), dict):
                        usage = data["usage"]
                        continue

                    if "model" in data:
                        model_name = data["model"]

                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]

                        delta = choice.get("delta", {})
                        delta_content = (
                            _coerce_openai_text_content(delta.get("content"))
                            if isinstance(delta, dict)
                            else ""
                        )
                        if delta_content:
                            full_content += delta_content
                            if enable_early_stop and deps.has_stop_tag(full_content):
                                deps.log("Collect: detected STOP_TAG")
                                full_content = full_content[:deps.find_stop_tag(full_content)]
                                finish_reason = "stop"
                                await _close_upstream_stream(
                                    response=response,
                                    client=client,
                                    deps=deps,
                                    trace_prefix=trace_prefix,
                                    label="early_stop",
                                    reason="openai_collect_delta_stop_tag",
                                    started_at=request_t0,
                                    line_count=line_count,
                                    data_line_count=data_line_count,
                                    out_chars=len(full_content),
                                )
                                break

                        message = choice.get("message", {})
                        message_content = (
                            _coerce_openai_text_content(message.get("content"))
                            if isinstance(message, dict)
                            else ""
                        )
                        if message_content:
                            full_content += message_content
                            if enable_early_stop and deps.has_stop_tag(full_content):
                                full_content = full_content[:deps.find_stop_tag(full_content)]
                                finish_reason = "stop"
                                await _close_upstream_stream(
                                    response=response,
                                    client=client,
                                    deps=deps,
                                    trace_prefix=trace_prefix,
                                    label="early_stop",
                                    reason="openai_collect_message_stop_tag",
                                    started_at=request_t0,
                                    line_count=line_count,
                                    data_line_count=data_line_count,
                                    out_chars=len(full_content),
                                )
                                break

                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                            break

                    if "usage" in data:
                        usage = data["usage"]

                except json.JSONDecodeError:
                    continue

    deps.log(
        f"{trace_prefix}collect_done elapsed={deps.fmt_ms(request_t0)} "
        f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
        f"lines={line_count} data_lines={data_line_count} out_chars={len(full_content)}"
    )
    return full_content, model_name, usage, finish_reason, "\n".join(raw_sse_lines)
