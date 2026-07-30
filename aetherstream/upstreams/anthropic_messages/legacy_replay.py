import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator

from .protocol import (
    _build_openai_chunk,
    _extract_anthropic_sse_text,
    _extract_saved_raw_sse_lines,
    _iterate_saved_raw_sse_lines,
    _map_stop_reason,
    _normalize_usage,
    _truncate_for_stop_tag,
)
from .types import AnthropicMessagesDeps


async def replay_anthropic_chat_stream(
    *,
    raw_sse_text: str,
    request_data: dict[str, Any],
    model: str,
    messages: list,
    trace_id: str,
    caller_key: str,
    caller_desc: str,
    deps: AnthropicMessagesDeps,
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
    deps: AnthropicMessagesDeps,
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
