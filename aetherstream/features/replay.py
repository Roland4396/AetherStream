"""Provider-independent saved-response replay for the chat completions API."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any


class ReplayPreparationError(RuntimeError):
    """Raised when replay is enabled but the selected artifact is unusable."""


@dataclass(frozen=True)
class ReplayDelta:
    content: str = ""
    reasoning_content: str = ""


@dataclass(frozen=True)
class ReplayRecord:
    source_format: str
    model: str
    deltas: tuple[ReplayDelta, ...]
    text: str
    reasoning_content: str
    usage: dict[str, Any]
    finish_reason: str
    source_complete: bool
    source_error: str | None


@dataclass(frozen=True)
class PreparedReplay:
    spec: dict[str, Any]
    record: ReplayRecord


_ANTHROPIC_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}

_INCOMPLETE_MARKERS = (
    "CANCELLED",
    "GENERATOR_CLOSED",
    "INCOMPLETE",
    "EXCEPTION",
    "FINALIZED_WITHOUT_TERMINAL_EVENT",
    "FINALIZER:",
)

_COMPLETE_MARKERS = (
    "FINISH_REASON:",
    "NONSTREAM_REPLAY_FINISH_REASON:",
    "RESPONSES_NONSTREAM_REPLAY_COMPLETED",
    "CODEX_NONSTREAM_REPLAY_COMPLETED",
    "REPLAY_DONE",
    "EARLY_STOP",
)


def _strip_log_envelope(raw_text: str) -> list[str]:
    lines = raw_text.splitlines()
    if lines and lines[0].startswith("Time: "):
        for index, line in enumerate(lines[:10]):
            if line and set(line) == {"="}:
                return lines[index + 1 :]
    return lines


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _normalize_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    usage = dict(value)
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    usage.setdefault("prompt_tokens", prompt_tokens)
    usage.setdefault("completion_tokens", completion_tokens)
    usage.setdefault("total_tokens", total_tokens)
    return usage


def _merge_usage(current: dict[str, Any], value: Any) -> dict[str, Any]:
    incoming = _normalize_usage(value)
    return incoming or current


def _responses_completed_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "".join(parts)


def _marker_state(markers: list[str]) -> tuple[bool, bool, str | None]:
    upper = [marker.upper() for marker in markers]
    incomplete = any(token in marker for marker in upper for token in _INCOMPLETE_MARKERS)
    complete = any(token in marker for marker in upper for token in _COMPLETE_MARKERS)
    source_error = markers[-1] if incomplete and markers else None
    return complete, incomplete, source_error


def _parse_wire_objects(raw_text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool, list[str]]:
    lines = _strip_log_envelope(raw_text)
    events: list[dict[str, Any]] = []
    plain_lines: list[str] = []
    markers: list[str] = []
    saw_done = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            markers.append(stripped)
            continue
        if stripped.startswith("event:"):
            continue
        if stripped.startswith("data:"):
            payload = stripped[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
            continue
        plain_lines.append(line)

    plain_payload = None
    if plain_lines:
        try:
            parsed = json.loads("\n".join(plain_lines).strip())
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            plain_payload = parsed

    return events, plain_payload, saw_done, markers


def _detect_format(events: list[dict[str, Any]], plain_payload: dict[str, Any] | None) -> str:
    objects = [*events]
    if plain_payload is not None:
        objects.append(plain_payload)

    for item in objects:
        event_type = str(item.get("type") or "")
        if event_type in _ANTHROPIC_TYPES:
            return "anthropic_sse"
    for item in objects:
        event_type = str(item.get("type") or "")
        if event_type.startswith("response."):
            return "responses_sse"
    for item in objects:
        if isinstance(item.get("candidates"), list):
            return "gemini_sse"
    for item in objects:
        if isinstance(item.get("choices"), list):
            return "openai_json" if plain_payload is item and not events else "openai_sse"
    return "unknown"


def _parse_openai(
    objects: list[dict[str, Any]],
    *,
    source_format: str,
    fallback_model: str,
    saw_done: bool,
) -> ReplayRecord:
    model = fallback_model
    deltas: list[ReplayDelta] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = ""
    source_error = None
    saw_completion_object = False

    for data in objects:
        if isinstance(data.get("error"), dict):
            error = data["error"]
            source_error = str(error.get("message") or json.dumps(error, ensure_ascii=False))
        if data.get("model"):
            model = str(data["model"])
        usage = _merge_usage(usage, data.get("usage"))
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue

        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = _extract_text(delta.get("content"))
            reasoning = ""
            for key in ("reasoning_content", "reasoning"):
                value = delta.get(key)
                if isinstance(value, str):
                    reasoning = value
                    break
            if content or reasoning:
                deltas.append(ReplayDelta(content=content, reasoning_content=reasoning))
                text_parts.append(content)
                reasoning_parts.append(reasoning)
            continue

        message = choice.get("message")
        if isinstance(message, dict):
            saw_completion_object = True
            content = _extract_text(message.get("content"))
            reasoning = ""
            for key in ("reasoning_content", "reasoning"):
                value = message.get(key)
                if isinstance(value, str):
                    reasoning = value
                    break
            if content or reasoning:
                deltas.append(ReplayDelta(content=content, reasoning_content=reasoning))
                text_parts.append(content)
                reasoning_parts.append(reasoning)

    return ReplayRecord(
        source_format=source_format,
        model=model,
        deltas=tuple(deltas),
        text="".join(text_parts),
        reasoning_content="".join(reasoning_parts),
        usage=usage,
        finish_reason=finish_reason or "stop",
        source_complete=bool(saw_done or finish_reason or saw_completion_object),
        source_error=source_error,
    )


def _map_anthropic_finish_reason(value: Any) -> str:
    reason = str(value or "").strip()
    if reason == "max_tokens":
        return "length"
    if reason in {"tool_use", "pause_turn"}:
        return "tool_calls"
    return "stop"


def _parse_anthropic(objects: list[dict[str, Any]], *, fallback_model: str) -> ReplayRecord:
    model = fallback_model
    deltas: list[ReplayDelta] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = "stop"
    source_complete = False
    source_error = None

    for data in objects:
        event_type = str(data.get("type") or "")
        if event_type == "error":
            error = data.get("error") if isinstance(data.get("error"), dict) else {}
            source_error = str(error.get("message") or json.dumps(data, ensure_ascii=False))
            continue
        if event_type == "message_start":
            message = data.get("message") if isinstance(data.get("message"), dict) else {}
            if message.get("model"):
                model = str(message["model"])
            usage = _merge_usage(usage, message.get("usage"))
            continue
        if event_type == "content_block_start":
            block = data.get("content_block") if isinstance(data.get("content_block"), dict) else {}
            text = block.get("text") if isinstance(block.get("text"), str) else ""
            reasoning = block.get("thinking") if isinstance(block.get("thinking"), str) else ""
            if text or reasoning:
                deltas.append(ReplayDelta(content=text, reasoning_content=reasoning))
                text_parts.append(text)
                reasoning_parts.append(reasoning)
            continue
        if event_type == "content_block_delta":
            delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
            delta_type = str(delta.get("type") or "")
            text = delta.get("text") if isinstance(delta.get("text"), str) else ""
            reasoning = ""
            if delta_type in {"thinking_delta", "reasoning_delta"}:
                reasoning = str(delta.get("thinking") or delta.get("reasoning") or text)
                text = ""
            if text or reasoning:
                deltas.append(ReplayDelta(content=text, reasoning_content=reasoning))
                text_parts.append(text)
                reasoning_parts.append(reasoning)
            continue
        if event_type == "message_delta":
            delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
            if delta.get("stop_reason"):
                finish_reason = _map_anthropic_finish_reason(delta.get("stop_reason"))
            usage = _merge_usage(usage, data.get("usage"))
            continue
        if event_type == "message_stop":
            source_complete = True

    return ReplayRecord(
        source_format="anthropic_sse",
        model=model,
        deltas=tuple(deltas),
        text="".join(text_parts),
        reasoning_content="".join(reasoning_parts),
        usage=usage,
        finish_reason=finish_reason,
        source_complete=source_complete,
        source_error=source_error,
    )


def _parse_responses(objects: list[dict[str, Any]], *, fallback_model: str, saw_done: bool) -> ReplayRecord:
    model = fallback_model
    deltas: list[ReplayDelta] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    source_complete = False
    source_error = None
    completed_fallback = ""

    for data in objects:
        event_type = str(data.get("type") or "")
        response = data.get("response") if isinstance(data.get("response"), dict) else {}
        if response.get("model"):
            model = str(response["model"])
        if event_type == "response.output_text.delta":
            text = data.get("delta") if isinstance(data.get("delta"), str) else ""
            if text:
                deltas.append(ReplayDelta(content=text))
                text_parts.append(text)
            continue
        if event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            reasoning = data.get("delta") if isinstance(data.get("delta"), str) else ""
            if reasoning:
                deltas.append(ReplayDelta(reasoning_content=reasoning))
                reasoning_parts.append(reasoning)
            continue
        if event_type == "response.completed":
            source_complete = True
            usage = _merge_usage(usage, response.get("usage"))
            completed_fallback = _responses_completed_text(response)
            continue
        if event_type in {"response.failed", "response.incomplete", "error"}:
            error = data.get("error") or response.get("error") or data
            source_error = str(error.get("message") if isinstance(error, dict) else error)

    if not text_parts and completed_fallback:
        deltas.append(ReplayDelta(content=completed_fallback))
        text_parts.append(completed_fallback)

    return ReplayRecord(
        source_format="responses_sse",
        model=model,
        deltas=tuple(deltas),
        text="".join(text_parts),
        reasoning_content="".join(reasoning_parts),
        usage=usage,
        finish_reason="stop",
        source_complete=bool(source_complete or saw_done),
        source_error=source_error,
    )


def _parse_gemini(objects: list[dict[str, Any]], *, fallback_model: str) -> ReplayRecord:
    deltas: list[ReplayDelta] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    source_complete = False

    for data in objects:
        usage = _merge_usage(usage, data.get("usageMetadata"))
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            continue
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                continue
            if part.get("thought"):
                deltas.append(ReplayDelta(reasoning_content=part["text"]))
                reasoning_parts.append(part["text"])
            else:
                deltas.append(ReplayDelta(content=part["text"]))
                text_parts.append(part["text"])
        if candidate.get("finishReason"):
            source_complete = True

    meta = objects[-1].get("usageMetadata") if objects else None
    if isinstance(meta, dict):
        usage = {
            **meta,
            "prompt_tokens": int(meta.get("promptTokenCount") or 0),
            "completion_tokens": int(meta.get("candidatesTokenCount") or 0),
            "total_tokens": int(meta.get("totalTokenCount") or 0),
        }

    return ReplayRecord(
        source_format="gemini_sse",
        model=fallback_model,
        deltas=tuple(deltas),
        text="".join(text_parts),
        reasoning_content="".join(reasoning_parts),
        usage=usage,
        finish_reason="stop",
        source_complete=source_complete,
        source_error=None,
    )


def parse_replay_record(
    raw_sse_text: str,
    *,
    fallback_model: str,
    fallback_output_text: str = "",
) -> ReplayRecord:
    events, plain_payload, saw_done, markers = _parse_wire_objects(raw_sse_text)
    source_format = _detect_format(events, plain_payload)
    objects = events if events else ([plain_payload] if plain_payload is not None else [])

    if source_format in {"openai_sse", "openai_json"}:
        record = _parse_openai(
            objects,
            source_format=source_format,
            fallback_model=fallback_model,
            saw_done=saw_done,
        )
    elif source_format == "anthropic_sse":
        record = _parse_anthropic(objects, fallback_model=fallback_model)
    elif source_format == "responses_sse":
        record = _parse_responses(objects, fallback_model=fallback_model, saw_done=saw_done)
    elif source_format == "gemini_sse":
        record = _parse_gemini(objects, fallback_model=fallback_model)
    elif fallback_output_text:
        record = ReplayRecord(
            source_format="output_snapshot",
            model=fallback_model,
            deltas=(ReplayDelta(content=fallback_output_text),),
            text=fallback_output_text,
            reasoning_content="",
            usage={},
            finish_reason="stop",
            source_complete=False,
            source_error=None,
        )
    else:
        raise ReplayPreparationError("selected replay file has no recognized response payload")

    marker_complete, marker_incomplete, marker_error = _marker_state(markers)
    source_complete = record.source_complete or marker_complete
    source_error = record.source_error or marker_error
    if marker_incomplete:
        source_complete = False

    if not record.text and not record.reasoning_content and fallback_output_text:
        record = ReplayRecord(
            source_format=record.source_format,
            model=record.model,
            deltas=(ReplayDelta(content=fallback_output_text),),
            text=fallback_output_text,
            reasoning_content=record.reasoning_content,
            usage=record.usage,
            finish_reason=record.finish_reason,
            source_complete=source_complete,
            source_error=source_error,
        )
    else:
        record = ReplayRecord(
            source_format=record.source_format,
            model=record.model,
            deltas=record.deltas,
            text=record.text,
            reasoning_content=record.reasoning_content,
            usage=record.usage,
            finish_reason=record.finish_reason,
            source_complete=source_complete,
            source_error=source_error,
        )

    if not record.text and not record.reasoning_content and not record.source_complete:
        detail = f": {record.source_error}" if record.source_error else ""
        raise ReplayPreparationError(f"selected replay contains no usable assistant output{detail}")
    return record


class ReplayService:
    """Load, normalize, and replay saved responses before provider routing."""

    def __init__(
        self,
        *,
        store: Any,
        log: Callable[[str], None],
        save_request_log: Callable[..., None],
        release_caller: Callable[[str, str], None],
        chunk_size: int = 1200,
    ):
        self.store = store
        self._log = log
        self._save_request_log = save_request_log
        self._release_caller = release_caller
        self._chunk_size = max(1, int(chunk_size))

    def prepare(self, *, model: str, messages: list) -> PreparedReplay | None:
        lookup = self.store.lookup(model=model, messages=messages)
        status = lookup.get("status")
        if status in {"disabled", "mismatch"}:
            return None
        if status != "ready":
            raise ReplayPreparationError(str(lookup.get("reason") or "replay lookup failed"))

        spec = lookup["spec"]
        record = parse_replay_record(
            spec["raw_sse_text"],
            fallback_model=model,
            fallback_output_text=str(spec.get("output_text") or ""),
        )
        if not self.store.consume_if_needed(spec):
            raise ReplayPreparationError(
                "once replay was already consumed or its selection changed during preparation"
            )
        return PreparedReplay(spec=spec, record=record)

    @staticmethod
    def _chunk_payload(
        *,
        stream_id: str,
        created: int,
        model: str,
        delta: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    @staticmethod
    def _emit(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    def _iter_deltas(self, record: ReplayRecord):
        if record.deltas:
            yield from record.deltas
            return
        for start in range(0, len(record.text), self._chunk_size):
            yield ReplayDelta(content=record.text[start : start + self._chunk_size])

    def _save(
        self,
        *,
        prepared: PreparedReplay,
        model: str,
        messages: list,
        request_payload: dict[str, Any],
        response_text: str,
        trace_id: str,
        stream: bool,
        error_type: str | None = None,
    ) -> None:
        try:
            self._save_request_log(
                model,
                messages,
                response_text,
                stream=stream,
                raw_sse=prepared.spec["raw_sse_text"],
                request_payload=request_payload,
                debug_meta={
                    "replay": {
                        "source_path": prepared.spec.get("raw_sse_path"),
                        "source_format": prepared.record.source_format,
                        "source_complete": prepared.record.source_complete,
                        "source_error": prepared.record.source_error,
                    }
                },
                error_type=error_type,
                trace_id=trace_id,
            )
        except Exception as save_error:
            self._log(f"[TRACE {trace_id}] replay_log_error err={save_error}")

    async def stream(
        self,
        prepared: PreparedReplay,
        *,
        model: str,
        messages: list,
        request_payload: dict[str, Any],
        trace_id: str,
        caller_key: str,
        caller_desc: str,
    ) -> AsyncGenerator[bytes, None]:
        record = prepared.record
        stream_id = f"chatcmpl-replay-{uuid.uuid4().hex[:18]}"
        created = int(time.time())
        sent_text = ""
        saved = False
        finish_status = "unknown"

        def persist(error_type: str | None = None) -> None:
            nonlocal saved
            if saved:
                return
            self._save(
                prepared=prepared,
                model=model,
                messages=messages,
                request_payload=request_payload,
                response_text=sent_text or record.text or "[empty replay response]",
                trace_id=trace_id,
                stream=True,
                error_type=error_type,
            )
            saved = True

        self._log(
            f"[TRACE {trace_id}] replay_stream_start caller={caller_key} {caller_desc} "
            f"format={record.source_format} source_complete={str(record.source_complete).lower()} "
            f"source={prepared.spec.get('raw_sse_path')}"
        )
        try:
            yield self._emit(self._chunk_payload(
                stream_id=stream_id,
                created=created,
                model=model,
                delta={"role": "assistant"},
            ))
            for delta in self._iter_deltas(record):
                delta_payload: dict[str, Any] = {}
                if delta.content:
                    delta_payload["content"] = delta.content
                    sent_text += delta.content
                if delta.reasoning_content:
                    delta_payload["reasoning_content"] = delta.reasoning_content
                if delta_payload:
                    yield self._emit(self._chunk_payload(
                        stream_id=stream_id,
                        created=created,
                        model=model,
                        delta=delta_payload,
                    ))
                await asyncio.sleep(0)

            persist()
            finish_status = "snapshot_completed" if not record.source_complete else "source_completed"
            yield self._emit(self._chunk_payload(
                stream_id=stream_id,
                created=created,
                model=model,
                finish_reason=record.finish_reason or "stop",
                usage=record.usage or None,
            ))
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            finish_status = "downstream_cancelled"
            persist("downstream_cancelled")
            raise
        except GeneratorExit:
            finish_status = "downstream_closed"
            persist("downstream_closed")
            raise
        finally:
            if not saved:
                persist("replay_stream_finalized")
            if caller_key:
                self._release_caller(caller_key, trace_id)
            self._log(
                f"[TRACE {trace_id}] replay_stream_done reason={finish_status} "
                f"format={record.source_format} out_chars={len(sent_text)}"
            )

    def complete(
        self,
        prepared: PreparedReplay,
        *,
        model: str,
        messages: list,
        request_payload: dict[str, Any],
        trace_id: str,
    ) -> dict[str, Any]:
        record = prepared.record
        self._save(
            prepared=prepared,
            model=model,
            messages=messages,
            request_payload=request_payload,
            response_text=record.text or "[empty replay response]",
            trace_id=trace_id,
            stream=False,
        )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": record.text,
        }
        if record.reasoning_content:
            message["reasoning_content"] = record.reasoning_content
        self._log(
            f"[TRACE {trace_id}] replay_nonstream_done format={record.source_format} "
            f"source_complete={str(record.source_complete).lower()} out_chars={len(record.text)}"
        )
        return {
            "id": f"chatcmpl-replay-{uuid.uuid4().hex[:18]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": record.finish_reason or "stop",
            }],
            "usage": record.usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
