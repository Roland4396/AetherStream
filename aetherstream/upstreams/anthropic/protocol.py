import asyncio
import json
from typing import Any, AsyncGenerator, Callable


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
