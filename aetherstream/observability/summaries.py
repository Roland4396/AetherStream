"""Request-shape summaries used by trace logs and cache diagnostics."""

import copy
import json
from typing import Any


def _summarize_content_shape(content):
    if isinstance(content, str):
        return {
            "content_type": "str",
            "text_len": len(content),
            "line_count": content.count("\n") + 1,
        }

    if isinstance(content, list):
        block_types = []
        text_lens = []
        for item in content:
            if isinstance(item, str):
                block_types.append("str")
                text_lens.append(len(item))
                continue
            if isinstance(item, dict):
                item_type = str(item.get("type", "dict"))
                block_types.append(item_type)
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_lens.append(len(text_value))
                continue
            block_types.append(type(item).__name__)

        return {
            "content_type": "list",
            "block_count": len(content),
            "block_types": block_types[:12],
            "text_lens": text_lens[:12],
        }

    if content is None:
        return {"content_type": "none"}

    return {
        "content_type": type(content).__name__,
        "repr_len": len(str(content)),
    }


def summarize_openai_messages(messages: list) -> list[dict]:
    summary = []
    for idx, msg in enumerate(messages or []):
        if not isinstance(msg, dict):
            summary.append({"index": idx, "message_type": type(msg).__name__})
            continue
        item = {
            "index": idx,
            "role": msg.get("role"),
        }
        item.update(_summarize_content_shape(msg.get("content")))
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            item["tool_call_count"] = len(tool_calls)
        summary.append(item)
    return summary



def _normalize_claude_system_text(system_value: Any) -> tuple[str, int, int]:
    """Return (text, block_count, char_count) for top-level Anthropic system content."""
    parts: list[str] = []
    block_count = 0

    if isinstance(system_value, str):
        text = system_value.strip()
        if text:
            parts.append(text)
            block_count = 1
    elif isinstance(system_value, list):
        for item in system_value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                    block_count += 1
                continue
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    parts.append(text)
                    block_count += 1
                continue
            if item.get("type") == "text":
                text = item.get("text", "")
                if not isinstance(text, str):
                    text = str(text)
                text = text.strip()
                if text:
                    parts.append(text)
                    block_count += 1
                continue
            # Keep non-text system blocks visible as text rather than sending
            # unsupported top-level system content to Claude Code stream.
            text = json.dumps(item, ensure_ascii=False, separators=(",", ":")).strip()
            if text:
                parts.append(text)
                block_count += 1
    elif system_value is not None:
        text = str(system_value).strip()
        if text:
            parts.append(text)
            block_count = 1

    folded_text = "\n\n".join(parts).strip()
    return folded_text, block_count, len(folded_text)


def fold_claude_system_into_first_user_message(payload: dict) -> tuple[dict, dict[str, int | str]]:
    """Claude Code's streaming endpoint rejects top-level `system`.

    Preserve the instructions by moving them to the beginning of the first user
    message as ordinary text blocks, then remove the top-level field.
    """
    if not isinstance(payload, dict) or "system" not in payload:
        return payload, {"action": "absent", "blocks": 0, "chars": 0}

    system_value = payload.get("system")
    system_text, block_count, char_count = _normalize_claude_system_text(system_value)

    sanitized = copy.deepcopy(payload)
    sanitized.pop("system", None)

    if not system_text:
        return sanitized, {"action": "removed_empty", "blocks": block_count, "chars": 0}

    messages = sanitized.get("messages")
    if not isinstance(messages, list):
        messages = []
        sanitized["messages"] = messages

    prefix_block = {"type": "text", "text": system_text}
    insert_index = None
    for idx, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            insert_index = idx
            break

    if insert_index is None:
        messages.insert(0, {"role": "user", "content": [prefix_block]})
        return sanitized, {"action": "inserted_new_user", "blocks": block_count, "chars": char_count}

    message = messages[insert_index]
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [prefix_block, *content]
    elif isinstance(content, str):
        message["content"] = [prefix_block, {"type": "text", "text": content}]
    elif content is None:
        message["content"] = [prefix_block]
    else:
        message["content"] = [
            prefix_block,
            {"type": "text", "text": str(content)},
        ]

    return sanitized, {"action": "folded", "blocks": block_count, "chars": char_count}


def summarize_anthropic_request(request_payload: dict) -> dict:
    system_summary = []
    for idx, block in enumerate(request_payload.get("system", []) or []):
        if not isinstance(block, dict):
            system_summary.append({"index": idx, "block_type": type(block).__name__})
            continue
        system_summary.append({
            "index": idx,
            "type": block.get("type"),
            "text_len": len(block.get("text", "")) if isinstance(block.get("text"), str) else 0,
            "has_cache_control": isinstance(block.get("cache_control"), dict),
        })

    message_summary = []
    for idx, msg in enumerate(request_payload.get("messages", []) or []):
        if not isinstance(msg, dict):
            message_summary.append({"index": idx, "message_type": type(msg).__name__})
            continue
        content = msg.get("content")
        item = {
            "index": idx,
            "role": msg.get("role"),
        }
        item.update(_summarize_content_shape(content))
        if isinstance(content, list):
            item["cache_control_blocks"] = sum(
                1
                for block in content
                if isinstance(block, dict) and isinstance(block.get("cache_control"), dict)
            )
        message_summary.append(item)

    return {
        "top_level_keys": sorted(request_payload.keys()),
        "has_top_level_cache_control": isinstance(request_payload.get("cache_control"), dict),
        "system_blocks": system_summary,
        "messages": message_summary,
    }


def summarize_anthropic_cache_breakpoints(request_payload: dict) -> list[dict]:
    """Compact cache breakpoint layout for debugging prompt-cache misses.

    Do not include text content in logs; only structural location and sizes.
    `prefix_chars` is the cumulative text char count through that breakpoint in
    Anthropic request order, matching the prefix shape keepalive will trim to.
    """
    layout: list[dict] = []
    prefix_chars = 0

    system_value = request_payload.get("system")
    if isinstance(system_value, list):
        for block_index, block in enumerate(system_value):
            if not isinstance(block, dict):
                continue
            text_len = len(block.get("text", "")) if isinstance(block.get("text"), str) else 0
            prefix_chars += text_len
            cache_control = block.get("cache_control")
            if isinstance(cache_control, dict):
                layout.append({
                    "loc": "system",
                    "block": block_index,
                    "text_len": text_len,
                    "prefix_chars": prefix_chars,
                    "type": cache_control.get("type", "-"),
                    "ttl": cache_control.get("ttl", ""),
                })
    elif isinstance(system_value, str):
        prefix_chars += len(system_value)

    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return layout

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                text_len = len(block.get("text", "")) if isinstance(block.get("text"), str) else 0
                prefix_chars += text_len
                cache_control = block.get("cache_control")
                if isinstance(cache_control, dict):
                    layout.append({
                        "loc": "message",
                        "message": message_index,
                        "role": role,
                        "block": block_index,
                        "text_len": text_len,
                        "prefix_chars": prefix_chars,
                        "type": cache_control.get("type", "-"),
                        "ttl": cache_control.get("ttl", ""),
                    })
        elif isinstance(content, str):
            prefix_chars += len(content)

    return layout
