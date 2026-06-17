import json
from typing import Any

SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"


def append_to_last_user_message(messages: list, append_text: str, marker: str = "") -> bool:
    """给最后一条 user 消息追加指令文本（支持 OpenAI / Anthropic 常见 content 结构）。"""
    if not isinstance(messages, list) or not append_text:
        return False

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            if marker and marker in content:
                return False
            msg["content"] = f"{content}\n\n{append_text}"
            return True

        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    if marker and marker in block["text"]:
                        return False
                    block["text"] = f"{block['text']}\n\n{append_text}"
                    return True

            content.append({"type": "text", "text": append_text})
            return True

        return False

    return False


def prefix_last_user_message(messages: list, prefix_text: str, marker: str = "") -> bool:
    """给最后一条 user 消息的正文前缀插入文本。

    若消息使用 <user_input> 包裹，则插入到 <user_input> 内部开头，避免把前缀放到整段
    XML/伪 XML 外面；否则插入到文本最前面。支持 OpenAI / Anthropic 常见 content 结构。
    """
    if not isinstance(messages, list) or not prefix_text:
        return False

    def apply_prefix(text: str) -> tuple[str, bool]:
        if not isinstance(text, str):
            return text, False
        if marker and marker in text:
            return text, False

        open_tag = "<user_input>"
        tag_pos = text.find(open_tag)
        if tag_pos != -1:
            insert_pos = tag_pos + len(open_tag)
            if text[insert_pos:insert_pos + 2] == "\r\n":
                insert_pos += 2
            elif text[insert_pos:insert_pos + 1] == "\n":
                insert_pos += 1
            elif text[insert_pos:insert_pos + 1] == "\r":
                insert_pos += 1

            if text[insert_pos:].lstrip().startswith(prefix_text):
                return text, False
            return text[:insert_pos] + prefix_text + text[insert_pos:], True

        if text.lstrip().startswith(prefix_text):
            return text, False
        return prefix_text + text, True

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            new_text, changed = apply_prefix(content)
            if changed:
                msg["content"] = new_text
            return changed

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    new_text, changed = apply_prefix(block["text"])
                    if changed:
                        block["text"] = new_text
                    return changed

            content.insert(0, {"type": "text", "text": prefix_text})
            return True

        return False

    return False


def insert_after_latest_human_message(messages: list, insert_text: str, marker: str = "") -> bool:
    """把文本插入到最后一条 user 消息里的 </latest_human_message> 后面。

    只处理已经由上游/酒馆包装好的 <latest_human_message> 块；如果找不到该块，
    返回 False，让调用方决定是否 fallback 到普通追加。
    """
    if not isinstance(messages, list) or not insert_text:
        return False

    close_tag = "</latest_human_message>"

    def apply_insert(text: str) -> tuple[str, bool]:
        if not isinstance(text, str):
            return text, False
        if marker and marker in text:
            return text, False

        pos = text.rfind(close_tag)
        if pos == -1:
            return text, False

        insert_pos = pos + len(close_tag)
        suffix = text[insert_pos:]
        if suffix.lstrip().startswith(insert_text):
            return text, False

        return text[:insert_pos] + f"\n\n{insert_text}" + text[insert_pos:], True

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            new_text, changed = apply_insert(content)
            if changed:
                msg["content"] = new_text
            return changed

        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    new_text, changed = apply_insert(block["text"])
                    if changed:
                        block["text"] = new_text
                    return changed
            return False

        return False

    return False


def extract_text_from_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
    return "\n".join(part for part in text_parts if part)


def normalize_chat_content_for_responses(role: str, content: Any):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    normalized_parts = []
    text_type = "output_text" if role == "assistant" else "input_text"

    for item in content:
        if isinstance(item, str):
            normalized_parts.append({
                "type": text_type,
                "text": item,
            })
            continue

        if not isinstance(item, dict):
            normalized_parts.append({
                "type": text_type,
                "text": str(item),
            })
            continue

        item_type = item.get("type")
        if item_type == "text":
            normalized_parts.append({
                "type": text_type,
                "text": item.get("text", ""),
            })
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url", "")
            normalized_parts.append({
                "type": "input_image",
                "image_url": image_url,
            })
        elif item_type == "input_audio":
            normalized_parts.append({
                "type": "input_audio",
                "input_audio": item.get("input_audio"),
            })
        elif item_type == "file":
            normalized_parts.append({
                "type": "input_file",
                "file": item.get("file"),
            })
        elif item_type == "video_url":
            video_url = item.get("video_url")
            if isinstance(video_url, dict):
                video_url = video_url.get("url", "")
            normalized_parts.append({
                "type": "input_video",
                "video_url": video_url,
            })
        else:
            fallback_text = item.get("text")
            if isinstance(fallback_text, str):
                normalized_parts.append({
                    "type": text_type,
                    "text": fallback_text,
                })

    return normalized_parts or ""


def convert_chat_response_format_to_responses_text(response_format: Any):
    if not isinstance(response_format, dict):
        return None

    response_type = str(response_format.get("type", "")).strip()
    if not response_type:
        return None

    fmt = {"type": response_type}
    if response_type == "json_schema":
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            for key, value in json_schema.items():
                if key == "type":
                    continue
                fmt[key] = value

    return {"format": fmt}


def convert_chat_to_responses_request(chat_request: dict) -> dict:
    messages = chat_request.get("messages", [])
    instructions_parts = []
    input_items = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "")).strip()
        if not role:
            continue

        content = msg.get("content")

        if role in {"system", "developer"}:
            text = extract_text_from_chat_content(content).strip()
            if text:
                instructions_parts.append(text)
            continue

        if role in {"tool", "function"}:
            call_id = str(msg.get("tool_call_id", "") or msg.get("toolCallId", "")).strip()
            if isinstance(content, str):
                output = content
            else:
                output = json.dumps(content, ensure_ascii=False)
            if call_id:
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                })
            else:
                input_items.append({
                    "role": "user",
                    "content": f"[tool_output_missing_call_id] {output}",
                })
            continue

        input_items.append({
            "role": role,
            "content": normalize_chat_content_for_responses(role, content),
        })

        if role == "assistant":
            for tool_call in msg.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id", "")).strip()
                if not call_id:
                    continue
                if tool_call.get("type") not in (None, "", "function"):
                    continue
                fn = tool_call.get("function") or {}
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name", "")).strip()
                if not name:
                    continue
                input_items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": fn.get("arguments", ""),
                })

    response_request = {
        "model": chat_request.get("model", ""),
        "input": input_items,
        "stream": bool(chat_request.get("stream", False)),
    }

    if instructions_parts:
        response_request["instructions"] = "\n\n".join(instructions_parts)

    passthrough_fields = [
        "temperature",
        "top_p",
        "metadata",
        "store",
        "user",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "truncation",
    ]
    for field in passthrough_fields:
        if field in chat_request:
            response_request[field] = chat_request[field]

    if isinstance(chat_request.get("reasoning"), dict):
        response_request["reasoning"] = chat_request["reasoning"]

    text_config = convert_chat_response_format_to_responses_text(chat_request.get("response_format"))
    if text_config:
        response_request["text"] = text_config

    max_tokens = chat_request.get("max_tokens")
    max_completion_tokens = chat_request.get("max_completion_tokens")
    if isinstance(max_tokens, int) and isinstance(max_completion_tokens, int):
        response_request["max_output_tokens"] = max(max_tokens, max_completion_tokens)
    elif isinstance(max_tokens, int):
        response_request["max_output_tokens"] = max_tokens
    elif isinstance(max_completion_tokens, int):
        response_request["max_output_tokens"] = max_completion_tokens

    return response_request
def build_anthropic_text_blocks(
    text: str,
) -> list[dict[str, Any]]:
    if not text:
        return []
    return [{"type": "text", "text": text}]


def normalize_chat_content_for_anthropic(
    content: Any,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    if content is None:
        return blocks

    if isinstance(content, str):
        return build_anthropic_text_blocks(content)

    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    for item in content:
        if isinstance(item, str):
            if item:
                blocks.append({"type": "text", "text": item})
            continue

        if not isinstance(item, dict):
            blocks.append({"type": "text", "text": str(item)})
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            blocks.append({"type": "text", "text": text})
            continue

        if item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url", "")
            if image_url:
                blocks.append({"type": "text", "text": f"[image_url] {image_url}"})
            continue

        if item_type == "input_audio":
            blocks.append({"type": "text", "text": "[input_audio omitted]"})
            continue

        if item_type == "file":
            file_info = item.get("file")
            blocks.append({
                "type": "text",
                "text": f"[file omitted] {json.dumps(file_info, ensure_ascii=False)}",
            })
            continue

        fallback_text = item.get("text")
        if isinstance(fallback_text, str) and fallback_text:
            blocks.append({"type": "text", "text": fallback_text})
            continue

        blocks.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})

    return blocks


def _append_anthropic_message(messages: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    if not blocks:
        return

    if messages and messages[-1].get("role") == role and isinstance(messages[-1].get("content"), list):
        messages[-1]["content"].extend(blocks)
        return

    messages.append({
        "role": role,
        "content": blocks,
    })


def _clone_cache_control(
    cache_control: dict[str, Any] | None,
    *,
    scope: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(cache_control, dict):
        return None
    cloned = dict(cache_control)
    if scope:
        cloned["scope"] = scope
    else:
        cloned.pop("scope", None)
    return cloned


def _build_system_blocks(
    *,
    system_prefix: str | list[dict[str, Any]],
    system_messages: list[str],
    cache_control: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    prefix_blocks: list[dict[str, Any]] = []
    prefix_parts: list[str] = []
    prefix_has_explicit_blocks = isinstance(system_prefix, list)
    if isinstance(system_prefix, list):
        for item in system_prefix:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    prefix_blocks.append({"type": "text", "text": text})
                continue
            block = dict(item)
            text = block.get("text")
            if isinstance(text, str):
                block["text"] = text.strip()
                if block["text"]:
                    prefix_blocks.append(block)
    else:
        prefix_parts = [
            part.strip()
            for part in str(system_prefix or "").split("\n\n")
            if isinstance(part, str) and part.strip()
        ]
    normalized_parts = [
        text.strip()
        for text in system_messages
        if isinstance(text, str) and text.strip()
    ]

    boundary_index = -1
    for idx, text in enumerate(normalized_parts):
        if text == SYSTEM_PROMPT_DYNAMIC_BOUNDARY:
            boundary_index = idx
            break

    if boundary_index != -1:
        static_parts = normalized_parts[:boundary_index]
        dynamic_parts = normalized_parts[boundary_index + 1:]
        static_text = "\n\n".join(static_parts)
        dynamic_text = "\n\n".join(dynamic_parts)

        if prefix_blocks:
            blocks.extend(dict(block) for block in prefix_blocks)
        else:
            for prefix_text in prefix_parts:
                blocks.append({"type": "text", "text": prefix_text})
        if static_text:
            static_block = {"type": "text", "text": static_text}
            global_cache_control = _clone_cache_control(cache_control, scope="global")
            if isinstance(global_cache_control, dict):
                static_block["cache_control"] = global_cache_control
            blocks.append(static_block)
        if dynamic_text:
            blocks.append({"type": "text", "text": dynamic_text})
        return blocks

    rest_joined = "\n\n".join(normalized_parts)

    org_cache_control = None if prefix_has_explicit_blocks else _clone_cache_control(cache_control)
    if prefix_blocks:
        blocks.extend(dict(block) for block in prefix_blocks)
    else:
        for prefix_text in prefix_parts:
            prefix_block = {"type": "text", "text": prefix_text}
            if isinstance(org_cache_control, dict):
                prefix_block["cache_control"] = dict(org_cache_control)
            blocks.append(prefix_block)

    if rest_joined:
        rest_block = {"type": "text", "text": rest_joined}
        if isinstance(org_cache_control, dict):
            rest_block["cache_control"] = dict(org_cache_control)
        blocks.append(rest_block)

    return blocks


def _find_last_cacheable_block_index(blocks: list[dict[str, Any]]) -> int | None:
    for idx in range(len(blocks) - 1, -1, -1):
        block = blocks[idx]
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "") or "")
        if block_type in {"thinking", "redacted_thinking"}:
            continue
        text = block.get("text")
        if isinstance(text, str):
            if not text:
                continue
            return idx
        return idx
    return None


def _apply_message_cache_marker(
    messages: list[dict[str, Any]],
    *,
    cache_control: dict[str, Any] | None,
) -> bool:
    if not isinstance(cache_control, dict):
        return False

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list) or not content:
            continue
        block_index = _find_last_cacheable_block_index(content)
        if block_index is None:
            continue
        content[block_index]["cache_control"] = dict(cache_control)
        return True

    return False


def _text_len_for_anthropic_blocks(blocks: Any) -> int:
    if not isinstance(blocks, list):
        return 0
    total = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            total += len(text)
    return total


def _message_text_for_cache_strategy(message: dict[str, Any]) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _find_marker_position(text: str, markers: tuple[str, ...]) -> int:
    positions = [text.find(marker) for marker in markers if marker and text.find(marker) >= 0]
    return min(positions) if positions else -1


def _replace_text_message_with_cache_segments(
    message: dict[str, Any],
    *,
    text: str,
    split_positions: list[int],
    cache_segment_indices: set[int],
    cache_control: dict[str, Any],
) -> int:
    """Split a text-only message into Anthropic text blocks and mark selected segments.

    The split preserves the original concatenated text byte-for-byte at the
    Python string level.  This is important for prompt-cache equality: changing
    whitespace to make prettier blocks would change the cache prefix itself.
    """
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return 0
    if not all(isinstance(block, dict) and block.get("type") == "text" for block in content):
        return 0

    normalized_positions = sorted({
        pos
        for pos in split_positions
        if isinstance(pos, int) and 0 < pos < len(text)
    })
    if not normalized_positions:
        return 0

    boundaries = [0, *normalized_positions, len(text)]
    segments: list[str] = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment = text[start:end]
        if segment:
            segments.append(segment)
    if len(segments) <= 1:
        return 0

    new_content: list[dict[str, Any]] = []
    applied = 0
    for idx, segment in enumerate(segments):
        block: dict[str, Any] = {"type": "text", "text": segment}
        if idx in cache_segment_indices:
            block["cache_control"] = dict(cache_control)
            applied += 1
        new_content.append(block)

    if applied <= 0:
        return 0
    message["content"] = new_content
    return applied


def _apply_opus_roleplay_cache_markers(
    messages: list[dict[str, Any]],
    *,
    cache_control: dict[str, Any] | None,
) -> int:
    if not isinstance(cache_control, dict) or not isinstance(messages, list):
        return 0

    # Opus roleplay cache policy: create only the first stable preset block.
    #
    # Previous layered markers also cached the Lore/table/memory tail.  In long
    # SillyTavern sessions those later sections can reorder or change between
    # turns, causing Claude to miss the whole prompt-cache read and rewrite a
    # large prefix.  Keep Opus conservative: split at the live table boundary
    # and mark only segment 0.  That segment is the fixed writing/style/user-role
    # preset before <Lore>, matching the old block0.
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text_for_cache_strategy(message)
        if len(text) < 4096:
            continue

        table_pos = _find_marker_position(
            text,
            (
                "\n# 全局数据表",
                "# 全局数据表",
                "\n# 当前表格数据",
                "# 当前表格数据",
            ),
        )
        if table_pos <= 0:
            continue

        lore_pos = _find_marker_position(
            text,
            (
                "\n<Lore>",
                "<Lore>",
                "\n<Background>",
                "<Background>",
            ),
        )
        if not (4096 <= lore_pos <= table_pos - 4096):
            continue

        applied = _replace_text_message_with_cache_segments(
            message,
            text=text,
            split_positions=[lore_pos],
            cache_segment_indices={0},
            cache_control=cache_control,
        )
        if applied > 0:
            return applied

    return 0


def _apply_sonnet_fill_table_cache_markers(
    messages: list[dict[str, Any]],
    *,
    cache_control: dict[str, Any] | None,
) -> int:
    if not isinstance(cache_control, dict) or not isinstance(messages, list):
        return 0

    # Sonnet is used here as a staged fill-table worker.  Its cache goal has
    # two layers:
    # 1) exact retry after an upstream failure should read almost the full
    #    request, so the final execution message gets a breakpoint;
    # 2) the next successful fill-table turn should still have a stable prefix,
    #    so the rule/background/body material messages get earlier breakpoints.
    wanted: dict[str, int] = {}

    for idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text_for_cache_strategy(message).lstrip()
        if "rule" not in wanted and text.startswith("你是【填表AI】"):
            wanted["rule"] = idx
        elif "background" not in wanted and text.startswith("第一份材料：<背景设定>"):
            wanted["background"] = idx
        elif "table" not in wanted and text.startswith("第三份材料：<当前表格数据>"):
            wanted["table"] = idx
        elif "execute" not in wanted and text.startswith("现在开始执行填表"):
            wanted["execute"] = idx

    target_indices = [
        wanted[key]
        for key in ("rule", "background", "table", "execute")
        if key in wanted
    ]

    applied = 0
    for idx in target_indices[:4]:
        content = messages[idx].get("content")
        if not isinstance(content, list):
            continue
        block_index = _find_last_cacheable_block_index(content)
        if block_index is None:
            continue
        content[block_index]["cache_control"] = dict(cache_control)
        applied += 1

    return applied


def _apply_user_message_cache_markers(
    messages: list[dict[str, Any]],
    *,
    cache_control: dict[str, Any] | None,
    max_markers: int = 3,
    min_text_len: int = 1024,
) -> int:
    """Mark stable user-message breakpoints instead of the volatile latest user.

    SillyTavern/OpenAI-format requests here already split the prompt into large
    staged user messages: preset / background / current tables / older context /
    latest human input.  The latest user message changes every turn, so cache it
    only if there is no better stable candidate.  Prefer the largest previous
    user messages, then apply markers in original order.
    """
    if not isinstance(cache_control, dict) or not isinstance(messages, list):
        return 0

    last_user_index = None
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = idx
            break

    candidates: list[tuple[int, int]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if idx == last_user_index:
            continue
        content = message.get("content")
        if not isinstance(content, list) or not content:
            continue
        if _find_last_cacheable_block_index(content) is None:
            continue
        text_len = _text_len_for_anthropic_blocks(content)
        if text_len < min_text_len:
            continue
        candidates.append((idx, text_len))

    selected = sorted(
        sorted(candidates, key=lambda item: item[1], reverse=True)[:max_markers],
        key=lambda item: item[0],
    )

    applied = 0
    for idx, _text_len in selected:
        content = messages[idx].get("content")
        if not isinstance(content, list):
            continue
        block_index = _find_last_cacheable_block_index(content)
        if block_index is None:
            continue
        content[block_index]["cache_control"] = dict(cache_control)
        applied += 1

    if applied == 0:
        return 1 if _apply_message_cache_marker(messages, cache_control=cache_control) else 0

    return applied


def convert_chat_to_anthropic_messages_request(
    chat_request: dict,
    *,
    system_prefix: str = "",
    metadata_user_id: str = "",
    default_max_tokens: int = 8192,
    prompt_cache_control: dict[str, Any] | None = None,
    prompt_cache_strategy: str = "generic",
) -> dict:
    messages = chat_request.get("messages", [])
    system_messages: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "")).strip()
        if not role:
            continue

        content = msg.get("content")

        if role in {"system", "developer"}:
            text = extract_text_from_chat_content(content).strip()
            if text:
                system_messages.append(text)
            continue

        if role in {"tool", "function"}:
            call_id = str(msg.get("tool_call_id", "") or msg.get("toolCallId", "")).strip()
            tool_name = str(msg.get("name", "")).strip()
            if isinstance(content, str):
                output = content
            else:
                output = json.dumps(content, ensure_ascii=False)
            prefix = "[tool_result]"
            if tool_name:
                prefix += f" name={tool_name}"
            if call_id:
                prefix += f" call_id={call_id}"
            _append_anthropic_message(
                anthropic_messages,
                "user",
                [{"type": "text", "text": f"{prefix}\n{output}"}],
            )
            continue

        if role not in {"user", "assistant"}:
            text = extract_text_from_chat_content(content).strip()
            if text:
                _append_anthropic_message(
                    anthropic_messages,
                    "user",
                    [{"type": "text", "text": f"[{role}]\n{text}"}],
                )
            continue

        blocks = normalize_chat_content_for_anthropic(
            content,
        )

        if role == "assistant":
            for tool_call in msg.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") or {}
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name", "")).strip()
                call_id = str(tool_call.get("id", "")).strip()
                arguments = fn.get("arguments", "")
                call_payload = {
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments,
                }
                blocks.append({
                    "type": "text",
                    "text": f"[assistant_tool_call]\n{json.dumps(call_payload, ensure_ascii=False)}",
                })

        _append_anthropic_message(anthropic_messages, role, blocks)

    system_blocks = _build_system_blocks(
        system_prefix=system_prefix,
        system_messages=system_messages,
        cache_control=prompt_cache_control,
    )
    if prompt_cache_strategy == "sonnet_fill_table":
        applied_cache_markers = _apply_sonnet_fill_table_cache_markers(
            anthropic_messages,
            cache_control=prompt_cache_control,
        )
        if applied_cache_markers == 0:
            _apply_user_message_cache_markers(
                anthropic_messages,
                cache_control=prompt_cache_control,
            )
    elif prompt_cache_strategy == "opus_roleplay_layered":
        applied_cache_markers = _apply_opus_roleplay_cache_markers(
            anthropic_messages,
            cache_control=prompt_cache_control,
        )
        if applied_cache_markers == 0:
            _apply_user_message_cache_markers(
                anthropic_messages,
                cache_control=prompt_cache_control,
            )
    else:
        _apply_user_message_cache_markers(
            anthropic_messages,
            cache_control=prompt_cache_control,
        )

    anthropic_request = {
        "model": chat_request.get("model", ""),
        "messages": anthropic_messages,
        "stream": bool(chat_request.get("stream", False)),
    }

    if system_blocks:
        anthropic_request["system"] = system_blocks

    metadata = chat_request.get("metadata")
    if isinstance(metadata, dict):
        anthropic_metadata = dict(metadata)
    else:
        anthropic_metadata = {}
    if metadata_user_id:
        anthropic_metadata["user_id"] = metadata_user_id
    if anthropic_metadata:
        anthropic_request["metadata"] = anthropic_metadata

    max_tokens = chat_request.get("max_completion_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = chat_request.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = default_max_tokens
    anthropic_request["max_tokens"] = max_tokens

    stop = chat_request.get("stop")
    if isinstance(stop, str) and stop:
        anthropic_request["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        stop_sequences = [item for item in stop if isinstance(item, str) and item]
        if stop_sequences:
            anthropic_request["stop_sequences"] = stop_sequences

    passthrough_fields = [
        "temperature",
        "top_p",
        "top_k",
        "thinking",
        "context_management",
        "output_config",
    ]
    for field in passthrough_fields:
        if field in chat_request:
            anthropic_request[field] = chat_request[field]

    return anthropic_request
