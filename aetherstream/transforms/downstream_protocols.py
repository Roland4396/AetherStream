"""Ingress and egress adapters for the public Responses and Messages APIs."""

from __future__ import annotations

import copy
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _content_is_empty(content: Any) -> bool:
    if isinstance(content, str):
        return not content.strip()
    if not isinstance(content, list):
        return content is None
    for part in content:
        if isinstance(part, str) and part.strip():
            return False
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            if str(part.get("text") or "").strip():
                return False
        else:
            return False
    return True


def _chat_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return copy.deepcopy(tool)
    name = str(tool.get("name") or "").strip()
    if not name:
        return None
    function: dict[str, Any] = {
        "name": name,
        "parameters": copy.deepcopy(
            tool.get("parameters") or tool.get("input_schema") or {"type": "object"}
        ),
    }
    if tool.get("description") is not None:
        function["description"] = str(tool["description"])
    if "strict" in tool:
        function["strict"] = bool(tool["strict"])
    return {"type": "function", "function": function}


def responses_to_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    chat: dict[str, Any] = {
        "model": payload.get("model", ""),
        "stream": bool(payload.get("stream", False)),
        "messages": [],
    }
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        chat["messages"].append({"role": "system", "content": instructions})

    raw_input = payload.get("input", [])
    if isinstance(raw_input, str):
        raw_input = [{"role": "user", "content": raw_input}]
    elif isinstance(raw_input, dict):
        raw_input = [raw_input]
    elif not isinstance(raw_input, list):
        raw_input = []

    loose_user_parts: list[dict[str, Any]] = []
    for item in raw_input:
        if isinstance(item, str):
            loose_user_parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            chat["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": arguments,
                    },
                }],
            })
            continue
        if item_type == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            chat["messages"].append({
                "role": "tool",
                "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                "content": output,
            })
            continue
        if item_type in {"input_text", "output_text"}:
            loose_user_parts.append({"type": "text", "text": str(item.get("text") or "")})
            continue
        if item_type in {"input_image", "image_url"}:
            image_url = item.get("image_url") or item.get("url")
            if image_url:
                loose_user_parts.append({"type": "image_url", "image_url": {"url": image_url}})
            continue
        if item_type == "message" or item.get("role"):
            role = str(item.get("role") or "user")
            content = item.get("content", "")
            normalized: list[dict[str, Any]] = []
            if isinstance(content, str):
                normalized.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        if isinstance(part, str):
                            normalized.append({"type": "text", "text": part})
                        continue
                    part_type = part.get("type")
                    if part_type in {"input_text", "output_text", "text"}:
                        normalized.append({"type": "text", "text": str(part.get("text") or "")})
                    elif part_type in {"input_image", "image_url"}:
                        image_url = part.get("image_url") or part.get("url")
                        if image_url:
                            normalized.append({"type": "image_url", "image_url": {"url": image_url}})
            normalized_content: Any = (
                normalized
                if len(normalized) != 1 or normalized[0].get("type") != "text"
                else normalized[0]["text"]
            )
            if role == "developer":
                if _content_is_empty(normalized_content):
                    continue
                if payload.get("model") == "deepseek-v4-flash-local":
                    role = "system"
            chat["messages"].append({"role": role, "content": normalized_content})

    if loose_user_parts:
        chat["messages"].append({"role": "user", "content": loose_user_parts})

    converted_tools = [converted for tool in payload.get("tools", []) or [] if (converted := _chat_tool(tool))]
    if converted_tools:
        chat["tools"] = converted_tools
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and tool_choice.get("name"):
        chat["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    elif tool_choice is not None:
        chat["tool_choice"] = copy.deepcopy(tool_choice)

    passthrough = (
        "temperature", "top_p", "metadata", "user", "parallel_tool_calls",
        "service_tier", "store", "seed", "stop", "truncation",
    )
    for field in passthrough:
        if field in payload:
            chat[field] = copy.deepcopy(payload[field])
    if isinstance(payload.get("max_output_tokens"), int):
        chat["max_tokens"] = payload["max_output_tokens"]
    if isinstance(payload.get("reasoning"), dict):
        chat["reasoning"] = copy.deepcopy(payload["reasoning"])
        if payload["reasoning"].get("effort"):
            chat["reasoning_effort"] = payload["reasoning"]["effort"]
    text_config = payload.get("text")
    if isinstance(text_config, dict) and isinstance(text_config.get("format"), dict):
        chat["response_format"] = copy.deepcopy(text_config["format"])
    return chat


def anthropic_to_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    chat: dict[str, Any] = {
        "model": payload.get("model", ""),
        "stream": bool(payload.get("stream", False)),
        "messages": [],
    }
    system = payload.get("system")
    if system:
        system_text = _text_from_content(system)
        if system_text:
            chat["messages"].append({"role": "system", "content": system_text})

    for message in payload.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if isinstance(content, str):
            chat["messages"].append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        text_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        def flush_text() -> None:
            nonlocal text_parts
            if text_parts:
                chat["messages"].append({"role": role, "content": text_parts})
                text_parts = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_item = {"type": "text", "text": str(block.get("text") or "")}
                if isinstance(block.get("cache_control"), dict):
                    text_item["cache_control"] = copy.deepcopy(block["cache_control"])
                text_parts.append(text_item)
            elif block_type == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    url = f"data:{source.get('media_type', 'application/octet-stream')};base64,{source.get('data', '')}"
                else:
                    url = source.get("url") or ""
                if url:
                    text_parts.append({"type": "image_url", "image_url": {"url": url}})
            elif block_type == "tool_use":
                arguments = block.get("input")
                tool_calls.append({
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":")),
                    },
                })
            elif block_type == "tool_result":
                flush_text()
                result = block.get("content", "")
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
                chat["messages"].append({
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": result,
                })

        if text_parts or tool_calls:
            chat["messages"].append({
                "role": role,
                "content": text_parts if text_parts else None,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })

    converted_tools = [converted for tool in payload.get("tools", []) or [] if (converted := _chat_tool(tool))]
    if converted_tools:
        chat["tools"] = converted_tools
    choice = payload.get("tool_choice")
    if isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "tool" and choice.get("name"):
            chat["tool_choice"] = {
                "type": "function",
                "function": {"name": choice["name"]},
            }
        else:
            chat["tool_choice"] = {"any": "required"}.get(choice_type, choice_type)
        if choice.get("disable_parallel_tool_use") is True:
            chat["parallel_tool_calls"] = False
    if isinstance(payload.get("max_tokens"), int):
        chat["max_tokens"] = payload["max_tokens"]
    if "stop_sequences" in payload:
        chat["stop"] = copy.deepcopy(payload["stop_sequences"])
    for field in ("temperature", "top_p", "top_k", "metadata"):
        if field in payload:
            chat[field] = copy.deepcopy(payload[field])
    return chat


async def iter_sse_events(body: AsyncIterator[Any]) -> AsyncIterator[tuple[str, str]]:
    buffer = ""
    try:
        async for chunk in body:
            if isinstance(chunk, bytes):
                buffer += chunk.decode("utf-8", errors="replace")
            else:
                buffer += str(chunk)
            buffer = buffer.replace("\r\n", "\n")
            while "\n\n" in buffer:
                raw_event, buffer = buffer.split("\n\n", 1)
                event_name = "message"
                data_lines: list[str] = []
                for line in raw_event.split("\n"):
                    if line.startswith("event:"):
                        event_name = line[6:].strip() or "message"
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines:
                    yield event_name, "\n".join(data_lines)
        if buffer.strip():
            data_lines = [line[5:].lstrip() for line in buffer.split("\n") if line.startswith("data:")]
            if data_lines:
                yield "message", "\n".join(data_lines)
    finally:
        close = getattr(body, "aclose", None)
        if callable(close):
            with suppress(StopAsyncIteration):
                await close()


def _sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    ).encode()


async def chat_stream_to_responses(
    body: AsyncIterator[Any],
    *,
    model: str,
) -> AsyncIterator[bytes]:
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created_at = int(time.time())
    sequence = 0
    text = ""
    text_started = False
    response_model = model
    usage: dict[str, Any] = {}
    tools: dict[int, dict[str, Any]] = {}
    finished = False
    failed = False
    last_error: dict[str, Any] | None = None

    def event(event_type: str, payload: dict[str, Any]) -> bytes:
        nonlocal sequence
        payload = dict(payload)
        payload.setdefault("type", event_type)
        payload.setdefault("sequence_number", sequence)
        sequence += 1
        return _sse(event_type, payload)

    def response_object(status: str) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        if text_started:
            output.append({
                "id": message_id,
                "type": "message",
                "status": "completed" if status == "completed" else "in_progress",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }],
            })
        for index in sorted(tools):
            tool = tools[index]
            if not tool.get("started"):
                continue
            output.append({
                "id": tool["item_id"],
                "type": "function_call",
                "status": "completed" if status == "completed" else "in_progress",
                "call_id": tool["call_id"],
                "name": tool["name"],
                "arguments": tool["arguments"],
            })
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": response_model,
            "output": output,
            "usage": usage or None,
            "error": None,
        }

    async def finalize() -> AsyncIterator[bytes]:
        nonlocal finished
        if finished:
            return
        if text_started:
            yield event("response.output_text.done", {
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            })
            yield event("response.content_part.done", {
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            })
            yield event("response.output_item.done", {
                "output_index": 0,
                "item": response_object("completed")["output"][0],
            })
        output_offset = 1 if text_started else 0
        for index in sorted(tools):
            tool = tools[index]
            if not tool.get("started"):
                continue
            output_index = output_offset + index
            yield event("response.function_call_arguments.done", {
                "item_id": tool["item_id"],
                "output_index": output_index,
                "arguments": tool["arguments"],
            })
            yield event("response.output_item.done", {
                "output_index": output_index,
                "item": {
                    "id": tool["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tool["call_id"],
                    "name": tool["name"],
                    "arguments": tool["arguments"],
                },
            })
        yield event("response.completed", {"response": response_object("completed")})
        finished = True

    yield event("response.created", {"response": response_object("in_progress")})
    yield event("response.in_progress", {"response": response_object("in_progress")})
    async for _event_name, payload in iter_sse_events(body):
        if payload == "[DONE]":
            if failed:
                failed_response = response_object("failed")
                failed_response["error"] = last_error
                yield event("response.failed", {"response": failed_response})
                return
            async for tail in finalize():
                yield tail
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data.get("error"), dict):
            failed = True
            error = data["error"]
            last_error = {
                "code": error.get("type") or "upstream_error",
                "message": error.get("message") or "Upstream error",
            }
            yield event("error", {
                **last_error,
                "param": None,
            })
            continue
        if data.get("model"):
            response_model = str(data["model"])
        if isinstance(data.get("created"), int):
            created_at = data["created"]
        if isinstance(data.get("usage"), dict):
            raw_usage = data["usage"]
            usage = {
                "input_tokens": int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0),
                "output_tokens": int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0),
                "total_tokens": int(raw_usage.get("total_tokens") or 0),
            }
            if not usage["total_tokens"]:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        emitted_payload = False
        if isinstance(content, str) and content:
            emitted_payload = True
            if not text_started:
                text_started = True
                yield event("response.output_item.added", {
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                })
                yield event("response.content_part.added", {
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
            text += content
            yield event("response.output_text.delta", {
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "delta": content,
            })
        raw_tool_calls = delta.get("tool_calls", []) or []
        if not raw_tool_calls and isinstance(delta.get("function_call"), dict):
            raw_tool_calls = [{"index": 0, "function": delta["function_call"]}]
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            emitted_payload = True
            index = int(raw_call.get("index") or 0)
            state = tools.setdefault(index, {
                "item_id": f"fc_{uuid.uuid4().hex}",
                "call_id": "",
                "name": "",
                "arguments": "",
                "started": False,
                "pending_arguments": "",
            })
            if raw_call.get("id"):
                state["call_id"] = str(raw_call["id"])
            function = raw_call.get("function")
            if isinstance(function, dict):
                if function.get("name"):
                    state["name"] += str(function["name"])
                if function.get("arguments"):
                    state["pending_arguments"] += str(function["arguments"])
            if state["name"] and not state["started"]:
                state["started"] = True
                state["call_id"] = state["call_id"] or f"call_{uuid.uuid4().hex}"
                yield event("response.output_item.added", {
                    "output_index": (1 if text_started else 0) + index,
                    "item": {
                        "id": state["item_id"],
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": state["call_id"],
                        "name": state["name"],
                        "arguments": "",
                    },
                })
            if state["started"] and state["pending_arguments"]:
                arguments_delta = state["pending_arguments"]
                state["pending_arguments"] = ""
                state["arguments"] += arguments_delta
                yield event("response.function_call_arguments.delta", {
                    "item_id": state["item_id"],
                    "output_index": (1 if text_started else 0) + index,
                    "delta": arguments_delta,
                })
        if not emitted_payload and not choices[0].get("finish_reason"):
            yield b": keepalive\n\n"

    if failed:
        failed_response = response_object("failed")
        failed_response["error"] = last_error
        yield event("response.failed", {"response": failed_response})
        return
    async for tail in finalize():
        yield tail


async def chat_stream_to_anthropic(
    body: AsyncIterator[Any],
    *,
    model: str,
) -> AsyncIterator[bytes]:
    message_id = f"msg_{uuid.uuid4().hex}"
    response_model = model
    text = ""
    text_index: int | None = None
    next_block_index = 0
    tools: dict[int, dict[str, Any]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    finish_reason = "stop"
    finished = False
    failed = False

    message = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": response_model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage,
    }
    yield _sse("message_start", {"type": "message_start", "message": message})

    async def finalize() -> AsyncIterator[bytes]:
        nonlocal finished
        if finished:
            return
        if text_index is not None:
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": text_index,
            })
        for index in sorted(tools):
            state = tools[index]
            if state["started"]:
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": state["block_index"],
                })
        stop_reason = {
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
        }.get(finish_reason, "end_turn")
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        })
        yield _sse("message_stop", {"type": "message_stop"})
        finished = True

    async for _event_name, payload in iter_sse_events(body):
        if payload == "[DONE]":
            if failed:
                yield _sse("message_stop", {"type": "message_stop"})
                return
            async for tail in finalize():
                yield tail
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data.get("error"), dict):
            failed = True
            error = data["error"]
            yield _sse("error", {
                "type": "error",
                "error": {
                    "type": error.get("type") or "upstream_error",
                    "message": error.get("message") or "Upstream error",
                },
            })
            continue
        if data.get("model"):
            response_model = str(data["model"])
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            usage = {
                "input_tokens": int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0),
                "output_tokens": int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0),
            }
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        emitted_payload = False
        if isinstance(content, str) and content:
            emitted_payload = True
            if text_index is None:
                text_index = next_block_index
                next_block_index += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": text_index,
                    "content_block": {"type": "text", "text": "", "citations": None},
                })
            text += content
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": content},
            })
        raw_tool_calls = delta.get("tool_calls", []) or []
        if not raw_tool_calls and isinstance(delta.get("function_call"), dict):
            raw_tool_calls = [{"index": 0, "function": delta["function_call"]}]
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            emitted_payload = True
            call_index = int(raw_call.get("index") or 0)
            state = tools.setdefault(call_index, {
                "id": "",
                "name": "",
                "arguments": "",
                "pending_arguments": "",
                "started": False,
                "block_index": -1,
            })
            if raw_call.get("id"):
                state["id"] = str(raw_call["id"])
            function = raw_call.get("function")
            if isinstance(function, dict):
                if function.get("name"):
                    state["name"] += str(function["name"])
                if function.get("arguments"):
                    state["pending_arguments"] += str(function["arguments"])
            if state["name"] and not state["started"]:
                state["started"] = True
                state["block_index"] = next_block_index
                next_block_index += 1
                state["id"] = state["id"] or f"toolu_{uuid.uuid4().hex}"
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": state["block_index"],
                    "content_block": {
                        "type": "tool_use",
                        "id": state["id"],
                        "name": state["name"],
                        "input": {},
                    },
                })
            if state["started"] and state["pending_arguments"]:
                arguments_delta = state["pending_arguments"]
                state["pending_arguments"] = ""
                state["arguments"] += arguments_delta
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state["block_index"],
                    "delta": {"type": "input_json_delta", "partial_json": arguments_delta},
                })
        if not emitted_payload and not choice.get("finish_reason"):
            yield _sse("ping", {"type": "ping"})

    if failed:
        yield _sse("message_stop", {"type": "message_stop"})
        return
    async for tail in finalize():
        yield tail
