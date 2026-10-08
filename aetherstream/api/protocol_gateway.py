"""Public protocol adapters backed by the canonical Chat routing pipeline."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from aetherstream.api.chat_routes import chat_completions
from aetherstream.streaming.responses import DisconnectSafeStreamingResponse as StreamingResponse
from aetherstream.transforms.downstream_protocols import (
    anthropic_to_chat_request,
    chat_stream_to_anthropic,
    chat_stream_to_responses,
    responses_to_chat_request,
)


def _clone_json_request(request: Request, payload: dict[str, Any]) -> Request:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    scope = dict(request.scope)
    headers = [
        (name, value)
        for name, value in request.scope.get("headers", [])
        if name.lower() not in {b"content-length", b"content-type"}
    ]
    if request.headers.get("authorization") is None and request.headers.get("x-api-key"):
        headers.append((b"authorization", f"Bearer {request.headers['x-api-key']}".encode("latin-1")))
    headers.extend([
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ])
    scope["headers"] = headers
    consumed = False

    async def receive() -> dict[str, Any]:
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return Request(scope, receive)


def _decode_json_response(response: Response) -> dict[str, Any]:
    try:
        return json.loads(response.body)
    except Exception:
        return {"error": {"type": "proxy_error", "message": response.body.decode(errors="replace")}}


def _chat_json_to_responses(data: dict[str, Any], *, fallback_model: str) -> dict[str, Any]:
    if isinstance(data.get("error"), dict):
        return data
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    output: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })
    tool_calls = message.get("tool_calls", []) or []
    if not tool_calls and isinstance(message.get("function_call"), dict):
        tool_calls = [{"id": f"call_{uuid.uuid4().hex}", "function": message["function_call"]}]
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        output.append({
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "status": "completed",
            "call_id": str(call.get("id") or ""),
            "name": str(function.get("name") or ""),
            "arguments": str(function.get("arguments") or ""),
        })
    raw_usage = data.get("usage") or {}
    usage = {
        "input_tokens": int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0),
        "output_tokens": int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0),
        "total_tokens": int(raw_usage.get("total_tokens") or 0),
    }
    return {
        "id": str(data.get("id") or f"resp_{uuid.uuid4().hex}"),
        "object": "response",
        "created_at": int(data.get("created") or 0),
        "status": "completed",
        "model": str(data.get("model") or fallback_model),
        "output": output,
        "usage": usage,
        "error": None,
    }


def _chat_json_to_anthropic(data: dict[str, Any], *, fallback_model: str) -> dict[str, Any]:
    if isinstance(data.get("error"), dict):
        error = data["error"]
        return {
            "type": "error",
            "error": {
                "type": error.get("type") or "proxy_error",
                "message": error.get("message") or "Proxy error",
            },
        }
    choices = data.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls", []) or []
    if not tool_calls and isinstance(message.get("function_call"), dict):
        tool_calls = [{"id": f"call_{uuid.uuid4().hex}", "function": message["function_call"]}]
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments", "")
        try:
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            parsed_arguments = {"raw_arguments": arguments}
        if not isinstance(parsed_arguments, dict):
            parsed_arguments = {"value": parsed_arguments}
        content.append({
            "type": "tool_use",
            "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex}"),
            "name": str(function.get("name") or ""),
            "input": parsed_arguments,
        })
    raw_usage = data.get("usage") or {}
    finish_reason = str(choice.get("finish_reason") or "stop")
    return {
        "id": str(data.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": str(data.get("model") or fallback_model),
        "content": content,
        "stop_reason": {
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0),
            "output_tokens": int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0),
        },
    }


async def route_protocol_request(request: Request, chat_deps: Any, *, protocol: str) -> Response:
    try:
        payload = await request.json()
    except Exception:
        if protocol == "anthropic":
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON body"}},
                status_code=400,
            )
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": "Invalid JSON body"}},
            status_code=400,
        )
    if not isinstance(payload, dict):
        return JSONResponse({"error": {"type": "invalid_request_error", "message": "JSON body must be an object"}}, status_code=400)

    if protocol == "responses":
        chat_payload = responses_to_chat_request(payload)
    elif protocol == "anthropic":
        chat_payload = anthropic_to_chat_request(payload)
    else:
        raise ValueError(f"Unsupported downstream protocol: {protocol}")

    inner_request = _clone_json_request(request, chat_payload)
    response = await chat_completions(inner_request, chat_deps)
    model = str(payload.get("model") or chat_payload.get("model") or "")
    if isinstance(response, StreamingResponse):
        if protocol == "responses":
            body = chat_stream_to_responses(response.body_iterator, model=model)
        else:
            body = chat_stream_to_anthropic(response.body_iterator, model=model)
        return StreamingResponse(body, status_code=response.status_code, media_type="text/event-stream")

    data = _decode_json_response(response)
    if protocol == "responses":
        converted = _chat_json_to_responses(data, fallback_model=model)
    else:
        converted = _chat_json_to_anthropic(data, fallback_model=model)
    return JSONResponse(converted, status_code=response.status_code)
