import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable
from urllib.parse import urlsplit

import httpx

from aetherstream.upstreams.route_logging import format_account_pool_route

from aetherstream.features.terminal_tool import (
    TERMINAL_TOOL_NAME,
    convert_openai_chat_tools_to_gemini,
)


@dataclass
class GeminiGenerateContentConfig:
    base_url: str
    api_key: str
    include_thoughts: bool
    heartbeat_interval: int
    max_retries: int
    retry_delay: float
    timeout: httpx.Timeout


@dataclass
class GeminiGenerateContentDeps:
    log: Callable[[str], None]
    save_request_log: Callable[..., None]
    build_openai_sse_error: Callable[[int, str, str], bytes]


def _fmt_elapsed_ms(start: float, end: float | None = None) -> str:
    return f"{((end or time.perf_counter()) - start) * 1000:.1f}ms"


def _build_gemini_prime_url(base_url: str) -> str | None:
    try:
        parts = urlsplit(base_url)
    except Exception:
        return None

    host = (parts.hostname or "").lower()
    hosts = {h.strip().lower() for h in os.environ.get("GEMINI_PRIME_HOSTS", "").split(",") if h.strip()}
    if parts.scheme != "https" or not hosts or host not in hosts:
        return None

    return f"{parts.scheme}://{parts.netloc}/"


async def _prime_gemini_connection(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    deps: GeminiGenerateContentDeps,
) -> None:
    prime_url = _build_gemini_prime_url(base_url)
    if not prime_url:
        return

    try:
        prime_t0 = time.perf_counter()
        response = await client.head(prime_url)
        deps.log(
            f"Gemini prime HEAD status={response.status_code} "
            f"elapsed={(time.perf_counter() - prime_t0) * 1000:.1f}ms"
        )
    except Exception as e:
        deps.log(f"Gemini prime HEAD error: {type(e).__name__}: {e}")


def _build_gemini_headers(model: str, config: GeminiGenerateContentConfig) -> dict:
    return {
        "Content-Type": "application/json",
        "x-goog-api-key": config.api_key,
        "User-Agent": f"StreamProxy/1.0 Gemini/{model}",
    }


def _convert_openai_to_gemini(openai_request: dict, *, include_thoughts: bool) -> dict:
    messages = openai_request.get("messages", [])
    system_parts = []
    contents = []

    tool_names_by_id: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            call_id = str(tool_call.get("id") or "").strip()
            name = str(function.get("name") or "").strip()
            if call_id and name:
                tool_names_by_id[call_id] = name

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "\n".join(text_parts)

        if role in {"system", "developer"}:
            system_parts.append(content)
        elif role in {"tool", "function"}:
            call_id = str(msg.get("tool_call_id") or msg.get("toolCallId") or "").strip()
            name = str(msg.get("name") or tool_names_by_id.get(call_id) or "").strip()
            if not name:
                continue
            if isinstance(content, dict):
                response_payload = content
            else:
                try:
                    response_payload = json.loads(content) if isinstance(content, str) else {"output": content}
                except (TypeError, ValueError):
                    response_payload = {"output": str(content)}
            if not isinstance(response_payload, dict):
                response_payload = {"output": response_payload}
            function_response: dict = {
                "name": name,
                "response": response_payload,
            }
            if call_id:
                function_response["id"] = call_id
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": function_response}],
            })
        else:
            gemini_role = "model" if role == "assistant" else "user"
            parts = [{"text": content}] if content else []
            if role == "assistant":
                for tool_call in msg.get("tool_calls", []) or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = str(function.get("name") or "").strip()
                    if not name:
                        continue
                    arguments = function.get("arguments", "")
                    if isinstance(arguments, dict):
                        args = arguments
                    else:
                        try:
                            args = json.loads(arguments or "{}")
                        except (TypeError, ValueError):
                            args = {"raw_arguments": str(arguments or "")}
                    if not isinstance(args, dict):
                        args = {"value": args}
                    function_call: dict = {"name": name, "args": args}
                    call_id = str(tool_call.get("id") or "").strip()
                    if call_id:
                        function_call["id"] = call_id
                    parts.append({"functionCall": function_call})
            contents.append({
                "role": gemini_role,
                "parts": parts or [{"text": ""}],
            })

    gemini_request = {
        "contents": contents,
        "generationConfig": {},
        "safetySettings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        ],
    }

    if include_thoughts:
        gemini_request["generationConfig"]["thinkingConfig"] = {
            "includeThoughts": True,
            "thinkingLevel": "HIGH",
        }

    if system_parts:
        gemini_request["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }

    function_declarations = convert_openai_chat_tools_to_gemini(openai_request.get("tools"))
    if function_declarations:
        gemini_request["tools"] = [{"functionDeclarations": function_declarations}]
        tool_choice = openai_request.get("tool_choice")
        function_calling_config: dict[str, Any] = {"mode": "AUTO"}
        if isinstance(tool_choice, str):
            function_calling_config["mode"] = {
                "auto": "AUTO",
                "none": "NONE",
                "required": "ANY",
            }.get(tool_choice, "AUTO")
        elif isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if tool_choice.get("type") == "function" and isinstance(function, dict) and function.get("name"):
                function_calling_config = {
                    "mode": "ANY",
                    "allowedFunctionNames": [function["name"]],
                }
        gemini_request["toolConfig"] = {"functionCallingConfig": function_calling_config}

    if "temperature" in openai_request:
        gemini_request["generationConfig"]["temperature"] = openai_request["temperature"]
    if "top_p" in openai_request:
        gemini_request["generationConfig"]["topP"] = openai_request["top_p"]
    if "max_tokens" in openai_request:
        gemini_request["generationConfig"]["maxOutputTokens"] = openai_request["max_tokens"]

    return gemini_request


def _extract_text_from_gemini_chunk(data: dict) -> tuple[str, str, str | None]:
    visible_text_parts: list[str] = []
    thought_text_parts: list[str] = []
    finish_reason = None

    candidates = data.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            text = part.get("text", "")
            if not text:
                continue
            if part.get("thought"):
                thought_text_parts.append(text)
            else:
                visible_text_parts.append(text)

        finish_reason = candidate.get("finishReason")

    return "".join(visible_text_parts), "".join(thought_text_parts), finish_reason


async def _request_gemini_generate_content(
    *,
    model: str,
    openai_request: dict,
    config: GeminiGenerateContentConfig,
    deps: GeminiGenerateContentDeps,
    trace_id: str,
) -> dict:
    """All native Gemini requests use generateContent, never streamGenerateContent."""
    payload = _convert_openai_to_gemini(
        openai_request, include_thoughts=config.include_thoughts,
    )
    url = f"{config.base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
    headers = _build_gemini_headers(model, config)
    headers["Accept"] = "application/json"
    prefix = f"[TRACE {trace_id}] " if trace_id else ""
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        await _prime_gemini_connection(client=client, base_url=config.base_url, deps=deps)
        deps.log(
            f"{prefix}gemini_nonstream_request_start model={model} "
            "upstream_method=generateContent upstream_stream=false"
        )
        response = await client.post(url, json=payload, headers=headers)
        deps.log(
            f"{prefix}gemini_nonstream_response status={response.status_code} "
            f"{format_account_pool_route(response)} elapsed={_fmt_elapsed_ms(started)}"
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Gemini generateContent returned a non-object JSON response")
        return data


def _gemini_nonstream_fields(data: dict, *, include_thoughts: bool) -> tuple[str, dict, str, list[dict]]:
    text, thoughts, native_finish = _extract_text_from_gemini_chunk(data)
    # Preserve the existing native adapter's include_thoughts selection policy.
    content = thoughts if include_thoughts else text
    meta = data.get("usageMetadata") or {}
    usage = {
        "prompt_tokens": meta.get("promptTokenCount", 0),
        "completion_tokens": meta.get("candidatesTokenCount", 0),
        "total_tokens": meta.get("totalTokenCount", 0),
    } if meta else {}
    tools = []
    terminal_seen = False
    for candidate in data.get("candidates", []) or []:
        for part in (candidate.get("content") or {}).get("parts", []) or []:
            call = part.get("functionCall") or part.get("function_call")
            if not isinstance(call, dict):
                continue
            if call.get("name") == TERMINAL_TOOL_NAME:
                terminal_seen = True
                continue
            args = call.get("args")
            tools.append({
                "index": len(tools),
                "id": str(call.get("id") or f"call_{len(tools)}"),
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False, separators=(",", ":")),
                },
            })
    finish = "length" if native_finish == "MAX_TOKENS" else "stop"
    if tools:
        finish = "tool_calls"
    elif terminal_seen:
        finish = "stop"
    return content, usage, finish, tools


async def forward_gemini_generate_content_stream(
    *,
    model: str,
    openai_request: dict,
    config: GeminiGenerateContentConfig,
    deps: GeminiGenerateContentDeps,
    messages: list | None = None,
    trace_id: str = "",
) -> AsyncGenerator[bytes, None]:
    """Request complete native Gemini JSON, then replay it as OpenAI SSE."""
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    started = time.perf_counter()
    prefix = f"[TRACE {trace_id}] " if trace_id else ""
    finish_status = "unknown"
    full_content = ""
    task: asyncio.Task | None = None
    log_content = ""
    error_type = None

    def emit(delta: dict | None = None, finish_reason: str | None = None, usage: dict | None = None) -> bytes:
        chunk = {
            "id": response_id, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
        }
        if usage:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

    try:
        # Keep the downstream connection alive without leaking partial upstream content.
        yield emit({"role": "assistant"})
        task = asyncio.create_task(_request_gemini_generate_content(
            model=model, openai_request=openai_request, config=config, deps=deps, trace_id=trace_id,
        ))
        interval = max(0.01, float(config.heartbeat_interval))
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                data = await task
                break
            yield emit()

        full_content, usage, finish_reason, tools = _gemini_nonstream_fields(
            data, include_thoughts=config.include_thoughts,
        )
        for offset in range(0, len(full_content), 1200):
            yield emit({"content": full_content[offset:offset + 1200]})
        for tool in tools:
            yield emit({"tool_calls": [tool]})
        yield emit(finish_reason=finish_reason, usage=usage)
        yield b"data: [DONE]\n\n"
        log_content = full_content
        finish_status = f"finish:{finish_reason}"
    except asyncio.CancelledError:
        finish_status = "downstream_cancelled"
        log_content = full_content + "\n[CANCELLED: downstream client disconnected]"
        raise
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        error_text = exc.response.text[:4000]
        error_type = "gemini_upstream_http_error"
        finish_status = f"upstream_http_{status}"
        log_content = f"[ERROR] {error_text}"
        yield deps.build_openai_sse_error(status, error_text, error_type=error_type)
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        error_type = "gemini_proxy_stream_error"
        finish_status = f"exception:{type(exc).__name__}"
        log_content = f"[ERROR] {exc}"
        yield deps.build_openai_sse_error(502, str(exc), error_type=error_type)
        yield b"data: [DONE]\n\n"
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if messages is not None:
            kwargs = dict(stream=True, request_payload=openai_request, trace_id=trace_id)
            if error_type:
                kwargs["error_type"] = error_type
            deps.save_request_log(model, messages, log_content or full_content or "[empty response]", **kwargs)
        deps.log(
            f"{prefix}gemini_nonstream_replay_done model={model} reason={finish_status} "
            f"elapsed={_fmt_elapsed_ms(started)} out_chars={len(full_content)}"
        )


async def collect_gemini_generate_content(
    *,
    model: str,
    openai_request: dict,
    config: GeminiGenerateContentConfig,
    deps: GeminiGenerateContentDeps,
    trace_id: str = "",
) -> tuple[str, dict, str]:
    """Non-stream downstream also uses a genuine generateContent JSON request."""
    data = await _request_gemini_generate_content(
        model=model, openai_request=openai_request, config=config, deps=deps, trace_id=trace_id,
    )
    content, usage, finish, _ = _gemini_nonstream_fields(data, include_thoughts=config.include_thoughts)
    return content, usage, finish
