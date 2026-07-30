import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Callable
from urllib.parse import urlsplit

import httpx


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

    for msg in messages:
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

        if role == "system":
            system_parts.append(content)
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}],
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


async def forward_gemini_generate_content_stream(
    *,
    model: str,
    openai_request: dict,
    config: GeminiGenerateContentConfig,
    deps: GeminiGenerateContentDeps,
    messages: list | None = None,
    trace_id: str = "",
) -> AsyncGenerator[bytes, None]:
    """Gemini 流式直连并转换为 OpenAI SSE。"""
    gemini_request = _convert_openai_to_gemini(
        openai_request,
        include_thoughts=config.include_thoughts,
    )
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    full_response = ""

    url = f"{config.base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse"
    headers = _build_gemini_headers(model, config)
    request_t0 = time.perf_counter()
    first_data_time: float | None = None
    finish_status = "unknown"
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            await _prime_gemini_connection(
                client=client,
                base_url=config.base_url,
                deps=deps,
            )
            deps.log(
                f"{trace_prefix}gemini_stream_request_start "
                f"model={model} elapsed={_fmt_elapsed_ms(request_t0)}"
            )
            upstream_t0 = time.perf_counter()
            async with client.stream("POST", url, json=gemini_request, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}gemini_stream_headers status={response.status_code} "
                    f"elapsed={_fmt_elapsed_ms(upstream_t0)} total={_fmt_elapsed_ms(request_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    error_text = (await response.aread()).decode(errors="replace")[:4000]
                    yield deps.build_openai_sse_error(
                        response.status_code,
                        error_text,
                        error_type="gemini_upstream_http_error",
                    )
                    yield b"data: [DONE]\n\n"
                    if messages is not None:
                        deps.save_request_log(
                            model,
                            messages,
                            f"[ERROR] {error_text}",
                            stream=True,
                            request_payload=openai_request,
                            error_type="gemini_upstream_http_error",
                            trace_id=trace_id,
                        )
                    return

                last_data_time = time.time()
                async for line in response.aiter_lines():
                    now = time.time()
                    if now - last_data_time > config.heartbeat_interval:
                        yield b": heartbeat\n\n"
                        last_data_time = now

                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:]
                    try:
                        data = json.loads(json_str)
                        if first_data_time is None:
                            first_data_time = time.perf_counter()
                            deps.log(
                                f"{trace_prefix}gemini_stream_first_data model={model} "
                                f"elapsed={_fmt_elapsed_ms(request_t0, first_data_time)}"
                            )
                        text, thought_text, finish_reason = _extract_text_from_gemini_chunk(data)
                        last_data_time = time.time()

                        emitted_text = thought_text if config.include_thoughts else text
                        if emitted_text:
                            full_response += emitted_text
                            chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": emitted_text},
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

                        if finish_reason:
                            deps.log(
                                "Gemini stream finish "
                                f"reason={finish_reason} visible_len={len(text)} "
                                f"thought_len={len(thought_text)}"
                            )
                            finish_status = f"finish:{finish_reason}"
                        if finish_reason:
                            end_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }],
                            }
                            yield f"data: {json.dumps(end_chunk)}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            if messages is not None:
                                deps.save_request_log(
                                    model,
                                    messages,
                                    full_response,
                                    stream=True,
                                    request_payload=openai_request,
                                    trace_id=trace_id,
                                )
                            return
                    except json.JSONDecodeError:
                        continue

                finish_status = "stream_end_without_finish_reason"
                end_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(end_chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                if messages is not None:
                    deps.save_request_log(
                        model,
                        messages,
                        full_response,
                        stream=True,
                        request_payload=openai_request,
                        trace_id=trace_id,
                    )
                return
    except Exception as e:
        finish_status = f"exception:{type(e).__name__}"
        yield deps.build_openai_sse_error(
            502,
            str(e),
            error_type="gemini_proxy_stream_error",
        )
        yield b"data: [DONE]\n\n"
        if messages is not None:
            deps.save_request_log(
                model,
                messages,
                f"[ERROR] {e}",
                stream=True,
                request_payload=openai_request,
                error_type="gemini_proxy_stream_error",
                trace_id=trace_id,
            )
        return
    finally:
        deps.log(
            f"{trace_prefix}gemini_stream_done model={model} reason={finish_status} "
            f"elapsed={_fmt_elapsed_ms(request_t0)} "
            f"first_data={_fmt_elapsed_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"out_chars={len(full_response)}"
        )


async def collect_gemini_generate_content(
    *,
    model: str,
    openai_request: dict,
    config: GeminiGenerateContentConfig,
    deps: GeminiGenerateContentDeps,
    trace_id: str = "",
) -> tuple[str, dict, str]:
    """Gemini 非流请求：内部走流式收集。"""
    gemini_request = _convert_openai_to_gemini(
        openai_request,
        include_thoughts=config.include_thoughts,
    )
    url = f"{config.base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse"
    headers = _build_gemini_headers(model, config)
    full_content = ""
    usage = {}
    finish_reason = "stop"
    request_t0 = time.perf_counter()
    first_data_time: float | None = None
    finish_status = "unknown"
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            await _prime_gemini_connection(
                client=client,
                base_url=config.base_url,
                deps=deps,
            )
            deps.log(
                f"{trace_prefix}gemini_collect_request_start "
                f"model={model} elapsed={_fmt_elapsed_ms(request_t0)}"
            )
            upstream_t0 = time.perf_counter()
            async with client.stream("POST", url, json=gemini_request, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}gemini_collect_headers status={response.status_code} "
                    f"elapsed={_fmt_elapsed_ms(upstream_t0)} total={_fmt_elapsed_ms(request_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    content = await response.aread()
                    raise RuntimeError(
                        f"Gemini upstream error: {response.status_code} - "
                        f"{content.decode(errors='replace')}"
                    )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:]
                    try:
                        data = json.loads(json_str)
                        if first_data_time is None:
                            first_data_time = time.perf_counter()
                            deps.log(
                                f"{trace_prefix}gemini_collect_first_data model={model} "
                                f"elapsed={_fmt_elapsed_ms(request_t0, first_data_time)}"
                            )
                        text, thought_text, fr = _extract_text_from_gemini_chunk(data)
                        emitted_text = thought_text if config.include_thoughts else text
                        if emitted_text:
                            full_content += emitted_text

                        if "usageMetadata" in data:
                            meta = data["usageMetadata"]
                            usage = {
                                "prompt_tokens": meta.get("promptTokenCount", 0),
                                "completion_tokens": meta.get("candidatesTokenCount", 0),
                                "total_tokens": meta.get("totalTokenCount", 0),
                            }

                        if fr:
                            deps.log(
                                "Gemini non-stream finish "
                                f"reason={fr} visible_len={len(text)} "
                                f"thought_len={len(thought_text)} "
                                f"collected_len={len(full_content)}"
                            )
                            finish_status = f"finish:{fr}"
                            finish_reason = "stop"
                            return full_content, usage, finish_reason
                    except json.JSONDecodeError:
                        continue

                finish_status = "stream_end_without_finish_reason"
                return full_content, usage, finish_reason
    finally:
        deps.log(
            f"{trace_prefix}gemini_collect_done model={model} reason={finish_status} "
            f"elapsed={_fmt_elapsed_ms(request_t0)} "
            f"first_data={_fmt_elapsed_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"out_chars={len(full_content)}"
        )
