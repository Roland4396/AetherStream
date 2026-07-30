import asyncio
import json
import time
from typing import AsyncGenerator

import httpx

from .protocol import _extract_anthropic_sse_text
from .transport import _close_upstream_stream, _prime_code_cli_connection
from .types import AnthropicMessagesDeps


async def forward_anthropic_messages_stream(
    *,
    url: str,
    request_data: dict,
    headers: dict,
    timeout: httpx.Timeout,
    deps: AnthropicMessagesDeps,
    enable_early_stop: bool = True,
    trace_id: str = "",
) -> AsyncGenerator[bytes, None]:
    """Anthropic SSE 原样透传，保留现有 /v1/messages 路由能力。"""
    request_t0 = time.perf_counter()
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    content_buffer = ""
    first_data_time: float | None = None
    line_count = 0
    data_line_count = 0
    finish_status = "unknown"

    try:
        async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
            await _prime_code_cli_connection(client=client, url=url, deps=deps, trace_prefix=trace_prefix)
            deps.log(f"{trace_prefix}anthropic_stream_request_start elapsed={deps.fmt_ms(request_t0)}")
            upstream_t0 = time.perf_counter()
            async with client.stream("POST", url, json=request_data, headers=headers) as response:
                deps.log(
                    f"{trace_prefix}anthropic_stream_headers status={response.status_code} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if response.status_code != 200:
                    finish_status = f"upstream_http_{response.status_code}"
                    content = await response.aread()
                    error_text = content.decode(errors="replace")[:4000]
                    payload = {
                        "type": "error",
                        "error": {
                            "type": "upstream_http_error",
                            "status": response.status_code,
                            "message": error_text,
                        },
                    }
                    yield b"event: error\n"
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                    yield b"event: message_stop\n"
                    yield b'data: {"type":"message_stop"}\n\n'
                    return

                async for line in response.aiter_lines():
                    line_count += 1
                    if not line:
                        yield b"\n"
                        continue

                    if line.startswith("data: "):
                        data_line_count += 1
                        if first_data_time is None:
                            first_data_time = time.perf_counter()
                            deps.log(
                                f"{trace_prefix}anthropic_stream_first_data "
                                f"elapsed={deps.fmt_ms(request_t0, first_data_time)}"
                            )
                        payload = line[6:].strip()
                        if payload:
                            try:
                                data = json.loads(payload)
                            except json.JSONDecodeError:
                                data = None

                            if isinstance(data, dict):
                                content = _extract_anthropic_sse_text(data)
                                if content:
                                    content_buffer += content
                                    if enable_early_stop and deps.has_stop_tag(content_buffer):
                                        deps.log("Anthropic stream: detected STOP_TAG")
                                        finish_status = "early_stop_tag"
                                        await _close_upstream_stream(
                                            response=response,
                                            client=client,
                                            deps=deps,
                                            trace_prefix=trace_prefix,
                                            label="early_stop",
                                            reason="anthropic_raw_stop_tag",
                                            started_at=request_t0,
                                            line_count=line_count,
                                            data_line_count=data_line_count,
                                            out_chars=len(content_buffer),
                                        )
                                        yield b"event: message_stop\n"
                                        yield b'data: {"type":"message_stop"}\n\n'
                                        return

                                if data.get("type") == "message_stop":
                                    finish_status = "message_stop"
                                    yield f"{line}\n".encode()
                                    return

                    yield f"{line}\n".encode()

                finish_status = "stream_end_without_message_stop"

    except Exception as e:
        finish_status = f"exception:{type(e).__name__}"
        deps.log(f"Anthropic stream forward error: {e}")
        payload = {
            "type": "error",
            "error": {
                "type": "proxy_stream_error",
                "message": str(e),
            },
        }
        yield b"event: error\n"
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
        yield b"event: message_stop\n"
        yield b'data: {"type":"message_stop"}\n\n'
    finally:
        deps.log(
            f"{trace_prefix}anthropic_stream_done reason={finish_status} "
            f"elapsed={deps.fmt_ms(request_t0)} "
            f"first_data={deps.fmt_ms(request_t0, first_data_time) if first_data_time else '-'} "
            f"lines={line_count} data_lines={data_line_count} out_chars={len(content_buffer)}"
        )
