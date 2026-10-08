import asyncio
import json
import time
from typing import AsyncGenerator

import httpx

from aetherstream.observability.refusals import AnthropicRefusalDiagnostics

from aetherstream.features.early_stop import is_himodels_upstream
from aetherstream.upstreams.route_logging import format_account_pool_route, scope_stop_detection

from aetherstream.features.terminal_tool import TERMINAL_TOOL_NAME

from .protocol import _extract_anthropic_sse_text
from .transport import _close_upstream_stream, _prime_code_cli_connection, _stream_with_oauth_recovery
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
    refusal_diagnostics = AnthropicRefusalDiagnostics()
    ordinary_tool_seen = False
    suppressed_terminal_blocks: set[int] = set()
    pending_event_line: str | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
            await _prime_code_cli_connection(client=client, url=url, deps=deps, trace_prefix=trace_prefix)
            deps.log(f"{trace_prefix}anthropic_stream_request_start elapsed={deps.fmt_ms(request_t0)}")
            upstream_t0 = time.perf_counter()
            async with _stream_with_oauth_recovery(client=client, url=url, request_data=request_data,
                                                  headers=headers, deps=deps, trace_prefix=trace_prefix) as response:
                deps = scope_stop_detection(deps, response, url, trace_prefix)
                deps.log(
                    f"{trace_prefix}anthropic_stream_headers status={response.status_code} "
                    f"{format_account_pool_route(response)} "
                    f"elapsed={deps.fmt_ms(upstream_t0)}"
                )
                if is_himodels_upstream(response, url):
                    enable_early_stop = True
                    deps.log(f"{trace_prefix}himodels_early_stop=runtime_config")
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
                        continue

                    if line.startswith("event:"):
                        pending_event_line = line
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
                                refusal_diagnostics.observe(data)
                                event_type = data.get("type")
                                block_index = int(data.get("index") or 0)
                                if event_type == "content_block_start":
                                    block = data.get("content_block")
                                    if isinstance(block, dict) and block.get("type") == "tool_use":
                                        tool_name = str(block.get("name") or "")
                                        if tool_name == TERMINAL_TOOL_NAME:
                                            suppressed_terminal_blocks.add(block_index)
                                            finish_status = "terminal_tool"
                                            deps.log(
                                                f"{trace_prefix}terminal_tool protocol=anthropic_messages "
                                                f"after_ordinary={ordinary_tool_seen} index={block_index}"
                                            )
                                            await _close_upstream_stream(
                                                response=response,
                                                client=client,
                                                deps=deps,
                                                trace_prefix=trace_prefix,
                                                label="terminal_tool",
                                                reason="anthropic_terminal_tool",
                                                started_at=request_t0,
                                                line_count=line_count,
                                                data_line_count=data_line_count,
                                                out_chars=len(content_buffer),
                                            )
                                            yield b"event: message_stop\n"
                                            yield b'data: {"type":"message_stop"}\n\n'
                                            return
                                        ordinary_tool_seen = True

                                if (
                                    event_type in {"content_block_delta", "content_block_stop"}
                                    and block_index in suppressed_terminal_blocks
                                ):
                                    pending_event_line = None
                                    continue

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

                                if event_type == "message_stop":
                                    finish_status = "message_stop_local"
                                    yield b"event: message_stop\n"
                                    yield b'data: {"type":"message_stop"}\n\n'
                                    return

                        if pending_event_line:
                            yield f"{pending_event_line}\n".encode()
                            pending_event_line = None
                        yield f"{line}\n\n".encode()
                        continue

                    if pending_event_line:
                        yield f"{pending_event_line}\n".encode()
                        pending_event_line = None
                    yield f"{line}\n".encode()

                finish_status = "stream_end_without_message_stop"
                payload = {
                    "type": "error",
                    "error": {
                        "type": "upstream_stream_incomplete",
                        "message": "Anthropic upstream stream closed without message_stop",
                    },
                }
                yield b"event: error\n"
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                yield b"event: message_stop\n"
                yield b'data: {"type":"message_stop"}\n\n'

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
            f"{refusal_diagnostics.log_suffix()}"
        )
