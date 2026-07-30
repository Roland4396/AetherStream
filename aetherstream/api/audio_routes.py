from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from aetherstream.api.dependencies import RouteDependencies, build_route_dependencies
from aetherstream.streaming.responses import DisconnectSafeStreamingResponse as StreamingResponse


AUDIO_DEPENDENCY_NAMES = (
    "TTS_UPSTREAM_URL",
    "TTS_MAX_REQUEST_BYTES",
    "build_tts_http_client",
    "log",
)


def _error(message: str, *, request_id: str, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "server_error" if status_code >= 500 else "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
        status_code=status_code,
        headers={"Cache-Control": "no-store", "X-Request-Id": request_id},
    )


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-client-request-id", "").strip()
    if supplied and len(supplied) <= 512 and supplied.isascii():
        return supplied
    return uuid.uuid4().hex


def _upstream_headers(request: Request, request_id: str) -> dict[str, str]:
    headers = {
        "Accept": request.headers.get("accept", "application/octet-stream"),
        "X-Client-Request-Id": request_id,
    }
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _response_headers(response: httpx.Response, request_id: str) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-store",
        "X-Request-Id": response.headers.get("x-request-id", request_id),
    }
    voice = response.headers.get("x-tts-voice")
    if voice:
        headers["X-TTS-Voice"] = voice
    return headers


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return
        await asyncio.sleep(0)


async def _send_until_disconnect(
    client: httpx.AsyncClient,
    upstream_request: httpx.Request,
    downstream_request: Request,
) -> httpx.Response | None:
    send_task = asyncio.create_task(client.send(upstream_request, stream=True))
    disconnect_task = asyncio.create_task(_wait_for_disconnect(downstream_request))
    try:
        done, _ = await asyncio.wait(
            (send_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if send_task in done:
            return send_task.result()
        send_task.cancel()
        with suppress(asyncio.CancelledError):
            await send_task
        return None
    finally:
        if not send_task.done():
            send_task.cancel()
            with suppress(asyncio.CancelledError):
                await send_task
        disconnect_task.cancel()
        with suppress(asyncio.CancelledError):
            await disconnect_task


async def _send_upstream(
    *,
    request: Request,
    deps: RouteDependencies,
    method: str,
    path: str,
    body: bytes = b"",
):
    request_id = _request_id(request)
    client = deps.build_tts_http_client()
    target = f"{str(deps.TTS_UPSTREAM_URL).rstrip('/')}{path}"
    upstream_request = client.build_request(
        method,
        target,
        content=body or None,
        headers=_upstream_headers(request, request_id),
    )
    try:
        upstream = await _send_until_disconnect(client, upstream_request, request)
        if upstream is None:
            await client.aclose()
            deps.log(
                f"[TTS {request_id}] downstream_cancelled stage=upstream_headers"
            )
            return _error(
                "client disconnected",
                request_id=request_id,
                status_code=499,
                code="client_disconnected",
            )
    except asyncio.CancelledError:
        await client.aclose()
        deps.log(f"[TTS {request_id}] downstream_cancelled stage=handler_task")
        raise
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        await client.aclose()
        deps.log(f"[TTS {request_id}] upstream_unavailable type={type(exc).__name__}")
        return _error(
            "TTS runtime is unavailable",
            request_id=request_id,
            status_code=503,
            code="tts_unavailable",
        )

    headers = _response_headers(upstream, request_id)
    content_type = upstream.headers.get("content-type", "application/octet-stream")
    if upstream.status_code >= 400:
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
            await client.aclose()
        deps.log(
            f"[TTS {request_id}] upstream_rejected status={upstream.status_code} bytes={len(content)}"
        )
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=content_type.split(";", 1)[0],
            headers=headers,
        )

    async def body_iterator() -> AsyncIterator[bytes]:
        sent = 0
        completed = False
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    sent += len(chunk)
                    yield chunk
            completed = True
        finally:
            await upstream.aclose()
            await client.aclose()
            deps.log(
                f"[TTS {request_id}] stream_closed bytes={sent} completed={str(completed).lower()}"
            )

    deps.log(f"[TTS {request_id}] upstream_connected status={upstream.status_code}")
    return StreamingResponse(
        body_iterator(),
        status_code=upstream.status_code,
        media_type=content_type.split(";", 1)[0],
        headers=headers,
    )


async def create_speech(request: Request, deps: RouteDependencies):
    request_id = _request_id(request)
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return _error(
            "Content-Type must be application/json",
            request_id=request_id,
            status_code=415,
            code="invalid_content_type",
        )
    body = await request.body()
    if len(body) > int(deps.TTS_MAX_REQUEST_BYTES):
        return _error(
            "TTS request body is too large",
            request_id=request_id,
            status_code=413,
            code="request_too_large",
        )
    deps.log(
        f"[TTS {request_id}] request_received bytes={len(body)} "
        f"authorization={'present' if request.headers.get('authorization') else 'missing'}"
    )
    return await _send_upstream(
        request=request,
        deps=deps,
        method="POST",
        path="/v1/audio/speech",
        body=body,
    )


async def voices(request: Request, deps: RouteDependencies):
    return await _send_upstream(
        request=request,
        deps=deps,
        method="GET",
        path="/v1/audio/voices",
    )


def register_routes(app, ctx: dict[str, Any]) -> RouteDependencies:
    deps = build_route_dependencies(ctx, AUDIO_DEPENDENCY_NAMES)

    async def create_speech_endpoint(request: Request):
        return await create_speech(request, deps)

    async def voices_endpoint(request: Request):
        return await voices(request, deps)

    app.post("/v1/audio/speech")(create_speech_endpoint)
    app.get("/v1/audio/voices")(voices_endpoint)
    return deps
