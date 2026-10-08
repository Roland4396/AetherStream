"""Opt-in time-limited SSE entry point; no provider is enabled automatically."""

from __future__ import annotations

import time
import uuid
from typing import Any

import anyio
from fastapi import Query, Request
from fastapi.responses import JSONResponse, Response


class TimedSSEResponse(Response):
    def __init__(self, response, *, deadline, seconds, trace_id, deps):
        super().__init__(status_code=response.status_code)
        self.raw_headers = list(response.raw_headers)
        self.headers['x-sse-time-limit'] = str(seconds)
        self.response = response
        self.deadline = deadline
        self.trace_id = trace_id
        self.deps = deps

    async def __call__(self, scope, receive, send):
        started = False
        finished = False

        async def tracked_send(message):
            nonlocal started, finished
            if message['type'] == 'http.response.start':
                message = {**message, 'headers': self.raw_headers}
            await send(message)
            if message['type'] == 'http.response.start':
                started = True
            elif message['type'] == 'http.response.body' and not message.get('more_body', False):
                finished = True

        # Cancellation uses the existing DisconnectSafeStreamingResponse cleanup:
        # its generator receives CancelledError and closes the upstream stream.
        try:
            with anyio.move_on_after(max(0, self.deadline - time.monotonic())) as timer:
                await self.response(scope, receive, tracked_send)
            if timer.cancel_called:
                self.deps.log(f'[TRACE {self.trace_id}] timed_sse_deadline forwarding_cancelled=true')
                # End the HTTP stream without fabricating finish_reason or [DONE].
                # Bound this send too, in case the client has stopped reading.
                with anyio.move_on_after(1):
                    if not started:
                        await send({'type': 'http.response.start', 'status': 504, 'headers': []})
                    if not finished:
                        await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        finally:
            self.deps.active_stream_registry.release_trace(self.trace_id)


def register_routes(app, *, handler, deps: Any):
    async def timed_chat_completions(
        request: Request,
        duration_seconds: float = Query(default=290, gt=0, le=600),
    ):
        deadline = time.monotonic() + duration_seconds
        trace_id = request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
        request.state.trace_id = trace_id
        handed_off = False
        deps.log(f'[TRACE {trace_id}] timed_sse_start seconds={duration_seconds}')
        try:
            with anyio.fail_after(duration_seconds):
                try:
                    payload = await request.json()
                except ValueError:
                    return JSONResponse({'error': {'type': 'invalid_request', 'message': 'Expected JSON body'}}, status_code=400)
                if not isinstance(payload, dict) or payload.get('stream') is not True:
                    return JSONResponse(
                        {'error': {'type': 'invalid_request', 'message': 'Timed SSE requires stream=true'}},
                        status_code=400,
                    )
                # Same Request preserves auth, model routing and cached JSON.
                response = await handler(request, deps)
            if not hasattr(response, 'body_iterator'):
                return response
            wrapped = TimedSSEResponse(
                response, deadline=deadline, seconds=duration_seconds, trace_id=trace_id, deps=deps,
            )
            handed_off = True
            return wrapped
        except TimeoutError:
            deps.log(f'[TRACE {trace_id}] timed_sse_deadline phase=before_response')
            return JSONResponse(
                {'error': {'type': 'stream_deadline_exceeded', 'message': 'SSE time limit reached'}},
                status_code=504,
            )
        finally:
            if not handed_off:
                deps.active_stream_registry.release_trace(trace_id)

    app.post('/v1/chat/completions/timed')(timed_chat_completions)
