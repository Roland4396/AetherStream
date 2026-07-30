"""Keep long-running non-stream JSON responses alive without changing their body format."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from starlette.responses import JSONResponse, Response
from starlette.types import Message, Receive, Scope, Send


class JSONKeepaliveResponse(Response):
    """Run a response-producing coroutine and emit JSON-safe whitespace while waiting.

    JSON permits leading whitespace, so non-stream clients can continue using
    ``response.json()`` while proxies receive a body chunk at least once per
    interval. Fast responses are sent unchanged, including their HTTP status.
    Once a heartbeat has started the status is necessarily committed as 200;
    any later error is still delivered using its original JSON error body.
    """

    media_type = 'application/json'

    def __init__(
        self,
        operation: Callable[[], Awaitable[Response]],
        *,
        interval: float = 10.0,
        trace_id: str = '',
        log: Callable[[str], None] | None = None,
    ) -> None:
        # Response.__init__ prepares headers for FastAPI's response handling.
        # __call__ sends either the operation's original response or our own
        # chunked response, so no static body is used here.
        super().__init__(content=b'', media_type=self.media_type)
        self._operation = operation
        self._interval = max(0.05, float(interval))
        self._trace_id = trace_id
        self._log = log or (lambda _message: None)

    def _prefix(self) -> str:
        return f'[TRACE {self._trace_id}] ' if self._trace_id else ''

    async def _wait_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message.get('type') == 'http.disconnect':
                return

    @staticmethod
    def _fallback_error(error: Exception) -> JSONResponse:
        return JSONResponse(
            {
                'error': {
                    'message': str(error),
                    'type': 'proxy_error',
                }
            },
            status_code=500,
        )

    async def _cancel_task(self, task: asyncio.Task[Any]) -> None:
        if task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started_at = time.perf_counter()
        operation_task = asyncio.create_task(self._operation())
        disconnect_task = asyncio.create_task(self._wait_for_disconnect(receive))
        heartbeat_count = 0
        response_started = False
        response: Response | None = None

        try:
            while not operation_task.done():
                done, _pending = await asyncio.wait(
                    {operation_task, disconnect_task},
                    timeout=self._interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in done:
                    self._log(
                        f'{self._prefix()}nonstream_keepalive_downstream_disconnect '
                        f'elapsed={(time.perf_counter() - started_at):.3f}s '
                        f'keepalives={heartbeat_count}'
                    )
                    await self._cancel_task(operation_task)
                    return
                if operation_task in done:
                    break

                if not response_started:
                    await send(
                        {
                            'type': 'http.response.start',
                            'status': 200,
                            'headers': [
                                (b'content-type', b'application/json; charset=utf-8'),
                                (b'cache-control', b'no-cache, no-store'),
                                (b'x-accel-buffering', b'no'),
                            ],
                        }
                    )
                    response_started = True

                heartbeat_count += 1
                await send({'type': 'http.response.body', 'body': b' \n', 'more_body': True})
                if heartbeat_count <= 3 or heartbeat_count % 6 == 0:
                    self._log(
                        f'{self._prefix()}nonstream_keepalive_emit '
                        f'count={heartbeat_count} elapsed={(time.perf_counter() - started_at):.3f}s'
                    )

            try:
                response = operation_task.result()
            except Exception as error:
                self._log(
                    f'{self._prefix()}nonstream_keepalive_operation_error '
                    f'error={type(error).__name__}: {error}'
                )
                response = self._fallback_error(error)

            await self._cancel_task(disconnect_task)

            if not response_started:
                await response(scope, receive, send)
                return

            body = getattr(response, 'body', None)
            if not isinstance(body, bytes):
                error = TypeError(
                    f'non-stream keepalive operation returned unsupported response '
                    f'{type(response).__name__}'
                )
                self._log(f'{self._prefix()}{error}')
                body = json.dumps(
                    {
                        'error': {
                            'message': str(error),
                            'type': 'proxy_error',
                        }
                    },
                    ensure_ascii=False,
                    separators=(',', ':'),
                ).encode()

            if response.status_code != 200:
                self._log(
                    f'{self._prefix()}nonstream_keepalive_status_already_committed '
                    f'original_status={response.status_code} keepalives={heartbeat_count}'
                )
            await send({'type': 'http.response.body', 'body': body, 'more_body': False})
            if response.background is not None:
                await response.background()
            self._log(
                f'{self._prefix()}nonstream_keepalive_done '
                f'elapsed={(time.perf_counter() - started_at):.3f}s '
                f'keepalives={heartbeat_count} body_bytes={len(body)}'
            )
        except asyncio.CancelledError:
            await self._cancel_task(operation_task)
            await self._cancel_task(disconnect_task)
            self._log(
                f'{self._prefix()}nonstream_keepalive_cancelled '
                f'elapsed={(time.perf_counter() - started_at):.3f}s '
                f'keepalives={heartbeat_count}'
            )
            raise
        finally:
            await self._cancel_task(disconnect_task)
            if response_started and not operation_task.done():
                await self._cancel_task(operation_task)


def keepalive_json_response(
    operation: Callable[[], Awaitable[Response]],
    *,
    interval: float = 10.0,
    trace_id: str = '',
    log: Callable[[str], None] | None = None,
) -> JSONKeepaliveResponse:
    return JSONKeepaliveResponse(
        operation,
        interval=interval,
        trace_id=trace_id,
        log=log,
    )
