"""ASGI-level request/response wiretap middleware."""

import asyncio
import time
import uuid
from typing import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ASGIWiretapMiddleware:
    """Lowest-level ASGI wiretap for downstream disconnect/cancel diagnosis.

    This deliberately wraps send/receive instead of business generators, so it can
    show whether cancellation comes from an ASGI http.disconnect event, from a
    send() failure, or only from task cancellation after the client socket closes.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        log: Callable[[str], None],
        fmt_ms: Callable[..., str],
        release_trace: Callable[[str], None] | None = None,
    ):
        self.app = app
        self._log = log
        self._fmt_ms = fmt_ms
        self._release_trace = release_trace

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get('type') != 'http':
            await self.app(scope, receive, send)
            return

        path = str(scope.get('path') or '')
        if path != '/v1/chat/completions':
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get('headers') or [])
        state = scope.setdefault('state', {})
        trace_id = (
            str(state.get('trace_id') or '').strip()
            or headers.get(b'x-request-id', b'').decode(errors='replace')
            or uuid.uuid4().hex[:8]
        )
        state['trace_id'] = trace_id
        t0 = time.perf_counter()
        log = self._log
        fmt_ms = self._fmt_ms
        client = scope.get('client') or ('?', 0)
        method = scope.get('method') or '?'
        body_in = 0
        request_chunks = 0
        response_status = None
        response_body_bytes = 0
        response_body_chunks = 0
        response_more_true = 0
        first_body_at: float | None = None
        last_body_at: float | None = None
        disconnect_seen = False
        disconnect_at: float | None = None
        send_error = None
        receive_error = None
        done_reason = 'unknown'

        def since() -> str:
            return fmt_ms(t0)

        log(
            f"[TRACE {trace_id}] asgi_start method={method} path={path} "
            f"client={client[0]}:{client[1]}"
        )

        async def tapped_receive() -> Message:
            nonlocal body_in, request_chunks, disconnect_seen, disconnect_at, receive_error
            try:
                message = await receive()
            except Exception as e:
                receive_error = f"{type(e).__name__}: {e}"
                log(f"[TRACE {trace_id}] asgi_receive_error elapsed={since()} err={receive_error}")
                raise

            msg_type = message.get('type')
            if msg_type == 'http.request':
                chunk = message.get('body') or b''
                body_in += len(chunk)
                request_chunks += 1
                if request_chunks <= 3 or not message.get('more_body', False):
                    log(
                        f"[TRACE {trace_id}] asgi_receive_request "
                        f"chunk={request_chunks} bytes={len(chunk)} total={body_in} "
                        f"more={bool(message.get('more_body', False))} elapsed={since()}"
                    )
            elif msg_type == 'http.disconnect':
                disconnect_seen = True
                disconnect_at = time.perf_counter()
                log(
                    f"[TRACE {trace_id}] asgi_receive_disconnect "
                    f"elapsed={fmt_ms(t0, disconnect_at)} "
                    f"after_resp_bytes={response_body_bytes} chunks={response_body_chunks}"
                )
            else:
                log(f"[TRACE {trace_id}] asgi_receive type={msg_type} elapsed={since()}")
            return message

        async def tapped_send(message: Message) -> None:
            nonlocal response_status, response_body_bytes, response_body_chunks
            nonlocal response_more_true, first_body_at, last_body_at, send_error
            msg_type = message.get('type')
            try:
                if msg_type == 'http.response.start':
                    response_status = message.get('status')
                    log(
                        f"[TRACE {trace_id}] asgi_send_start "
                        f"status={response_status} elapsed={since()}"
                    )
                elif msg_type == 'http.response.body':
                    body = message.get('body') or b''
                    more = bool(message.get('more_body', False))
                    now = time.perf_counter()
                    if first_body_at is None:
                        first_body_at = now
                        log(
                            f"[TRACE {trace_id}] asgi_send_first_body "
                            f"bytes={len(body)} more={more} elapsed={fmt_ms(t0, now)}"
                        )
                    response_body_chunks += 1
                    response_body_bytes += len(body)
                    last_body_at = now
                    if more:
                        response_more_true += 1
                    if (
                        response_body_chunks <= 5
                        or not more
                        or response_body_chunks % 100 == 0
                        or len(body) == 0
                    ):
                        log(
                            f"[TRACE {trace_id}] asgi_send_body "
                            f"chunk={response_body_chunks} bytes={len(body)} "
                            f"total={response_body_bytes} more={more} elapsed={fmt_ms(t0, now)}"
                        )
                await send(message)
            except Exception as e:
                send_error = f"{type(e).__name__}: {e}"
                log(
                    f"[TRACE {trace_id}] asgi_send_error type={msg_type} "
                    f"elapsed={since()} err={send_error} "
                    f"resp_bytes={response_body_bytes} chunks={response_body_chunks}"
                )
                raise

        try:
            await self.app(scope, tapped_receive, tapped_send)
            done_reason = 'app_returned'
        except asyncio.CancelledError:
            done_reason = 'task_cancelled'
            log(
                f"[TRACE {trace_id}] asgi_task_cancelled elapsed={since()} "
                f"disconnect_seen={disconnect_seen} resp_bytes={response_body_bytes} "
                f"chunks={response_body_chunks}"
            )
            raise
        except Exception as e:
            done_reason = f"exception:{type(e).__name__}"
            log(
                f"[TRACE {trace_id}] asgi_app_exception elapsed={since()} "
                f"err={type(e).__name__}: {e} disconnect_seen={disconnect_seen} "
                f"send_error={send_error} receive_error={receive_error}"
            )
            raise
        finally:
            if self._release_trace is not None:
                try:
                    self._release_trace(trace_id)
                except Exception as release_error:
                    log(
                        f"[TRACE {trace_id}] asgi_release_trace_error "
                        f"err={type(release_error).__name__}: {release_error}"
                    )
            idle_after_last = '-'
            if last_body_at is not None:
                idle_after_last = f"{(time.perf_counter() - last_body_at) * 1000:.1f}ms"
            log(
                f"[TRACE {trace_id}] asgi_done reason={done_reason} status={response_status} "
                f"elapsed={since()} request_bytes={body_in} request_chunks={request_chunks} "
                f"resp_bytes={response_body_bytes} resp_chunks={response_body_chunks} "
                f"resp_more_true={response_more_true} disconnect_seen={disconnect_seen} "
                f"disconnect_at={fmt_ms(t0, disconnect_at) if disconnect_at else '-'} "
                f"idle_after_last_body={idle_after_last} send_error={send_error or '-'} "
                f"receive_error={receive_error or '-'}"
            )
