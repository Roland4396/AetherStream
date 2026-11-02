"""The single real upstream shutdown path used by every protocol adapter.

Stopping output is not cancellation. We cancel/await the reader, close the
HTTP response, then close its client. No drain/read-to-end, retry, replacement
request or synthetic completion is performed here.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import anyio

from aetherstream.routing.channel_policy import mark_termination


async def close_upstream_stream(
    *, response: Any, client: Any, deps: Any, trace_prefix: str,
    label: str, reason: str, started_at: float, line_count: int,
    data_line_count: int, out_chars: int, reader_task: asyncio.Task | None = None,
) -> None:
    close_t0 = time.perf_counter()
    # The unconditional resource finalizer must not erase the semantic reason
    # recorded by the stop path immediately before it.
    if label != 'cleanup':
        mark_termination(label)
    response_closed = client_closed = False
    # A downstream ASGI disconnect cancels the enclosing scope. Cleanup must
    # still reach both socket owners before the lifecycle barrier is released.
    with anyio.CancelScope(shield=True):
        if reader_task is not None and reader_task is not asyncio.current_task() and not reader_task.done():
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
        for resource, name in ((response, "response"), (client, "client")):
            if resource is None:
                continue
            try:
                await resource.aclose()
                if name == "response":
                    response_closed = True
                else:
                    client_closed = True
            except Exception as exc:
                # Exception strings can contain provider URLs or credentials.
                deps.log(f"{trace_prefix}{label}_upstream_close_{name}_error reason={reason} type={type(exc).__name__}")
    deps.log(
        f"{trace_prefix}{label}_upstream_close reason={reason} "
        f"response_closed={str(response_closed).lower()} client_closed={str(client_closed).lower()} "
        f"close_elapsed={deps.fmt_ms(close_t0)} elapsed={deps.fmt_ms(started_at)} "
        f"lines={line_count} data_lines={data_line_count} out_chars={out_chars}"
    )


async def close_open_task(task, **kwargs) -> None:
    """Settle a pending header opener, including the done-before-handoff race."""
    if task is None:
        return
    with anyio.CancelScope(shield=True):
        if not task.done():
            task.cancel()
        try:
            client, _context, response = await task
        except (asyncio.CancelledError, Exception):
            return
        await close_upstream_stream(response=response, client=client, **kwargs)
