"""Streaming responses with deterministic iterator cleanup."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Send


class DisconnectSafeStreamingResponse(StreamingResponse):
    """Close the response iterator even when disconnect interrupts ``send``.

    Starlette cancels its response task when it receives ``http.disconnect``.
    If that cancellation lands while ``send`` is handling a yielded chunk, the
    async iterator is suspended at the yield and is not closed by Starlette.
    Explicitly closing it here propagates shutdown to the upstream HTTP stream.
    """

    async def stream_response(self, send: Send) -> None:
        completed = False
        try:
            await super().stream_response(send)
            completed = True
        finally:
            with anyio.CancelScope(shield=True):
                if not completed and inspect.isasyncgen(self.body_iterator):
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await self.body_iterator.athrow(asyncio.CancelledError())

                close = getattr(self.body_iterator, "aclose", None)
                if callable(close):
                    with suppress(StopAsyncIteration):
                        await close()
