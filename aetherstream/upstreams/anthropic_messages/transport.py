import asyncio
import os
import time
from urllib.parse import urlsplit

import httpx

from .types import AnthropicMessagesDeps


def _is_async_generator_close_race(exc: BaseException) -> bool:
    """Return True for benign async-generator shutdown races.

    Starlette/FastAPI may close a streaming response when the downstream client
    disconnects while our upstream reader is still inside httpx/httpcore's async
    iterator.  In that window Python can surface RuntimeError messages such as
    ``athrow(): asynchronous generator is already running`` or
    ``async generator ignored GeneratorExit``.  These are lifecycle/cancel
    events, not upstream provider failures, and we must not try to yield an
    error chunk after the downstream side is already gone.
    """
    if isinstance(exc, GeneratorExit):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return (
        "athrow(): asynchronous generator is already running" in msg
        or "async generator ignored GeneratorExit" in msg
    )


async def _close_upstream_stream(
    *,
    response: httpx.Response | None,
    client: httpx.AsyncClient | None,
    deps: "AnthropicMessagesDeps",
    trace_prefix: str,
    label: str,
    reason: str,
    started_at: float,
    line_count: int,
    data_line_count: int,
    out_chars: int,
    reader_task: asyncio.Task | None = None,
) -> None:
    close_t0 = time.perf_counter()
    if reader_task is not None and not reader_task.done():
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    response_closed = False
    client_closed = False
    if response is not None:
        try:
            await response.aclose()
            response_closed = True
        except Exception as e:
            deps.log(
                f"{trace_prefix}{label}_upstream_close_response_error "
                f"reason={reason} type={type(e).__name__} err={e}"
            )
    if client is not None:
        try:
            await client.aclose()
            client_closed = True
        except Exception as e:
            deps.log(
                f"{trace_prefix}{label}_upstream_close_client_error "
                f"reason={reason} type={type(e).__name__} err={e}"
            )
    deps.log(
        f"{trace_prefix}{label}_upstream_close "
        f"reason={reason} response_closed={str(response_closed).lower()} "
        f"client_closed={str(client_closed).lower()} close_elapsed={deps.fmt_ms(close_t0)} "
        f"elapsed={deps.fmt_ms(started_at)} lines={line_count} "
        f"data_lines={data_line_count} out_chars={out_chars}"
    )


def _build_code_cli_prime_url(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except Exception:
        return None

    host = (parts.hostname or "").lower()
    # Optional connection priming for selected hosts only.  Disabled by default
    # for public/open-source use; enable by setting CLAUDE_PRIME_HOSTS.
    hosts = {h.strip().lower() for h in os.environ.get("CLAUDE_PRIME_HOSTS", "").split(",") if h.strip()}
    if parts.scheme != "https" or not hosts or host not in hosts:
        return None

    return f"{parts.scheme}://{parts.netloc}/"


async def _prime_code_cli_connection(
    *,
    client: httpx.AsyncClient,
    url: str,
    deps: AnthropicMessagesDeps,
    trace_prefix: str = "",
) -> None:
    prime_url = _build_code_cli_prime_url(url)
    if not prime_url:
        return

    try:
        prime_t0 = time.perf_counter()
        response = await client.head(
            prime_url,
            headers={
                "Connection": "keep-alive",
                "User-Agent": "Bun/1.3.13",
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br, zstd",
            },
        )
        deps.log(
            f"{trace_prefix}claude_prime_head status={response.status_code} "
            f"elapsed={deps.fmt_ms(prime_t0)}"
        )
    except Exception as e:
        deps.log(f"{trace_prefix}claude_prime_head_error {type(e).__name__}: {e}")
