"""Request-scoped policy, owned by Stream and bound to the Pool-selected channel."""
from __future__ import annotations

import asyncio
import copy
import inspect
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any

import anyio


_POLICY: ContextVar[dict | None] = ContextVar("channel_policy", default=None)
_REQUEST: ContextVar[dict | None] = ContextVar("channel_request", default=None)
_TERMINATION: ContextVar[str | None] = ContextVar("channel_termination", default=None)
MISSING = object()


def current_policy() -> dict | None:
    return _POLICY.get()


def current_request() -> dict | None:
    return _REQUEST.get()


def termination_reason() -> str | None:
    return _TERMINATION.get()


def mark_termination(reason: str) -> None:
    _TERMINATION.set(reason)


def runtime_value(*keys: str) -> Any:
    policy = current_policy()
    if policy is None:
        return MISSING
    value: Any = policy.get("runtime", {})
    # Early-stop has one owner; expose it through legacy helper signatures too.
    if keys and keys[0] == "early_stop":
        value = {"early_stop": policy.get("early_stop", {})}
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


@contextmanager
def policy_scope(policy: dict, request: dict | None = None):
    token = _POLICY.set(copy.deepcopy(policy))
    request_token = _REQUEST.set(request)
    termination_token = _TERMINATION.set(None)
    try:
        yield
    finally:
        _TERMINATION.reset(termination_token)
        _REQUEST.reset(request_token)
        _POLICY.reset(token)


def selected_early_stop(settings: dict) -> None:
    """Apply the final selected account override to this request/task only."""
    policy = current_policy()
    if policy is not None:
        _POLICY.set({**policy, "early_stop": copy.deepcopy(settings)})


def disconnect_on_early_stop() -> bool:
    policy = current_policy()
    return bool(policy is not None and policy.get("early_stop", {}).get("action") == "disconnect")


async def policy_stream(iterator, policy: dict, request: dict | None = None):
    """Keep a policy alive for the *whole* stream, including close/cancellation."""
    with policy_scope(policy, request):
        try:
            async for chunk in iterator:
                yield chunk
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                if inspect.isasyncgen(iterator):
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await iterator.athrow(asyncio.CancelledError())
            raise
        finally:
            with anyio.CancelScope(shield=True):
                close = getattr(iterator, "aclose", None)
                if callable(close):
                    await close()
