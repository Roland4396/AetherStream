from __future__ import annotations

from typing import Any
from dataclasses import replace
from urllib.parse import urlsplit


def _log_value(value: Any, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return "_".join(text.split())[:80]


def format_account_pool_route(response: Any) -> str:
    """Format the non-sensitive account selected by account-pool."""
    headers = getattr(response, "headers", {})
    account_id = _log_value(headers.get("x-account-pool-id"), "direct")
    pool_status = _log_value(headers.get("x-account-pool-status"), "-")
    return f"account={account_id} pool_status={pool_status}"


def scope_stop_detection(deps: Any, response: Any, url: str, trace_prefix: str) -> Any:
    """Disable proxy tag detection only for the actual selected gproxy upstream.

    Copy the request dependencies, never mutate the global matcher or another
    request. Pool response headers identify the final account after fallback.
    """
    account = str(getattr(response, 'headers', {}).get('x-account-pool-id') or '').strip().lower()
    target = urlsplit(url)
    direct = target.hostname == 'gproxy' or (
        target.hostname in {'127.0.0.1', 'localhost', '::1'} and target.port == 8787
    )
    if not (direct or account == 'gproxy' or account.startswith(('gproxy_', 'gproxy-'))):
        return deps
    deps.log(f'{trace_prefix}early_stop_policy enabled=false reason=gproxy_upstream {format_account_pool_route(response)}')
    return replace(deps, has_stop_tag=lambda _text: False, find_stop_tag=lambda _text: -1)
