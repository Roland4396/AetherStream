from __future__ import annotations

from typing import Any
from dataclasses import replace
from urllib.parse import urlsplit
import base64
import json

from aetherstream.features.early_stop import EarlyStopMatcher, bind_content_guard, is_himodels_upstream
from aetherstream.features.normal_completion_guard import observe_selected_account
from aetherstream.routing.channel_policy import current_policy, current_request, selected_early_stop
from aetherstream.routing.policy_config import merge


def _selected_stop_settings(response: Any) -> dict | None:
    policy = current_policy()
    if policy is None:
        return None
    if policy.get('_local_policy'):
        account_id = str(getattr(response, 'headers', {}).get('x-account-pool-id') or '')
        override = policy.get('account_overrides', {}).get(account_id, {}).get('early_stop', {})
        return merge(policy.get('early_stop', {}), override)
    raw = str(getattr(response, 'headers', {}).get('x-acc-early-stop') or '')
    if not raw or len(raw) > 6000:
        return None
    try:
        settings = json.loads(base64.urlsafe_b64decode(raw))
    except (ValueError, TypeError):
        return None
    if not isinstance(settings, dict) or not isinstance(settings.get('enabled'), bool):
        return None
    return settings


def upstream_early_stop_enabled(response: Any, url: str, default: bool) -> bool:
    settings = _selected_stop_settings(response)
    if settings is not None:
        return settings['enabled']
    policy = current_policy()
    if policy is not None:
        return bool(policy.get('early_stop', {}).get('enabled', False))
    # Compatibility for deployments that have not enabled the ACC registry.
    return True if is_himodels_upstream(response, url) else default


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
    observe_selected_account(response)
    settings = _selected_stop_settings(response)
    policy = current_policy()
    if policy is not None:
        settings = settings if settings is not None else policy.get('early_stop', {})
        selected_early_stop(settings)
        matcher = EarlyStopMatcher(
            lookup=lambda *keys: settings.get(keys[-1]), env_enabled=False,
            env_tags=None, env_case_sensitive=False, default_tags=[],
        )
        find = matcher.find
        if settings.get('content_guard', True):
            find = bind_content_guard(find, current_request(), log=deps.log)
        deps.log(f'{trace_prefix}early_stop_policy enabled={str(bool(settings.get("enabled"))).lower()} '
                 f'source=stream_policy action=disconnect {format_account_pool_route(response)}')
        return replace(deps, has_stop_tag=lambda text: find(text) >= 0, find_stop_tag=find)
    account = str(getattr(response, 'headers', {}).get('x-account-pool-id') or '').strip().lower()
    target = urlsplit(url)
    direct = target.hostname == 'gproxy' or (
        target.hostname in {'127.0.0.1', 'localhost', '::1'} and target.port == 8787
    )
    if not (direct or account == 'gproxy' or account.startswith(('gproxy_', 'gproxy-'))):
        return deps
    deps.log(f'{trace_prefix}early_stop_policy enabled=false reason=gproxy_upstream {format_account_pool_route(response)}')
    return replace(deps, has_stop_tag=lambda _text: False, find_stop_tag=lambda _text: -1)
