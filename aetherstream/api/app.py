#!/usr/bin/env python3
"""
Stream Proxy - 统一入口，路由分发
- 普通模型 → New API (OpenAI 格式)
- Gemini 模型 → 直连 Gemini HTTP
- Anthropic 原生格式 → New API 透传 (/v1/messages)
- 非流转流（内部流式收集）
- 流式转发与中断管理
"""

import os
import json
import uuid
import time
import threading
import copy
import re
import asyncio
import hashlib
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from aetherstream.config.urls import (
    normalize_codex_responses_base_url,
    normalize_openai_chat_base_url,
)
from aetherstream.runtime.docker_control import DockerContainerRestarter
from aetherstream.runtime.flags import RuntimeFlags
from aetherstream.upstreams.anthropic import AnthropicUpstreamDeps, forward_anthropic_stream
from aetherstream.upstreams.anthropic import (
    collect_anthropic_chat_completion,
    collect_anthropic_chat_completion_from_raw_sse,
    collect_anthropic_message_response,
    forward_anthropic_chat_stream,
    replay_anthropic_chat_stream,
)
from aetherstream.upstreams.codex import (
    CodexUpstreamDeps,
    collect_codex_chat_completion,
    forward_codex_chat_stream,
)
from aetherstream.upstreams.gemini import (
    GeminiUpstreamConfig,
    GeminiUpstreamDeps,
    collect_gemini_non_stream,
    forward_gemini_stream,
)
from aetherstream.routing.model_policy import ModelPolicy
from aetherstream.upstreams.openai import (
    OpenAIUpstreamDeps,
    collect_non_stream,
    collect_stream,
    forward_non_stream_as_openai_stream,
    forward_stream,
    replay_openai_chat_stream,
)
from aetherstream.observability.logging import ProxyLogger
from aetherstream.observability.summaries import (
    fold_claude_system_into_first_user_message,
    summarize_anthropic_cache_breakpoints,
    summarize_anthropic_request,
    summarize_openai_messages,
)
from aetherstream.observability.wiretap import ASGIWiretapMiddleware
from aetherstream.transforms.requests import (
    append_to_last_user_message,
    convert_chat_to_anthropic_messages_request,
    convert_chat_to_responses_request,
    insert_after_latest_human_message,
)
from aetherstream.streaming.sse import build_openai_sse_error
from aetherstream.streaming.state import ActiveStreamRegistry
from aetherstream.utils.coerce import (
    coerce_bool as _coerce_bool,
    coerce_positive_float as _coerce_positive_float,
    coerce_string_list as _coerce_string_list,
)
from aetherstream.features.drawing_filter import (
    apply_drawing_context_filter,
    should_apply_deepseek_drawing_context_filter,
)
from aetherstream.features.claude_replay import ClaudeReplayStore
from aetherstream.features.gpt_policy import (
    build_gpt_prompt_cache_key,
    inject_gpt_usage_policies_system_message,
)
from aetherstream.features.early_stop import (
    DEFAULT_EARLY_STOP_TAGS,
    EarlyStopMatcher,
)
from aetherstream.features.pro_compat import (
    apply_pro_no_reasoning_payload,
    should_append_pro_opus46_last_user_note,
)
from aetherstream.features.opus_notes import (
    PRO_OPUS46_LAST_USER_APPEND_MARKER,
    PRO_OPUS46_LAST_USER_APPEND_TEXT,
    PRO_OPUS_LAST_USER_APPEND_MARKER,
    PRO_OPUS_LAST_USER_APPEND_TEXT,
    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
    PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
)

app = FastAPI()




# Upstream service configuration
GEMINI_BASE_URL = os.environ.get('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com').rstrip('/')
TIMEOUT = int(os.environ.get('TIMEOUT', '600'))
DEBUG = os.environ.get('DEBUG', 'true').lower() == 'true'
# 原始 SSE 日志不截断。保留变量仅兼容各 upstream 函数签名。
MAX_RAW_SSE_BYTES = int(os.environ.get('MAX_RAW_SSE_BYTES', '0'))
# 保留旧环境变量读取，兼容历史配置；NewAPI fallback 已移除，不再用于上游转发。
FIXED_API_KEY = os.environ.get('FIXED_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
CODEX_API_KEY = os.environ.get('CODEX_API_KEY', '')
RAW_CODEX_BASE_URL = os.environ.get('CODEX_BASE_URL', 'https://api.openai.com/v1')
GPT_BASE_URL = normalize_openai_chat_base_url(RAW_CODEX_BASE_URL)
CODEX_BASE_URL = normalize_codex_responses_base_url(RAW_CODEX_BASE_URL)
GPT_USE_RESPONSES = os.environ.get('GPT_USE_RESPONSES', 'true').lower() == 'true'
GPT_SERVICE_TIER = (os.environ.get('GPT_SERVICE_TIER', 'priority') or '').strip()
GPT_PROMPT_CACHE_KEY = (os.environ.get('GPT_PROMPT_CACHE_KEY', '') or '').strip()
GPT_PROMPT_CACHE_RETENTION = (os.environ.get('GPT_PROMPT_CACHE_RETENTION', '24h') or '').strip()


MODEL_DIRECTORY_TTL_SEC = float(os.environ.get('MODEL_DIRECTORY_TTL_SEC', '30'))
GEMINI_ENABLED = os.environ.get('GEMINI_ENABLED', 'false').lower() == 'true'
GEMINI_INCLUDE_THOUGHTS = os.environ.get('GEMINI_INCLUDE_THOUGHTS', 'false').lower() == 'true'
GEMINI_HEARTBEAT_INTERVAL = int(os.environ.get('GEMINI_HEARTBEAT_INTERVAL', '15'))
GEMINI_MAX_RETRIES = int(os.environ.get('GEMINI_MAX_RETRIES', '3'))
GEMINI_RETRY_DELAY = float(os.environ.get('GEMINI_RETRY_DELAY', '1'))
CODEX_REASONING_EFFORT = os.environ.get('CODEX_REASONING_EFFORT', 'xhigh')
CODEX_REASONING_SUMMARY = os.environ.get('CODEX_REASONING_SUMMARY', '').strip()
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', os.environ.get('ANTHROPIC_API_KEY', ''))
CLAUDE_BASE_URL = os.environ.get('CLAUDE_BASE_URL', os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')).rstrip('/')

# Claude provider prefixes: free/ → CLAUDE_*, codecli/ → CLAUDE2_*
CLAUDE2_API_KEY = os.environ.get('CLAUDE2_API_KEY', '').strip()
CLAUDE2_BASE_URL = os.environ.get('CLAUDE2_BASE_URL', '').rstrip('/')

CLAUDE_PROVIDER_PREFIX_MAP = {
    'free': {'api_key_attr': 'CLAUDE_API_KEY', 'base_url_attr': 'CLAUDE_BASE_URL'},
    'codecli': {'api_key_attr': 'CLAUDE2_API_KEY', 'base_url_attr': 'CLAUDE2_BASE_URL'},
}


def parse_claude_provider_prefix(model: str) -> tuple[str, str, str]:
    """Parse provider prefix from model name.

    Returns (provider, real_model, display_label).
    If no known prefix, returns ('', model, model).
    """
    if '/' in model:
        prefix, _, rest = model.partition('/')
        prefix_lower = prefix.lower()
        if prefix_lower in CLAUDE_PROVIDER_PREFIX_MAP:
            return prefix_lower, rest, f'{prefix_lower}/{rest}'
    return '', model, model


def get_claude_upstream_for_provider(provider: str) -> tuple[str, str]:
    """Return (api_key, base_url) for the given provider prefix."""
    if provider == 'codecli':
        return CLAUDE2_API_KEY, CLAUDE2_BASE_URL
    # default / 'free'
    return CLAUDE_API_KEY, CLAUDE_BASE_URL
CLAUDE_CODE_VERSION = os.environ.get('CLAUDE_CODE_VERSION', '2.1.119').strip() or '2.1.119'
CLAUDE_USER_ID = os.environ.get('CLAUDE_USER_ID', '').strip()
CLAUDE_BILLING_HEADER = os.environ.get('CLAUDE_BILLING_HEADER', '').strip()
CLAUDE_SYSTEM_PREFIX = os.environ.get(
    'CLAUDE_SYSTEM_PREFIX',
    "You are Claude Code, Anthropic's official CLI for Claude.",
)
CLAUDE_AGENT_BILLING_HEADER = os.environ.get('CLAUDE_AGENT_BILLING_HEADER', '').strip()
CLAUDE_AGENT_SYSTEM_PREFIX = os.environ.get(
    'CLAUDE_AGENT_SYSTEM_PREFIX',
    "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
).strip()
CLAUDE_INCLUDE_SYSTEM_PREFIX = os.environ.get('CLAUDE_INCLUDE_SYSTEM_PREFIX', 'true').lower() == 'true'
CLAUDE_ANTHROPIC_VERSION = os.environ.get('CLAUDE_ANTHROPIC_VERSION', '2023-06-01')
CLAUDE_BETA = os.environ.get(
    'CLAUDE_BETA',
    'claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,context-management-2025-06-27,prompt-caching-scope-2026-01-05,advisor-tool-2026-03-01,effort-2025-11-24',
)
CLAUDE_HAIKU_UNSUPPORTED_BETAS = {
    'context-management-2025-06-27',
    'effort-2025-11-24',
}
CLAUDE_ONE_M_CONTEXT_BETA = 'context-1m-2025-08-07'
CLAUDE_USER_AGENT = os.environ.get('CLAUDE_USER_AGENT', f'claude-cli/{CLAUDE_CODE_VERSION} (external, cli)')
CLAUDE_STAINLESS_PACKAGE_VERSION = (
    os.environ.get('CLAUDE_STAINLESS_PACKAGE_VERSION', '0.81.0') or '0.81.0'
).strip() or '0.81.0'
CLAUDE_STAINLESS_RUNTIME_VERSION = (
    os.environ.get('CLAUDE_STAINLESS_RUNTIME_VERSION', 'v24.3.0') or 'v24.3.0'
).strip() or 'v24.3.0'
CLAUDE_DEFAULT_MAX_TOKENS = int(os.environ.get('CLAUDE_DEFAULT_MAX_TOKENS', '8192'))
CLAUDE_PROMPT_CACHING_ENABLED = os.environ.get('CLAUDE_PROMPT_CACHING_ENABLED', 'false').lower() == 'true'
CLAUDE_PROMPT_CACHING_TYPE = (os.environ.get('CLAUDE_PROMPT_CACHING_TYPE', 'ephemeral') or 'ephemeral').strip()
CLAUDE_PROMPT_CACHING_TTL = (os.environ.get('CLAUDE_PROMPT_CACHING_TTL', '') or '').strip()
CLAUDE_PROMPT_CACHING_MODE = (os.environ.get('CLAUDE_PROMPT_CACHING_MODE', 'automatic') or 'automatic').strip().lower()
CLAUDE_OUTPUT_EFFORT = (os.environ.get('CLAUDE_OUTPUT_EFFORT', '') or '').strip()
CLAUDE_THINKING_TYPE = (os.environ.get('CLAUDE_THINKING_TYPE', 'disabled') or 'disabled').strip()
CLAUDE_SESSION_TTL_SEC = float(os.environ.get('CLAUDE_SESSION_TTL_SEC', '1800'))
CLAUDE_KEYWORD_FILTER_ENABLED = os.environ.get('CLAUDE_KEYWORD_FILTER_ENABLED', 'false').lower() == 'true'
CLAUDE_KEYWORD_FILTER_KEYWORDS = (os.environ.get('CLAUDE_KEYWORD_FILTER_KEYWORDS', '') or '').strip()
RUNTIME_FLAGS_PATH = (os.environ.get('RUNTIME_FLAGS_PATH', '/app/runtime-flags.json') or '/app/runtime-flags.json').strip()
RUNTIME_FLAGS_POLL_SEC = float(os.environ.get('RUNTIME_FLAGS_POLL_SEC', '1'))
AUTO_RESTART_SILLYTAVERN = os.environ.get('AUTO_RESTART_SILLYTAVERN', 'false').lower() == 'true'
RESTART_COOLDOWN_SEC = float(os.environ.get('RESTART_COOLDOWN_SEC', '60'))
_last_restart_scheduled_at = 0.0
_restart_schedule_lock = threading.Lock()
_model_directory_cache: dict[str, object] = {
    "checked_at": 0.0,
    "models": [],
    "routes": {},
}
_claude_session_states: dict[str, dict[str, object]] = {}
_claude_session_lock = threading.Lock()

_container_restarter: DockerContainerRestarter | None = None


def restart_sillytavern():
    """重启 SillyTavern 容器。"""
    global _container_restarter
    if _container_restarter is None:
        _container_restarter = DockerContainerRestarter(container_name='sillytavern', log=log)
    _container_restarter.restart(timeout=5)


def _resolve_configured_api_key(item: dict) -> tuple[str, str]:
    """Resolve an upstream API key from inline config or an env var name.

    Runtime config may use either `api_key` for local/private deployments or
    `api_key_env` for open-source-safe examples.  Prefer `api_key_env` in shared
    configs so real credentials never live in JSON.
    """
    api_key = str(item.get('api_key') or '').strip()
    api_key_env = str(item.get('api_key_env') or '').strip()
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env, '').strip()
    return api_key, api_key_env


def _normalize_openai_base_url(raw_url: Any) -> str:
    base = str(raw_url or '').strip().rstrip('/')
    if base.endswith('/v1/chat/completions'):
        return base[:-len('/chat/completions')]
    if base.endswith('/v1/models'):
        return base[:-len('/models')]
    return base


def get_openai_compatible_upstreams() -> list[dict[str, str]]:
    raw = _runtime_lookup('openai_compatible_upstreams')
    if raw is None:
        raw = _runtime_lookup('model_directory', 'openai_compatible_upstreams')
    if not isinstance(raw, list):
        return []

    upstreams: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        base_url = _normalize_openai_base_url(item.get('base_url') or item.get('url'))
        if not base_url:
            continue
        name = str(item.get('name') or f'openai-compatible-{idx + 1}').strip()
        key = name or base_url
        if key in seen:
            continue
        seen.add(key)
        api_key, api_key_env = _resolve_configured_api_key(item)
        upstreams.append({
            'name': name,
            'base_url': base_url,
            'api_key': api_key,
            'api_key_env': api_key_env,
        })
    return upstreams


def _openai_compatible_upstream_text(upstream: dict[str, str]) -> str:
    return f"{upstream.get('name') or ''} {upstream.get('base_url') or ''}".lower()


def _is_fake_openai_compatible_upstream(upstream: dict[str, str]) -> bool:
    text = _openai_compatible_upstream_text(upstream)
    return 'fake-' in text or 'fake_' in text or 'fake-openai' in text


def _copy_upstream_with_reason(upstream: dict[str, str], reason: str) -> dict[str, str]:
    selected = dict(upstream)
    selected['selection_reason'] = reason
    return selected


def _find_openai_compatible_upstream(
    upstreams: list[dict[str, str]],
    markers: tuple[str, ...],
    *,
    allow_fake: bool = False,
) -> dict[str, str] | None:
    # Prefer real upstreams. The local fake-openai service is for testing and
    # should not steal model-family pass-through just because its name contains
    # "openai".
    for upstream in upstreams:
        if not allow_fake and _is_fake_openai_compatible_upstream(upstream):
            continue
        text = _openai_compatible_upstream_text(upstream)
        if any(marker in text for marker in markers):
            return upstream
    if allow_fake:
        return None
    return _find_openai_compatible_upstream(upstreams, markers, allow_fake=True)


def resolve_model_name_passthrough_upstream(model: Any) -> dict[str, str] | None:
    """Pick an OpenAI-compatible pass-through upstream by model-id family.

    This is only used after explicit /v1/models routing and built-in
    Gemini/Claude/GPT routing miss, so it must not blindly pick the first
    upstream. Runtime test upstreams such as fake-openai are skipped unless
    they are the only configured choice.
    """
    upstreams = get_openai_compatible_upstreams()
    if not upstreams:
        return None

    model_id = str(model or '').strip().lower()
    if model_id:
        if model_id.startswith('deepseek') or 'deepseek' in model_id:
            matched = _find_openai_compatible_upstream(upstreams, ('deepseek', 'api.deepseek.com'))
            if matched:
                return _copy_upstream_with_reason(matched, 'model_family:deepseek')

        if model_id.startswith('claude-') or 'claude' in model_id or model_id.startswith('anthropic/'):
            matched = _find_openai_compatible_upstream(
                upstreams,
                ('claude', 'anthropic'),
            )
            if matched:
                return _copy_upstream_with_reason(matched, 'model_family:claude')

        if model_policy.is_gemini_model(model_id) or model_id.startswith('google/'):
            matched = _find_openai_compatible_upstream(
                upstreams,
                ('gemini', 'google'),
            )
            if matched:
                return _copy_upstream_with_reason(matched, 'model_family:gemini')

        if (
            model_policy.is_gpt_model(model_id)
            or model_id.startswith(('openai/', 'chatgpt-', 'o1', 'o3', 'o4', 'codex-'))
        ):
            matched = _find_openai_compatible_upstream(
                upstreams,
                ('openai', 'gpt', 'codex'),
            )
            if matched:
                return _copy_upstream_with_reason(matched, 'model_family:openai')

    configured_default = str(
        _runtime_lookup('default_openai_compatible_upstream')
        or _runtime_lookup('model_directory', 'default_openai_compatible_upstream')
        or ''
    ).strip().lower()
    if configured_default:
        for upstream in upstreams:
            if configured_default in _openai_compatible_upstream_text(upstream):
                return _copy_upstream_with_reason(upstream, 'configured_default')

    for upstream in upstreams:
        if not _is_fake_openai_compatible_upstream(upstream):
            return _copy_upstream_with_reason(upstream, 'first_non_fake')

    return _copy_upstream_with_reason(upstreams[0], 'first_configured')


async def refresh_model_directory(
    force: bool = False,
    request_authorization: str = "",
) -> dict[str, object]:
    global _model_directory_cache

    now = time.monotonic()
    auth_mode = "request-auth" if str(request_authorization or "").strip() else "no-auth"
    if (
        not force
        and now - float(_model_directory_cache.get('checked_at') or 0.0) < MODEL_DIRECTORY_TTL_SEC
        and _model_directory_cache.get('auth_mode') == auth_mode
    ):
        return _model_directory_cache

    local_models: list[dict[str, object]] = []
    routes: dict[str, dict[str, str]] = {}

    for model_id in sorted(model_policy.allowed_gpt_models):
        local_models.append({"id": model_id, "object": "model", "owned_by": "openai"})

    if GEMINI_ENABLED:
        for model_id in sorted(model_policy.allowed_gemini_models):
            local_models.append({"id": model_id, "object": "model", "owned_by": "google"})

    for model_id in sorted(model_policy.allowed_claude_models):
        local_models.append({"id": model_id, "object": "model", "owned_by": "anthropic"})
        # 添加带前缀的 Claude 模型（free/ 和 codecli/）
        if CLAUDE_API_KEY:
            local_models.append({"id": f"free/{model_id}", "object": "model", "owned_by": "anthropic-free"})
        if CLAUDE2_API_KEY:
            local_models.append({"id": f"codecli/{model_id}", "object": "model", "owned_by": "anthropic-codecli"})

    local_models.append({
        "id": "fake-slow-stream",
        "object": "model",
        "owned_by": "stream-proxy-local",
    })
    local_models.append({
        "id": "fake-slow-stream-600s",
        "object": "model",
        "owned_by": "stream-proxy-local",
    })

    upstreams = get_openai_compatible_upstreams()
    log(
        "model_directory refresh_start "
        f"force={force} auth_mode={auth_mode} upstreams={len(upstreams)}"
    )
    async with httpx.AsyncClient(timeout=get_timeout_config()) as client:
        for upstream in upstreams:
            name = upstream['name']
            base_url = upstream['base_url']
            models_url = f"{base_url}/models"
            headers = {'Accept': 'application/json'}
            if upstream.get('api_key'):
                headers['Authorization'] = f"Bearer {upstream['api_key']}"
            elif request_authorization:
                headers['Authorization'] = str(request_authorization)

            try:
                response = await client.get(models_url, headers=headers)
                log(f"model_directory upstream={name} status={response.status_code} url={models_url}")
                if response.status_code != 200:
                    continue
                payload = response.json()
            except Exception as e:
                log(f"model_directory upstream={name} failed: {e}")
                continue

            data = payload.get('data') if isinstance(payload, dict) else None
            if not isinstance(data, list):
                log(f"model_directory upstream={name} ignored: data_not_list")
                continue

            upstream_model_count = 0
            for item in data:
                if isinstance(item, dict):
                    model_id = str(item.get('id') or item.get('name') or item.get('model') or '').strip()
                    model_item = dict(item)
                else:
                    model_id = str(item or '').strip()
                    model_item = {"id": model_id, "object": "model"}
                if not model_id:
                    continue
                model_item.setdefault('id', model_id)
                model_item.setdefault('object', 'model')
                model_item.setdefault('owned_by', name)
                local_models.append(model_item)
                routes[model_id] = {
                    'name': name,
                    'base_url': base_url,
                    'api_key': upstream.get('api_key') or '',
                }
                upstream_model_count += 1
            log(
                "model_directory upstream_loaded "
                f"name={name} models={upstream_model_count} base_url={base_url}"
            )

    deduped: list[dict[str, object]] = []
    seen_models: set[str] = set()
    for item in local_models:
        model_id = str(item.get('id') or '').strip()
        if not model_id or model_id in seen_models:
            continue
        seen_models.add(model_id)
        deduped.append(item)

    _model_directory_cache = {
        "checked_at": now,
        "auth_mode": auth_mode,
        "models": deduped,
        "routes": routes,
    }
    log(
        "model_directory refresh_done "
        f"models={len(deduped)} dynamic_routes={len(routes)} auth_mode={auth_mode}"
    )
    return _model_directory_cache


async def resolve_openai_compatible_route(
    model: Any,
    request_authorization: str = "",
) -> dict[str, str] | None:
    model_id = str(model or '').strip()
    if not model_id:
        return None
    directory = await refresh_model_directory(request_authorization=request_authorization)
    routes = directory.get('routes')
    if isinstance(routes, dict) and model_id in routes:
        route = routes[model_id]
        if isinstance(route, dict):
            log(
                "model_directory route_hit "
                f"model={model_id} upstream={route.get('name', '-')}"
            )
        return routes[model_id]
    directory = await refresh_model_directory(force=True, request_authorization=request_authorization)
    routes = directory.get('routes')
    if isinstance(routes, dict):
        route = routes.get(model_id)
        if isinstance(route, dict):
            log(
                "model_directory route_hit_after_refresh "
                f"model={model_id} upstream={route.get('name', '-')}"
            )
        else:
            log(
                "model_directory route_miss "
                f"model={model_id} dynamic_routes={len(routes)}"
            )
        return route if isinstance(route, dict) else None
    return None


def get_claude_prompt_caching_settings() -> dict[str, object]:
    enabled = _coerce_bool(
        _runtime_lookup('claude', 'prompt_caching', 'enabled'),
        CLAUDE_PROMPT_CACHING_ENABLED,
    )
    cache_type = str(
        _runtime_lookup('claude', 'prompt_caching', 'type') or CLAUDE_PROMPT_CACHING_TYPE or 'ephemeral'
    ).strip() or 'ephemeral'
    ttl_raw = _runtime_lookup('claude', 'prompt_caching', 'ttl')
    ttl = CLAUDE_PROMPT_CACHING_TTL if ttl_raw is None else str(ttl_raw).strip()
    mode_raw = _runtime_lookup('claude', 'prompt_caching', 'mode')
    mode = str(CLAUDE_PROMPT_CACHING_MODE if mode_raw is None else mode_raw).strip().lower() or 'automatic'
    if mode not in {'automatic', 'explicit'}:
        mode = 'automatic'
    return {
        'enabled': enabled,
        'type': cache_type,
        'ttl': ttl,
        'mode': mode,
    }


def get_claude_cache_keepalive_settings() -> dict[str, object]:
    enabled = _coerce_bool(
        _runtime_lookup('claude', 'prompt_caching', 'keepalive', 'enabled'),
        False,
    )
    interval_sec = _coerce_positive_float(
        _runtime_lookup('claude', 'prompt_caching', 'keepalive', 'interval_sec'),
        240.0,
    )
    first_data_timeout_sec = _coerce_positive_float(
        _runtime_lookup('claude', 'prompt_caching', 'keepalive', 'first_data_timeout_sec'),
        60.0,
    )
    max_tokens_raw = _runtime_lookup('claude', 'prompt_caching', 'keepalive', 'max_tokens')
    try:
        max_tokens = int(max_tokens_raw)
    except Exception:
        max_tokens = 1
    if max_tokens <= 0:
        max_tokens = 1
    close_after_data_events_raw = _runtime_lookup('claude', 'prompt_caching', 'keepalive', 'close_after_data_events')
    try:
        close_after_data_events = int(close_after_data_events_raw)
    except Exception:
        close_after_data_events = 1
    if close_after_data_events <= 0:
        close_after_data_events = 1
    max_runs_raw = _runtime_lookup('claude', 'prompt_caching', 'keepalive', 'max_runs')
    try:
        max_runs = int(max_runs_raw)
    except Exception:
        max_runs = 5
    if max_runs <= 0:
        max_runs = 5
    max_runs = min(max_runs, 5)
    post_delay_sec = _coerce_positive_float(
        _runtime_lookup('claude', 'prompt_caching', 'keepalive', 'post_delay_sec'),
        interval_sec,
    )
    return {
        'enabled': enabled,
        'interval_sec': interval_sec,
        'max_tokens': max_tokens,
        'close_after_data_events': close_after_data_events,
        'first_data_timeout_sec': first_data_timeout_sec,
        'max_runs': max_runs,
        'post_delay_sec': post_delay_sec,
    }


def get_claude_output_settings() -> dict[str, object]:
    effort_raw = _runtime_lookup('claude', 'output_config', 'effort')
    effort = CLAUDE_OUTPUT_EFFORT if effort_raw is None else str(effort_raw).strip()
    return {
        'effort': effort,
    }


def get_claude_keyword_filter_settings() -> dict[str, object]:
    enabled = _coerce_bool(
        _runtime_lookup('claude', 'keyword_filter', 'enabled'),
        CLAUDE_KEYWORD_FILTER_ENABLED,
    )
    keywords_raw = _runtime_lookup('claude', 'keyword_filter', 'keywords')
    keywords_source = CLAUDE_KEYWORD_FILTER_KEYWORDS if keywords_raw is None else keywords_raw
    keywords = _coerce_string_list(keywords_source)
    return {
        'enabled': enabled,
        'keywords': keywords,
    }


def get_claude_client_compat_settings() -> dict[str, object]:
    edits_raw = _runtime_lookup('claude', 'client_compat', 'context_management', 'edits')
    default_edits = [
        {
            'type': 'clear_thinking_20251015',
            'keep': 'all',
        }
    ]
    edits = default_edits
    if isinstance(edits_raw, list):
        normalized_edits = [copy.deepcopy(item) for item in edits_raw if isinstance(item, dict)]
        if normalized_edits:
            edits = normalized_edits
    return {
        'thinking_type': str(
            _runtime_lookup('claude', 'client_compat', 'thinking', 'type')
            or CLAUDE_THINKING_TYPE
            or 'disabled'
        ).strip() or 'disabled',
        'tools_enabled': _coerce_bool(
            _runtime_lookup('claude', 'client_compat', 'tools', 'enabled'),
            True,
        ),
        'context_management_enabled': _coerce_bool(
            _runtime_lookup('claude', 'client_compat', 'context_management', 'enabled'),
            True,
        ),
        'context_management_edits': edits,
    }


def is_claude_haiku_model(model: Any) -> bool:
    return 'haiku' in str(model or '').lower()


def is_claude_opus_model(model: Any) -> bool:
    return 'opus' in str(model or '').lower()


def is_claude_sonnet_model(model: Any) -> bool:
    return 'sonnet' in str(model or '').lower()


def is_claude_fable_model(model: Any) -> bool:
    return 'fable' in str(model or '').lower()


def is_claude_prompt_cache_model(model: Any) -> bool:
    return is_claude_opus_model(model) or is_claude_sonnet_model(model)


def supports_claude_one_m_context(model: Any) -> bool:
    model_name = str(model or '').lower()
    return (
        'opus-4-6' in model_name
        or 'opus-4-7' in model_name
        or 'opus-4-8' in model_name
    )


def build_claude_beta_for_model(model: Any) -> str:
    tokens = [
        token.strip()
        for token in str(CLAUDE_BETA or '').split(',')
        if token.strip()
    ]
    if not supports_claude_one_m_context(model):
        tokens = [token for token in tokens if token != CLAUDE_ONE_M_CONTEXT_BETA]
    if is_claude_haiku_model(model):
        tokens = [token for token in tokens if token not in CLAUDE_HAIKU_UNSUPPORTED_BETAS]
    return ','.join(tokens)


CLAUDE_SESSION_ID_RE = re.compile(
    r'(?P<session_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
)


def extract_claude_session_id(value: Any = None) -> str:
    candidates: list[Any] = []
    if value is not None:
        candidates.append(value)
    if CLAUDE_USER_ID:
        candidates.append(CLAUDE_USER_ID)

    for candidate in candidates:
        if isinstance(candidate, dict):
            session_id = str(candidate.get('session_id') or '').strip()
            if session_id:
                return session_id
            continue

        if not isinstance(candidate, str):
            continue

        raw = candidate.strip()
        if not raw:
            continue

        try:
            loaded = json.loads(raw)
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            session_id = str(loaded.get('session_id') or '').strip()
            if session_id:
                return session_id

        match = CLAUDE_SESSION_ID_RE.search(raw)
        if match:
            return match.group('session_id')

    return str(uuid.uuid4())


def get_claude_session_ttl_sec() -> float:
    raw = _runtime_lookup('claude', 'session', 'ttl_sec')
    if raw is None:
        raw = _runtime_lookup('claude', 'session_ttl_sec')
    return _coerce_positive_float(raw, CLAUDE_SESSION_TTL_SEC)


def _normalize_claude_session_key(model: Any = None, provider: Any = None) -> str:
    provider_key = str(provider or "default").strip().lower() or "default"
    model_key = str(model or "unknown").strip().lower() or "unknown"
    return f"{provider_key}:{model_key}"


def _current_claude_session_id(
    now: float | None = None,
    *,
    model: Any = None,
    provider: Any = None,
) -> tuple[str, str, float, str]:
    """Reuse Claude Code session inside a continuous time window.

    Continuous SillyTavern turns should keep one upstream session so Code CLI's
    own continuity does not get reset every call.  A new session id is minted
    only after the session has been idle for the configured TTL.
    """
    now = time.monotonic() if now is None else now
    ttl_sec = get_claude_session_ttl_sec()
    session_key = _normalize_claude_session_key(model=model, provider=provider)
    with _claude_session_lock:
        state = _claude_session_states.setdefault(
            session_key,
            {
                "session_id": "",
                "created_at": 0.0,
                "last_used_at": 0.0,
            },
        )
        session_id = str(state.get('session_id') or '').strip()
        created_at = float(state.get('created_at') or 0.0)
        last_used_at = float(state.get('last_used_at') or 0.0)
        idle_for = now - last_used_at if last_used_at > 0.0 else ttl_sec
        expired = (not session_id) or (created_at <= 0.0) or (idle_for >= ttl_sec)
        mode = 'reused'
        if expired:
            session_id = str(uuid.uuid4())
            state['session_id'] = session_id
            state['created_at'] = now
            mode = 'new'
        state['last_used_at'] = now
    return session_id, mode, ttl_sec, session_key


def build_claude_user_id_for_session(session_id: str) -> str:
    raw = (CLAUDE_USER_ID or "").strip()
    if not raw:
        return json.dumps({"session_id": session_id}, separators=(",", ":"))

    try:
        loaded = json.loads(raw)
    except Exception:
        loaded = None

    if isinstance(loaded, dict):
        loaded = dict(loaded)
        loaded["session_id"] = session_id
        return json.dumps(loaded, ensure_ascii=False, separators=(",", ":"))

    if CLAUDE_SESSION_ID_RE.search(raw):
        return CLAUDE_SESSION_ID_RE.sub(session_id, raw, count=1)

    return f"{raw}_session_{session_id}"


def build_timed_claude_user_id(
    *,
    model: Any = None,
    provider: Any = None,
) -> tuple[str, str, str, float, str]:
    session_id, session_mode, ttl_sec, session_key = _current_claude_session_id(
        model=model,
        provider=provider,
    )
    return build_claude_user_id_for_session(session_id), session_id, session_mode, ttl_sec, session_key


def build_fresh_claude_user_id() -> tuple[str, str]:
    """Backward-compatible wrapper: return timed-session user_id and id."""
    user_id, session_id, _session_mode, _ttl_sec, _session_key = build_timed_claude_user_id()
    return user_id, session_id


def apply_claude_client_compat_request(payload: dict) -> tuple[dict, dict[str, str]]:
    cfg = get_claude_client_compat_settings()
    sanitized = copy.deepcopy(payload)
    compat_meta: dict[str, str] = {}

    if cfg['tools_enabled']:
        tools = sanitized.get('tools')
        if isinstance(tools, list):
            if tools:
                sanitized['tools'] = []
                compat_meta['tools'] = f'replaced_with_empty:{len(tools)}'
            else:
                sanitized['tools'] = []
                compat_meta['tools'] = 'kept_empty'
        else:
            sanitized['tools'] = []
            compat_meta['tools'] = 'injected_empty'
    else:
        if 'tools' in sanitized:
            sanitized.pop('tools', None)
            compat_meta['tools'] = 'removed'
        else:
            compat_meta['tools'] = 'disabled'

    thinking_type = str(cfg.get('thinking_type') or 'disabled').strip()
    if thinking_type:
        sanitized['thinking'] = {'type': thinking_type}
        compat_meta['thinking'] = f'set_{thinking_type}'
    else:
        thinking = sanitized.get('thinking')
        if isinstance(thinking, dict):
            compat_meta['thinking'] = 'preserved'
        else:
            sanitized['thinking'] = {'type': 'adaptive'}
            compat_meta['thinking'] = 'injected_adaptive'

    if thinking_type == 'disabled':
        sanitized.pop('context_management', None)
        compat_meta['context_management'] = 'removed_for_disabled_thinking'
    elif cfg['context_management_enabled']:
        context_management = sanitized.get('context_management')
        edits: list | None = None
        source = 'preserved'

        if isinstance(context_management, dict) and isinstance(context_management.get('edits'), list):
            edits = copy.deepcopy(context_management.get('edits'))
        elif cfg['context_management_edits']:
            edits = copy.deepcopy(cfg['context_management_edits'])
            source = 'injected'

        if edits:
            sanitized['context_management'] = {'edits': edits}
            compat_meta['context_management'] = source
        else:
            sanitized.pop('context_management', None)
            compat_meta['context_management'] = 'absent'
    else:
        if 'context_management' in sanitized:
            sanitized.pop('context_management', None)
            compat_meta['context_management'] = 'removed'
        else:
            compat_meta['context_management'] = 'disabled'

    return sanitized, compat_meta


def apply_claude_model_compat_request(payload: dict) -> tuple[dict, dict[str, str]]:
    model = str(payload.get('model') or '').strip()
    if is_claude_fable_model(model):
        sanitized = copy.deepcopy(payload)
        compat_meta: dict[str, str] = {}
        thinking = sanitized.get('thinking')
        if isinstance(thinking, dict) and str(thinking.get('type') or '').strip().lower() == 'disabled':
            sanitized.pop('thinking', None)
            compat_meta['thinking'] = 'removed_disabled_for_fable'
        else:
            compat_meta['thinking'] = 'preserved_for_fable' if 'thinking' in sanitized else 'absent_for_fable'
        return sanitized, compat_meta

    if not is_claude_haiku_model(model):
        return payload, {}

    sanitized = copy.deepcopy(payload)
    compat_meta: dict[str, str] = {}

    sanitized['thinking'] = {'type': 'disabled'}
    compat_meta['thinking'] = 'haiku_forced_disabled'

    if 'output_config' in sanitized:
        sanitized.pop('output_config', None)
        compat_meta['output_config'] = 'removed_for_haiku'
    else:
        compat_meta['output_config'] = 'absent'

    if 'context_management' in sanitized:
        sanitized.pop('context_management', None)
        compat_meta['context_management'] = 'removed_for_haiku'
    else:
        compat_meta['context_management'] = 'absent'

    return sanitized, compat_meta

def schedule_delayed_restart(delay: float = 2.0):
    """延迟重启 SillyTavern"""
    global _last_restart_scheduled_at
    if not AUTO_RESTART_SILLYTAVERN:
        log("Auto restart disabled; skip SillyTavern restart")
        return

    with _restart_schedule_lock:
        now = time.monotonic()
        elapsed = now - _last_restart_scheduled_at
        if elapsed < RESTART_COOLDOWN_SEC:
            log(f"Restart suppressed by cooldown: {elapsed:.1f}s < {RESTART_COOLDOWN_SEC:.1f}s")
            return
        _last_restart_scheduled_at = now

    log(f"Scheduling restart in {delay}s")
    threading.Timer(delay, restart_sillytavern).start()

# 请求/响应日志目录
LOG_DIR = os.environ.get('LOG_DIR', './logs')
CLAUDE_REPLAY_CONTROL_FILE = os.environ.get(
    'CLAUDE_REPLAY_CONTROL_FILE',
    os.path.join(LOG_DIR, 'claude_replay_switch.json'),
)

proxy_logger = ProxyLogger(debug=DEBUG, log_dir=LOG_DIR)
active_stream_registry = ActiveStreamRegistry(log=proxy_logger.log)

# Short-lived exact-request coalescing for true non-stream upstreams.
#
# Evidence from SillyTavern/pro routing showed the same non-stream Gemini body
# can be submitted multiple times while the first upstream call is still
# waiting for a body. A true non-stream response cannot emit keepalives before
# the JSON body, so the safest server-side mitigation is to avoid sending the
# same exact payload upstream more than once and let later retries wait for, or
# briefly replay, the first result.
NONSTREAM_DEDUPE_TTL = float(os.environ.get('NONSTREAM_DEDUPE_TTL', '180'))
_nonstream_dedupe_lock = asyncio.Lock()
_nonstream_inflight: dict[str, dict[str, Any]] = {}
_nonstream_recent_results: dict[str, dict[str, Any]] = {}

model_policy = ModelPolicy(
    codex_models=frozenset(),
    allowed_gpt_models=frozenset({'gpt-5.4', 'gpt-5.4-pro'}),
    allowed_gemini_models=frozenset({'gemini-3.1-pro-preview', 'gemini-3-flash-preview'}),
    allowed_claude_models=frozenset({
        'claude-fable-5',
        'claude-opus-4-6',
        'claude-opus-4-8',
        'claude-haiku-4-5-20251001',
        'claude-sonnet-4-6',
    }),
    codex_unsupported_fields=frozenset({'stop', 'presence_penalty', 'frequency_penalty'}),
)


def log(msg: str):
    proxy_logger.log(msg)


runtime_flags = RuntimeFlags(
    path=RUNTIME_FLAGS_PATH,
    poll_sec=RUNTIME_FLAGS_POLL_SEC,
    log=log,
)


def _runtime_lookup(*keys):
    return runtime_flags.lookup(*keys)


early_stop_matcher = EarlyStopMatcher(
    lookup=_runtime_lookup,
    env_enabled=os.environ.get('EARLY_STOP_ENABLED', 'false').lower() == 'true',
    env_tags=os.environ.get('EARLY_STOP_TAGS'),
    env_case_sensitive=os.environ.get('EARLY_STOP_CASE_SENSITIVE', 'true').lower() != 'false',
    default_tags=DEFAULT_EARLY_STOP_TAGS,
)


def get_early_stop_settings() -> dict[str, object]:
    return early_stop_matcher.settings()


def find_stop_tag(text: str) -> int:
    return early_stop_matcher.find(text)


def has_stop_tag(text: str) -> bool:
    return early_stop_matcher.has(text)



claude_replay = ClaudeReplayStore(
    log_dir=LOG_DIR,
    control_file=CLAUDE_REPLAY_CONTROL_FILE,
    log=log,
)




def build_caller_fingerprint(request: Request) -> tuple[str, str]:
    return proxy_logger.build_caller_fingerprint(request)


def fmt_ms(start: float, end: float | None = None) -> str:
    return proxy_logger.fmt_ms(start, end)


app.add_middleware(ASGIWiretapMiddleware, log=log, fmt_ms=fmt_ms)


def save_request_log(
    model: str,
    messages: list,
    response: str,
    stream: bool,
    raw_sse: str = "",
    request_payload: dict | None = None,
    inbound_request_payload: dict | None = None,
    debug_meta: dict | None = None,
    error_type: str | None = None,
    trace_id: str | None = None,
):
    proxy_logger.save_request_log(
        model=model,
        messages=messages,
        response=response,
        stream=stream,
        raw_sse=raw_sse,
        request_payload=request_payload,
        inbound_request_payload=inbound_request_payload,
        debug_meta=debug_meta,
        error_type=error_type,
        trace_id=trace_id,
    )


def release_active_stream_caller(caller_key: str, trace_id: str) -> None:
    active_stream_registry.release(caller_key, trace_id)


def build_exact_request_key(request_payload: dict) -> str:
    body = json.dumps(request_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(body.encode('utf-8', errors='replace')).hexdigest()


def clone_jsonable(value: Any) -> Any:
    return copy.deepcopy(value)


def extract_openai_chat_payload_content(payload: dict) -> str:
    try:
        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            return ''
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get('message')
        if isinstance(message, dict):
            content = message.get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get('text'), str):
                        parts.append(item['text'])
                    elif isinstance(item, str):
                        parts.append(item)
                return ''.join(parts)
        text = choice.get('text')
        return text if isinstance(text, str) else ''
    except Exception:
        return ''


def _cleanup_nonstream_dedupe(now: float) -> None:
    expired = [
        key
        for key, item in _nonstream_recent_results.items()
        if now - float(item.get('stored_at') or 0.0) > NONSTREAM_DEDUPE_TTL
    ]
    for key in expired:
        _nonstream_recent_results.pop(key, None)


async def run_exact_nonstream_once(
    *,
    dedupe_key: str,
    trace_id: str,
    upstream_label: str,
    runner,
) -> tuple[dict, bool]:
    """Run an exact non-stream request once across concurrent retries.

    Returns (payload, shared), where shared=True means this request reused an
    in-flight or very recent identical result instead of calling upstream again.
    """
    trace_prefix = f"[TRACE {trace_id}]"
    now = time.monotonic()
    leader = False
    waiter_count = 0

    async with _nonstream_dedupe_lock:
        _cleanup_nonstream_dedupe(now)
        recent = _nonstream_recent_results.get(dedupe_key)
        if recent:
            age = now - float(recent.get('stored_at') or now)
            log(
                f"{trace_prefix} nonstream_dedupe_recent_hit "
                f"upstream={upstream_label} key={dedupe_key[:16]} age={age:.1f}s"
            )
            return clone_jsonable(recent['payload']), True

        entry = _nonstream_inflight.get(dedupe_key)
        if entry is None:
            task = asyncio.create_task(runner())
            _nonstream_inflight[dedupe_key] = {
                'task': task,
                'trace_id': trace_id,
                'started_at': now,
                'waiters': 0,
                'upstream': upstream_label,
            }
            leader = True
            log(
                f"{trace_prefix} nonstream_dedupe_leader "
                f"upstream={upstream_label} key={dedupe_key[:16]}"
            )
        else:
            task = entry['task']
            entry['waiters'] = int(entry.get('waiters') or 0) + 1
            waiter_count = int(entry.get('waiters') or 0)
            age = now - float(entry.get('started_at') or now)
            log(
                f"{trace_prefix} nonstream_dedupe_wait "
                f"upstream={upstream_label} key={dedupe_key[:16]} "
                f"leader_trace={entry.get('trace_id')} age={age:.1f}s waiters={waiter_count}"
            )

    try:
        result = await asyncio.shield(task)
    except Exception:
        async with _nonstream_dedupe_lock:
            current = _nonstream_inflight.get(dedupe_key)
            if current and current.get('task') is task:
                _nonstream_inflight.pop(dedupe_key, None)
        raise

    async with _nonstream_dedupe_lock:
        current = _nonstream_inflight.get(dedupe_key)
        if current and current.get('task') is task:
            _nonstream_inflight.pop(dedupe_key, None)
            _nonstream_recent_results[dedupe_key] = {
                'payload': clone_jsonable(result),
                'stored_at': time.monotonic(),
                'trace_id': trace_id,
                'upstream': upstream_label,
            }
            log(
                f"{trace_prefix} nonstream_dedupe_store "
                f"upstream={upstream_label} key={dedupe_key[:16]} "
                f"waiters={current.get('waiters', 0)}"
            )

    return clone_jsonable(result), not leader






def build_codex_upstream_deps() -> CodexUpstreamDeps:
    return CodexUpstreamDeps(
        log=log,
        save_request_log=save_request_log,
        build_openai_sse_error=build_openai_sse_error,
        has_stop_tag=has_stop_tag,
        find_stop_tag=find_stop_tag,
        fmt_ms=fmt_ms,
        release_caller=release_active_stream_caller,
    )

@app.middleware("http")
async def trace_chat_completions_request(request: Request, call_next):
    """对 /v1/chat/completions 打统一入口耗时日志，定位慢点在代理前/代理内。"""
    if request.url.path != '/v1/chat/completions':
        return await call_next(request)

    trace_id = request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    request.state.trace_id = trace_id

    t0 = time.perf_counter()
    client_host = request.client.host if request.client else '?'
    content_length = request.headers.get('content-length', '?')
    transfer_encoding = request.headers.get('transfer-encoding', '-')
    log(
        f"[TRACE {trace_id}] inbound method={request.method} "
        f"path={request.url.path} from={client_host} cl={content_length} te={transfer_encoding}"
    )

    try:
        response = await call_next(request)
        log(
            f"[TRACE {trace_id}] handler_return status={getattr(response, 'status_code', '?')} "
            f"elapsed={fmt_ms(t0)}"
        )
        return response
    except Exception as e:
        log(f"[TRACE {trace_id}] handler_exception elapsed={fmt_ms(t0)} err={e}")
        raise



async def fake_slow_openai_stream(
    *,
    model: str,
    trace_id: str,
    chunks: int = 1200,
    delay: float = 0.5,
    token: str = "假流",
) -> AsyncGenerator[bytes, None]:
    """Local no-upstream SSE source used only for downstream abort diagnostics."""
    trace_prefix = f"[TRACE {trace_id}] " if trace_id else ""
    t0 = time.perf_counter()
    stream_id = f"chatcmpl-fake-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    sent = 0
    finish_status = "unknown"

    def chunk(delta: str = "", role: str | None = None, finish_reason: str | None = None) -> bytes:
        payload = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": ({"role": role} if role else ({"content": delta} if delta else {})),
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    try:
        log(
            f"{trace_prefix}fake_stream_start model={model} "
            f"chunks={chunks} delay={delay}s token_len={len(token)}"
        )
        yield chunk(role="assistant")
        for i in range(chunks):
            text = f"{token}{i:03d} "
            sent += len(text)
            yield chunk(delta=text)
            if i < 5 or i % 20 == 0:
                log(
                    f"{trace_prefix}fake_stream_emit i={i} sent_chars={sent} "
                    f"elapsed={fmt_ms(t0)}"
                )
            await asyncio.sleep(delay)
        yield chunk(finish_reason="stop")
        yield b"data: [DONE]\n\n"
        finish_status = "done"
    except asyncio.CancelledError:
        finish_status = "downstream_cancelled"
        log(
            f"{trace_prefix}fake_stream_cancelled sent_chars={sent} "
            f"elapsed={fmt_ms(t0)}"
        )
        raise
    finally:
        log(
            f"{trace_prefix}fake_stream_done reason={finish_status} "
            f"elapsed={fmt_ms(t0)} sent_chars={sent}"
        )

def get_timeout_config() -> httpx.Timeout:
    """统一超时配置，read 超时由 TIMEOUT 控制；TIMEOUT<=0 时为无限。"""
    read_timeout = float(TIMEOUT) if TIMEOUT > 0 else None
    return httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0)


# ============================================================
# Upstream Builder Helpers
# ============================================================

def build_openai_upstream_deps() -> OpenAIUpstreamDeps:
    return OpenAIUpstreamDeps(
        log=log,
        save_request_log=save_request_log,
        build_openai_sse_error=build_openai_sse_error,
        has_stop_tag=has_stop_tag,
        find_stop_tag=find_stop_tag,
        fmt_ms=fmt_ms,
        release_caller=release_active_stream_caller,
    )


def build_anthropic_upstream_deps() -> AnthropicUpstreamDeps:
    return AnthropicUpstreamDeps(
        log=log,
        save_request_log=save_request_log,
        build_openai_sse_error=build_openai_sse_error,
        has_stop_tag=has_stop_tag,
        find_stop_tag=find_stop_tag,
        fmt_ms=fmt_ms,
        release_caller=release_active_stream_caller,
    )


def build_claude_upstream_headers(session_id: str | None = None, model: Any = None, api_key: str | None = None) -> dict[str, str]:
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': api_key or CLAUDE_API_KEY,
        'anthropic-version': CLAUDE_ANTHROPIC_VERSION,
        'anthropic-beta': build_claude_beta_for_model(model),
        'anthropic-dangerous-direct-browser-access': 'true',
        'X-Stainless-Retry-Count': '0',
        'X-Stainless-Timeout': '600',
        'X-Stainless-Lang': 'js',
        'X-Stainless-Package-Version': CLAUDE_STAINLESS_PACKAGE_VERSION,
        'X-Stainless-OS': 'Linux',
        'X-Stainless-Arch': 'x64',
        'X-Stainless-Runtime': 'node',
        'X-Stainless-Runtime-Version': CLAUDE_STAINLESS_RUNTIME_VERSION,
        'x-app': 'cli',
        'User-Agent': CLAUDE_USER_AGENT,
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
    }
    if session_id:
        headers['X-Claude-Code-Session-Id'] = session_id
    return headers


def build_claude_system_prefix(model: Any = None) -> str | list[dict[str, Any]]:
    if not is_claude_haiku_model(model):
        blocks: list[dict[str, Any]] = []
        if CLAUDE_AGENT_BILLING_HEADER:
            blocks.append({'type': 'text', 'text': CLAUDE_AGENT_BILLING_HEADER})
        if CLAUDE_INCLUDE_SYSTEM_PREFIX and CLAUDE_AGENT_SYSTEM_PREFIX:
            system_prefix_block = {
                'type': 'text',
                'text': CLAUDE_AGENT_SYSTEM_PREFIX,
            }
            blocks.append(system_prefix_block)
        return blocks

    parts = [CLAUDE_BILLING_HEADER]
    if CLAUDE_INCLUDE_SYSTEM_PREFIX:
        parts.append(CLAUDE_SYSTEM_PREFIX)
    return "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def build_claude_messages_url(base_url: str | None = None) -> str:
    base = (base_url or CLAUDE_BASE_URL or '').rstrip('/')
    if not base:
        return 'https://api.anthropic.com/v1/messages?beta=true'

    normalized = base

    if normalized.endswith('/v1/messages?beta=true'):
        return normalized
    if normalized.endswith('/v1/messages'):
        return f'{normalized}?beta=true'
    return f'{normalized}/v1/messages?beta=true'


def build_claude_prompt_cache_control() -> dict[str, str] | None:
    cfg = get_claude_prompt_caching_settings()
    if not cfg['enabled']:
        return None
    cache_control = {'type': str(cfg['type'])}
    ttl = str(cfg['ttl'] or '').strip()
    if ttl:
        cache_control['ttl'] = ttl
    return cache_control


def should_strip_claude_cache_controls() -> bool:
    prompt_cfg = get_claude_prompt_caching_settings()
    return not prompt_cfg['enabled']


def strip_claude_cache_controls(payload: dict) -> dict:
    sanitized = copy.deepcopy(payload)

    def strip_inplace(value: Any) -> None:
        if isinstance(value, dict):
            value.pop('cache_control', None)
            value.pop('prompt_cache_key', None)
            value.pop('prompt_cache_retention', None)
            for child in value.values():
                strip_inplace(child)
        elif isinstance(value, list):
            for child in value:
                strip_inplace(child)

    strip_inplace(sanitized)

    return sanitized


def _remove_keywords_from_text(text: str, keywords: list[str]) -> tuple[str, dict[str, int]]:
    if not isinstance(text, str) or not text or not keywords:
        return text, {}

    sanitized = text
    removed: dict[str, int] = {}
    for keyword in keywords:
        count = sanitized.count(keyword)
        if count <= 0:
            continue
        sanitized = sanitized.replace(keyword, '')
        removed[keyword] = removed.get(keyword, 0) + count
    return sanitized, removed


def apply_claude_keyword_filter(payload: dict) -> tuple[dict, dict[str, object]]:
    cfg = get_claude_keyword_filter_settings()
    enabled = bool(cfg.get('enabled'))
    keywords = [kw for kw in cfg.get('keywords', []) if isinstance(kw, str) and kw]

    stats: dict[str, object] = {
        'enabled': enabled,
        'keywords': keywords,
        'total_removed': 0,
        'removed_keywords': {},
        'touched_paths': [],
    }
    if not enabled or not keywords:
        return payload, stats

    sanitized = copy.deepcopy(payload)
    removed_keywords: dict[str, int] = {}
    touched_paths: list[str] = []

    def record_hits(path: str, hits: dict[str, int]) -> None:
        if not hits:
            return
        touched_paths.append(path)
        for keyword, count in hits.items():
            removed_keywords[keyword] = removed_keywords.get(keyword, 0) + count

    system_blocks = sanitized.get('system')
    if isinstance(system_blocks, str):
        new_text, hits = _remove_keywords_from_text(system_blocks, keywords)
        if hits:
            sanitized['system'] = new_text
            record_hits('system', hits)
    elif isinstance(system_blocks, list):
        for idx, block in enumerate(system_blocks):
            if not isinstance(block, dict):
                continue
            text = block.get('text')
            if not isinstance(text, str):
                continue
            new_text, hits = _remove_keywords_from_text(text, keywords)
            if hits:
                block['text'] = new_text
                record_hits(f'system[{idx}]', hits)

    messages = sanitized.get('messages')
    if isinstance(messages, list):
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get('content')
            if isinstance(content, str):
                new_text, hits = _remove_keywords_from_text(content, keywords)
                if hits:
                    message['content'] = new_text
                    record_hits(f'messages[{msg_idx}].content', hits)
                continue
            if not isinstance(content, list):
                continue
            for block_idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                text = block.get('text')
                if not isinstance(text, str):
                    continue
                new_text, hits = _remove_keywords_from_text(text, keywords)
                if hits:
                    block['text'] = new_text
                    record_hits(f'messages[{msg_idx}].content[{block_idx}]', hits)

    stats['removed_keywords'] = removed_keywords
    stats['total_removed'] = sum(removed_keywords.values())
    stats['touched_paths'] = touched_paths
    return sanitized, stats


CLAUDE_OUTPUT_EFFORT_SKIP_MODELS = frozenset({
    "claude-haiku-4-5-20251001",
})


def apply_claude_output_settings(payload: dict) -> dict:
    cfg = get_claude_output_settings()
    effort = str(cfg.get('effort') or '').strip()
    if not effort:
        return payload
    model = str(payload.get('model') or '').strip()
    if model in CLAUDE_OUTPUT_EFFORT_SKIP_MODELS:
        return payload

    sanitized = copy.deepcopy(payload)
    output_config = sanitized.get('output_config')
    if isinstance(output_config, dict):
        output_config = dict(output_config)
    else:
        output_config = {}
    output_config['effort'] = effort
    sanitized['output_config'] = output_config
    return sanitized


def build_gemini_upstream_config() -> GeminiUpstreamConfig:
    return GeminiUpstreamConfig(
        base_url=GEMINI_BASE_URL,
        api_key=GEMINI_API_KEY,
        include_thoughts=GEMINI_INCLUDE_THOUGHTS,
        heartbeat_interval=GEMINI_HEARTBEAT_INTERVAL,
        max_retries=GEMINI_MAX_RETRIES,
        retry_delay=GEMINI_RETRY_DELAY,
        timeout=get_timeout_config(),
    )


def build_gemini_upstream_deps() -> GeminiUpstreamDeps:
    return GeminiUpstreamDeps(
        log=log,
        save_request_log=save_request_log,
        build_openai_sse_error=build_openai_sse_error,
    )


# ============================================================
# 路由：Anthropic Messages API 透传 (/v1/messages)
# ============================================================

@app.post('/v1/messages')
async def anthropic_messages(request: Request):
    """Anthropic Messages API 透传。"""
    data = None
    trace_id = getattr(request.state, 'trace_id', None) or request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    try:
        data = await request.json()
        if should_strip_claude_cache_controls():
            data = strip_claude_cache_controls(data)
        model = data.get('model', '')
        stream = data.get('stream', False)
        removed_fields = model_policy.apply_claude_sampling_compat(data)

        if not model_policy.is_model_allowed(model):
            save_request_log(
                model,
                data.get('messages', []),
                f"[REJECTED] Model disabled by proxy: {model}",
                stream=bool(stream),
                request_payload=data,
                error_type="invalid_model",
                trace_id=trace_id,
            )
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_model", "message": f"Model disabled by proxy: {model}"}},
                status_code=400
            )

        if removed_fields:
            log(
                "Anthropic passthrough: "
                f"removed={','.join(sorted(removed_fields))} "
                "for Claude sampling compatibility"
            )
        log(f"Anthropic passthrough: model={model}, stream={stream}")

        # NewAPI 已移除：Anthropic Messages 透传改为直连 Claude 上游。
        msg_provider, msg_real_model, _ = parse_claude_provider_prefix(model)
        msg_api_key, msg_base_url = get_claude_upstream_for_provider(msg_provider)
        if msg_provider:
            model = msg_real_model
            data['model'] = model
            log(f"Anthropic passthrough: claude_provider={msg_provider} real_model={model}")
        if not msg_api_key:
            missing_label = f"CLAUDE2_API_KEY (codecli)" if msg_provider == 'codecli' else "CLAUDE_API_KEY"
            return JSONResponse(
                {"type": "error", "error": {"type": "config_error", "message": f"{missing_label} is missing"}},
                status_code=500,
            )
        claude_session_id = extract_claude_session_id(data.get('metadata'))
        forward_headers = build_claude_upstream_headers(
            session_id=claude_session_id,
            model=model,
            api_key=msg_api_key,
        )
        target_url = build_claude_messages_url(base_url=msg_base_url)

        if stream:
            # 流式：透传
            return StreamingResponse(
                forward_anthropic_stream(
                    url=target_url,
                    request_data=data,
                    headers=forward_headers,
                    timeout=get_timeout_config(),
                    deps=build_anthropic_upstream_deps(),
                    enable_early_stop=False,
                    trace_id=trace_id,
                ),
                media_type='text/event-stream'
            )

        # 非流式：内部仍走上游流式，再聚合回标准 Anthropic JSON
        try:
            resp_data, raw_sse = await collect_anthropic_message_response(
                url=target_url,
                request_data=data,
                headers=forward_headers,
                model=model,
                timeout=get_timeout_config(),
                max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                deps=build_anthropic_upstream_deps(),
                trace_id=trace_id,
            )
            save_request_log(
                model,
                data.get('messages', []),
                json.dumps(resp_data, ensure_ascii=False),
                stream=bool(stream),
                raw_sse=raw_sse,
                request_payload=data,
                trace_id=trace_id,
            )
            return JSONResponse(resp_data, status_code=200)
        except Exception as e:
            err_text = str(e)
            save_request_log(
                model,
                data.get('messages', []),
                f"[UPSTREAM_HTTP_ERROR] {err_text}",
                stream=bool(stream),
                request_payload=data,
                error_type="upstream_http_error",
                trace_id=trace_id,
            )
            return JSONResponse(
                {"type": "error", "error": {"type": "upstream_http_error", "message": err_text}},
                status_code=502,
            )

    except Exception as e:
        log(f"Anthropic passthrough error: {e}")
        if isinstance(data, dict):
            save_request_log(
                data.get('model', ''),
                data.get('messages', []),
                f"[EXCEPTION] {e}",
                stream=bool(data.get('stream', False)),
                request_payload=data,
                error_type="proxy_error",
                trace_id=trace_id,
            )
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"type": "error", "error": {"type": "proxy_error", "message": str(e)}},
            status_code=500
        )


# ============================================================
# 路由：OpenAI Chat Completions (/v1/chat/completions)
# ============================================================

@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    data = None
    inbound_request = None
    inbound_structure = None
    trace_id = getattr(request.state, 'trace_id', None) or request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    trace_prefix = f"[TRACE {trace_id}]"
    route_t0 = time.perf_counter()
    caller_key, caller_desc = build_caller_fingerprint(request)
    try:
        json_t0 = time.perf_counter()
        data = await request.json()
        inbound_request = copy.deepcopy(data)
        inbound_structure = summarize_openai_messages(inbound_request.get('messages', []))
        log(f"{trace_prefix} json_parsed elapsed={fmt_ms(json_t0)} since_enter={fmt_ms(route_t0)}")
        log(
            f"{trace_prefix} inbound_openai_structure "
            f"{json.dumps(inbound_structure[:12], ensure_ascii=False)}"
        )
        model = data.get('model', '')
        stream = data.get('stream', False)
        removed_fields = model_policy.apply_claude_sampling_compat(data)
        msg_count = len(data.get('messages', []))
        payload_sig = (
            f"model={model} stream={stream} msgs={msg_count} "
            f"cl={request.headers.get('content-length', '-')}"
        )
        if removed_fields:
            log(
                f"{trace_prefix} claude_sampling_compat "
                f"removed={','.join(sorted(removed_fields))}"
            )
        log(f"{trace_prefix} caller={caller_key} {caller_desc}")
        log(f"{trace_prefix} request_meta {payload_sig}")

        if stream:
            prev = active_stream_registry.get(caller_key)
            if prev and prev.get('trace_id') != trace_id:
                prev_age = time.time() - prev.get('started_at', time.time())
                log(
                    f"{trace_prefix} caller_overlap caller={caller_key} "
                    f"prev_trace={prev.get('trace_id')} prev_age={prev_age:.1f}s "
                    f"prev_model={prev.get('model')} prev_msgs={prev.get('msg_count')} "
                    f"note=new stream from same caller may cancel previous stream"
                )
            active_stream_registry.register(
                caller_key,
                trace_id=trace_id,
                model=model,
                msg_count=msg_count,
            )

        if str(model).startswith("fake-slow-stream"):
            fake_chunks = int(data.get('fake_chunks') or 1200)
            fake_delay = float(data.get('fake_delay') or 0.5)
            fake_token = str(data.get('fake_token') or '假流')
            log(
                f"{trace_prefix} Route to local fake slow stream "
                f"model={model} chunks={fake_chunks} delay={fake_delay} since_enter={fmt_ms(route_t0)}"
            )
            if stream:
                return StreamingResponse(
                    fake_slow_openai_stream(
                        model=str(model),
                        trace_id=trace_id,
                        chunks=fake_chunks,
                        delay=fake_delay,
                        token=fake_token,
                    ),
                    media_type='text/event-stream',
                )
            return JSONResponse({
                "id": f"chatcmpl-fake-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": fake_token},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        if not model_policy.is_model_allowed(model):
            save_request_log(
                model,
                data.get('messages', []),
                f"[REJECTED] Model disabled by proxy: {model}",
                stream=bool(stream),
                request_payload=data,
                error_type="invalid_model",
                trace_id=trace_id,
            )
            return JSONResponse(
                {"error": {"message": f"Model disabled by proxy: {model}", "type": "invalid_model"}},
                status_code=400
            )

        directory_route = await resolve_openai_compatible_route(
            model,
            request_authorization=request.headers.get('authorization', ''),
        )
        if directory_route:
            route_name = directory_route.get('name') or 'openai-compatible'
            base_url = str(directory_route.get('base_url') or '').rstrip('/')
            target_url = f"{base_url}/chat/completions"
            outbound_data = copy.deepcopy(data)
            if should_append_pro_opus46_last_user_note(route_name, base_url, model):
                illustrated = insert_after_latest_human_message(
                    outbound_data.get('messages', []),
                    PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
                    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
                )
                if illustrated:
                    log(f"{trace_prefix} pro_opus46_latest_human_illustration inserted name={route_name} model={model}")
                else:
                    log(f"{trace_prefix} pro_opus46_latest_human_illustration skipped name={route_name} model={model}")
                injected = append_to_last_user_message(
                    outbound_data.get('messages', []),
                    PRO_OPUS46_LAST_USER_APPEND_TEXT,
                    PRO_OPUS46_LAST_USER_APPEND_MARKER,
                )
                if injected:
                    log(f"{trace_prefix} pro_opus46_last_user_note appended name={route_name} model={model}")
                else:
                    log(f"{trace_prefix} pro_opus46_last_user_note skipped name={route_name} model={model}")
            if should_apply_deepseek_drawing_context_filter(route_name, base_url):
                filter_stats = apply_drawing_context_filter(outbound_data)
                if filter_stats.get('blocks'):
                    log(
                        f"{trace_prefix} ds_drawing_context_filter "
                        f"name={route_name} messages={filter_stats['messages']} "
                        f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                        f"mode=closed_tags_and_bare_image_prompts"
                    )
            if model_policy.is_gemini_model(model):
                filter_stats = apply_drawing_context_filter(outbound_data)
                if filter_stats.get('blocks'):
                    log(
                        f"{trace_prefix} pro_gemini_drawing_context_filter "
                        f"name={route_name} model={model} messages={filter_stats['messages']} "
                        f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                        f"mode=closed_tags_and_bare_image_prompts"
                    )
            no_reasoning_meta = apply_pro_no_reasoning_payload(outbound_data, route_name, base_url)
            if no_reasoning_meta:
                log(
                    f"{trace_prefix} pro_no_reasoning_payload "
                    f"name={route_name} model={model} "
                    f"reasoning_effort={no_reasoning_meta.get('reasoning_effort')} "
                    f"removed={','.join(no_reasoning_meta.get('removed') or []) or '-'}"
                )
            headers = {
                'Content-Type': 'application/json',
            }
            route_key = str(directory_route.get('api_key') or '').strip()
            if route_key:
                headers['Authorization'] = f'Bearer {route_key}'
            else:
                inbound_auth = request.headers.get('authorization')
                if inbound_auth:
                    headers['Authorization'] = inbound_auth
            log(
                f"{trace_prefix} Route to model-directory upstream "
                f"name={route_name} model={model} url={target_url} since_enter={fmt_ms(route_t0)}"
            )

            replay_spec = claude_replay.load_spec(
                model=model,
                messages=data.get('messages', []),
            )
            if replay_spec:
                log(
                    f"{trace_prefix} model-directory OpenAI replay armed "
                    f"name={route_name} mode={replay_spec['mode']} "
                    f"file={replay_spec['raw_sse_path']} since_enter={fmt_ms(route_t0)}"
                )

            if stream:
                pro_gemini_nonstream_replay = bool(model_policy.is_gemini_model(model))
                outbound_data['stream'] = False if pro_gemini_nonstream_replay else True
                if replay_spec:
                    claude_replay.consume_if_needed(replay_spec)
                    return StreamingResponse(
                        replay_openai_chat_stream(
                            raw_sse_text=replay_spec['raw_sse_text'],
                            request_data=outbound_data,
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                            deps=build_openai_upstream_deps(),
                            enable_early_stop=False,
                        ),
                        media_type='text/event-stream'
                    )

                if pro_gemini_nonstream_replay:
                    log(
                        f"{trace_prefix} model-directory pro_gemini stream_to_nonstream_replay "
                        f"name={route_name} model={model}"
                    )
                    return StreamingResponse(
                        forward_non_stream_as_openai_stream(
                            url=target_url,
                            request_data=outbound_data,
                            headers=headers,
                            timeout=get_timeout_config(),
                            deps=build_openai_upstream_deps(),
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                        ),
                        media_type='text/event-stream'
                    )

                return StreamingResponse(
                    forward_stream(
                        url=target_url,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=build_openai_upstream_deps(),
                        enable_early_stop=False,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                    ),
                    media_type='text/event-stream'
                )

            try:
                use_plain_non_stream = bool(model_policy.is_gemini_model(model))
                upstream_nonstream_payload = None
                if use_plain_non_stream:
                    outbound_data['stream'] = False
                    dedupe_key = build_exact_request_key(outbound_data)
                    log(
                        f"{trace_prefix} model-directory Gemini nonstream passthrough "
                        f"name={route_name} model={model} key={dedupe_key[:16]}"
                    )

                    async def _run_plain_nonstream():
                        collected = await collect_non_stream(
                            url=target_url,
                            request_data=outbound_data,
                            headers=headers,
                            timeout=get_timeout_config(),
                            deps=build_openai_upstream_deps(),
                            trace_id=trace_id,
                        )
                        return collected[5]

                    upstream_nonstream_payload, dedupe_shared = await run_exact_nonstream_once(
                        dedupe_key=dedupe_key,
                        trace_id=trace_id,
                        upstream_label=f"{route_name}:{model}",
                        runner=_run_plain_nonstream,
                    )
                    full_content = extract_openai_chat_payload_content(upstream_nonstream_payload)
                    model_name = str(upstream_nonstream_payload.get("model") or model)
                    usage = upstream_nonstream_payload.get("usage") if isinstance(upstream_nonstream_payload.get("usage"), dict) else {}
                    finish_reason = "stop"
                    try:
                        choices = upstream_nonstream_payload.get("choices")
                        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                            finish_reason = str(choices[0].get("finish_reason") or "stop")
                    except Exception:
                        pass
                    raw_response = json.dumps(upstream_nonstream_payload, ensure_ascii=False)
                    if dedupe_shared:
                        log(
                            f"{trace_prefix} nonstream_dedupe_return_shared "
                            f"key={dedupe_key[:16]} out_chars={len(full_content)}"
                        )
                else:
                    full_content, model_name, usage, finish_reason, raw_response = await collect_stream(
                        url=target_url,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=build_openai_upstream_deps(),
                        enable_early_stop=False,
                        trace_id=trace_id,
                    )
                save_request_log(
                    model,
                    data.get('messages', []),
                    full_content,
                    stream=False,
                    raw_sse="" if use_plain_non_stream else raw_response,
                    request_payload=outbound_data,
                    trace_id=trace_id,
                )
                if use_plain_non_stream and isinstance(upstream_nonstream_payload, dict):
                    log(
                        f"{trace_prefix} model-directory Gemini nonstream return_shape "
                        f"keys={list(upstream_nonstream_payload.keys())[:12]} "
                        f"choices_type={type(upstream_nonstream_payload.get('choices')).__name__} "
                        f"has_content={'content' in upstream_nonstream_payload} "
                        f"has_message={'message' in upstream_nonstream_payload}"
                    )
                    return JSONResponse(upstream_nonstream_payload)
                return JSONResponse({
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name or model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": usage or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })
            except Exception as e:
                log(f"{trace_prefix} model-directory collect error upstream={route_name}: {e}")
                save_request_log(
                    model,
                    data.get('messages', []),
                    f"[ERROR] {e}",
                    stream=False,
                    request_payload=outbound_data,
                    error_type="model_directory_upstream_error",
                    trace_id=trace_id,
                )
                return JSONResponse(
                    {"error": {"message": str(e), "type": "model_directory_upstream_error"}},
                    status_code=502
                )

        # Gemini 模型：直接 HTTP 直连（不再依赖 gemini-proxy）
        if model_policy.is_gemini_model(model):
            if not GEMINI_ENABLED:
                return JSONResponse(
                    {"error": {"message": "Gemini is temporarily disabled by proxy", "type": "service_unavailable"}},
                    status_code=503
                )

            replay_spec = claude_replay.load_spec(
                model=model,
                messages=data.get('messages', []),
            )
            if replay_spec:
                log(
                    f"{trace_prefix} Gemini replay armed "
                    f"mode={replay_spec['mode']} file={replay_spec['raw_sse_path']} "
                    f"since_enter={fmt_ms(route_t0)}"
                )
                if stream:
                    data = dict(data)
                    data['stream'] = True
                    claude_replay.consume_if_needed(replay_spec)
                    return StreamingResponse(
                        replay_anthropic_chat_stream(
                            raw_sse_text=replay_spec['raw_sse_text'],
                            request_data=data,
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                            deps=build_anthropic_upstream_deps(),
                        ),
                        media_type='text/event-stream'
                    )

                try:
                    replay_request = dict(data)
                    replay_request['stream'] = True
                    claude_replay.consume_if_needed(replay_spec)
                    full_content, model_name, usage, finish_reason, _raw_response = await collect_anthropic_chat_completion_from_raw_sse(
                        raw_sse_text=replay_spec['raw_sse_text'],
                        request_data=replay_request,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        deps=build_anthropic_upstream_deps(),
                    )
                    save_request_log(
                        model_name,
                        data.get('messages', []),
                        full_content,
                        stream=False,
                        raw_sse=replay_spec['raw_sse_text'],
                        request_payload=data,
                        trace_id=trace_id,
                    )
                    return JSONResponse({
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_content
                            },
                            "finish_reason": finish_reason
                        }],
                        "usage": usage or {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    })
                except Exception as e:
                    log(f"{trace_prefix} Gemini replay collect error: {e}")
                    save_request_log(
                        model,
                        data.get('messages', []),
                        f"[ERROR] {e}",
                        stream=False,
                        request_payload=data,
                        error_type="gemini_replay_error",
                        trace_id=trace_id,
                    )
                    return JSONResponse(
                        {"error": {"message": str(e), "type": "gemini_replay_error"}},
                        status_code=502
                    )

            log(f"{trace_prefix} Route to Gemini HTTP: {model} since_enter={fmt_ms(route_t0)}")
            if not GEMINI_API_KEY:
                return JSONResponse(
                    {"error": {"message": "GEMINI_API_KEY is missing", "type": "config_error"}},
                    status_code=500
                )
            data = copy.deepcopy(data)
            filter_stats = apply_drawing_context_filter(data)
            if filter_stats.get('blocks'):
                log(
                    f"{trace_prefix} gemini_drawing_context_filter "
                    f"model={model} messages={filter_stats['messages']} "
                    f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                    f"mode=closed_tags_and_bare_image_prompts"
                )
            gemini_config = build_gemini_upstream_config()
            gemini_deps = build_gemini_upstream_deps()

            if stream:
                data['stream'] = True
                return StreamingResponse(
                    forward_gemini_stream(
                        model=model,
                        openai_request=data,
                        config=gemini_config,
                        deps=gemini_deps,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                    ),
                    media_type='text/event-stream'
                )

            try:
                data['stream'] = False
                dedupe_key = build_exact_request_key(data)

                async def _run_gemini_nonstream():
                    full_content_inner, usage_inner, finish_reason_inner = await collect_gemini_non_stream(
                        model=model,
                        openai_request=data,
                        config=gemini_config,
                        deps=gemini_deps,
                        trace_id=trace_id,
                    )
                    return {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_content_inner
                            },
                            "finish_reason": finish_reason_inner
                        }],
                        "usage": usage_inner or {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }

                response_payload, dedupe_shared = await run_exact_nonstream_once(
                    dedupe_key=dedupe_key,
                    trace_id=trace_id,
                    upstream_label=f"gemini-http:{model}",
                    runner=_run_gemini_nonstream,
                )
                full_content = extract_openai_chat_payload_content(response_payload)
                save_request_log(
                    model,
                    data.get('messages', []),
                    full_content,
                    stream=False,
                    request_payload=data,
                    trace_id=trace_id,
                )
                if dedupe_shared:
                    log(
                        f"{trace_prefix} nonstream_dedupe_return_shared "
                        f"key={dedupe_key[:16]} out_chars={len(full_content)}"
                    )
                return JSONResponse(response_payload)
            except Exception as e:
                log(f"Gemini collect error: {e}")
                schedule_delayed_restart()
                save_request_log(
                    model,
                    data.get('messages', []),
                    f"[ERROR] {e}",
                    stream=False,
                    request_payload=data,
                    error_type="gemini_upstream_error",
                    trace_id=trace_id,
                )
                return JSONResponse(
                    {"error": {"message": str(e), "type": "gemini_upstream_error"}},
                    status_code=502
                )

        if model_policy.is_claude_model(model):
            claude_provider, claude_real_model, claude_display = parse_claude_provider_prefix(model)
            claude_upstream_key, claude_upstream_base = get_claude_upstream_for_provider(claude_provider)
            if not claude_upstream_key:
                if stream:
                    release_active_stream_caller(caller_key, trace_id)
                missing_label = f"CLAUDE2_API_KEY (codecli)" if claude_provider == 'codecli' else "CLAUDE_API_KEY"
                return JSONResponse(
                    {"error": {"message": f"{missing_label} is missing", "type": "config_error"}},
                    status_code=500
                )
            # Use real model name (without prefix) for upstream requests
            if claude_provider:
                model = claude_real_model
                data = copy.deepcopy(data)
                data['model'] = model
                log(f"{trace_prefix} claude_provider={claude_provider} real_model={model}")

            if is_claude_haiku_model(model):
                data = copy.deepcopy(data)
                filter_stats = apply_drawing_context_filter(data)
                if filter_stats.get('blocks'):
                    log(
                        f"{trace_prefix} haiku_drawing_context_filter "
                        f"model={model} messages={filter_stats['messages']} "
                        f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                        f"mode=closed_tags_and_bare_image_prompts"
                    )

            if should_strip_claude_cache_controls():
                data = strip_claude_cache_controls(data)

            data = copy.deepcopy(data)
            if is_claude_opus_model(model):
                illustrated = insert_after_latest_human_message(
                    data.get('messages', []),
                    PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
                    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
                )
                if illustrated:
                    log(f"{trace_prefix} claude_code_latest_human_illustration inserted model={model}")
                else:
                    log(f"{trace_prefix} claude_code_latest_human_illustration skipped model={model}")
                injected = append_to_last_user_message(
                    data.get('messages', []),
                    PRO_OPUS_LAST_USER_APPEND_TEXT,
                    PRO_OPUS_LAST_USER_APPEND_MARKER,
                )
                if injected:
                    log(f"{trace_prefix} claude_code_last_user_note appended model={model}")
                else:
                    log(f"{trace_prefix} claude_code_last_user_note skipped model={model}")
            else:
                log(f"{trace_prefix} claude_code_last_user_note skipped_non_opus model={model}")

            claude_metadata_user_id, claude_session_id, claude_session_mode, claude_session_ttl_sec, claude_session_key = build_timed_claude_user_id(
                model=model,
                provider=claude_provider or 'default',
            )
            prompt_cache_cfg = get_claude_prompt_caching_settings()
            prompt_cache_enabled_for_model = bool(
                prompt_cache_cfg.get('enabled') and is_claude_prompt_cache_model(model)
            )
            prompt_cache_keepalive_cfg = get_claude_cache_keepalive_settings()
            prompt_cache_keepalive_enabled_for_model = bool(
                prompt_cache_enabled_for_model and prompt_cache_keepalive_cfg.get('enabled')
            )
            prompt_cache_control = build_claude_prompt_cache_control() if prompt_cache_enabled_for_model else None
            prompt_cache_strategy = (
                'sonnet_fill_table' if is_claude_sonnet_model(model)
                else 'opus_roleplay_layered' if is_claude_opus_model(model)
                else 'generic'
            )
            claude_request = convert_chat_to_anthropic_messages_request(
                data,
                system_prefix=build_claude_system_prefix(model),
                metadata_user_id=claude_metadata_user_id,
                default_max_tokens=CLAUDE_DEFAULT_MAX_TOKENS,
                prompt_cache_control=prompt_cache_control if prompt_cache_enabled_for_model and prompt_cache_cfg.get('mode') == 'explicit' else None,
                prompt_cache_strategy=prompt_cache_strategy,
            )
            if prompt_cache_enabled_for_model and prompt_cache_cfg.get('mode') == 'automatic' and isinstance(prompt_cache_control, dict):
                claude_request['cache_control'] = dict(prompt_cache_control)
            claude_request = apply_claude_output_settings(claude_request)
            claude_request, keyword_filter_stats = apply_claude_keyword_filter(claude_request)
            claude_request, compat_meta = apply_claude_client_compat_request(claude_request)
            claude_request, model_compat_meta = apply_claude_model_compat_request(claude_request)
            for key, value in model_compat_meta.items():
                compat_meta[f'model_{key}'] = value
            if should_strip_claude_cache_controls() and is_claude_haiku_model(model):
                claude_request = strip_claude_cache_controls(claude_request)
                compat_meta['cache_control'] = 'stripped'
            elif should_strip_claude_cache_controls():
                compat_meta['cache_control'] = 'preserved_agent_system'
            claude_structure = summarize_anthropic_request(claude_request)
            claude_cache_breakpoint_layout = summarize_anthropic_cache_breakpoints(claude_request)
            claude_headers = build_claude_upstream_headers(
                session_id=claude_session_id,
                model=claude_request.get('model'),
                api_key=claude_upstream_key,
            )
            claude_url = build_claude_messages_url(base_url=claude_upstream_base)
            replay_spec = claude_replay.load_spec(
                model=model,
                messages=data.get('messages', []),
            )

            log(
                "Claude messages shim: "
                f"model={model}, "
                f"anthropic_msgs={len(claude_request.get('messages', []))}, "
                f"system_blocks={len(claude_request.get('system', [])) if isinstance(claude_request.get('system'), list) else 0}, "
                f"max_tokens={claude_request.get('max_tokens')}, "
                f"output_effort={(claude_request.get('output_config') or {}).get('effort', '-')}, "
                f"compat_tools={compat_meta.get('tools')}, "
                f"compat_thinking={compat_meta.get('thinking')}, "
                f"compat_ctx={compat_meta.get('context_management')}, "
                f"session={claude_session_mode}:{claude_session_id[:8]}, "
                f"session_key={claude_session_key}, "
                f"session_ttl={claude_session_ttl_sec:.0f}s, "
                f"cache_mode={prompt_cache_cfg.get('mode') if prompt_cache_enabled_for_model else 'off'}, "
                f"cache_strategy={prompt_cache_strategy if prompt_cache_enabled_for_model else '-'}, "
                f"cache_keepalive={'on' if prompt_cache_keepalive_enabled_for_model else 'off'}, "
                f"top_cache={'yes' if isinstance(claude_request.get('cache_control'), dict) else 'no'}, "
                f"cache_breakpoints={sum(1 for item in claude_structure.get('system_blocks', []) if item.get('has_cache_control')) + sum(item.get('cache_control_blocks', 0) for item in claude_structure.get('messages', []))}"
            )
            if prompt_cache_enabled_for_model:
                log(
                    f"{trace_prefix}Claude cache breakpoint layout: "
                    f"{json.dumps(claude_cache_breakpoint_layout, ensure_ascii=False, separators=(',', ':'))}"
                )
            if int(keyword_filter_stats.get('total_removed', 0) or 0) > 0:
                log(
                    f"{trace_prefix} Claude keyword filter "
                    f"removed={json.dumps(keyword_filter_stats.get('removed_keywords', {}), ensure_ascii=False)} "
                    f"touched_paths={json.dumps(keyword_filter_stats.get('touched_paths', []), ensure_ascii=False)}"
                )
            log(f"{trace_prefix} Route to Claude upstream /v1/messages: {model} since_enter={fmt_ms(route_t0)}")
            if replay_spec:
                log(
                    f"{trace_prefix} Claude replay armed "
                    f"mode={replay_spec['mode']} file={replay_spec['raw_sse_path']}"
                )

            if stream:
                claude_request = dict(claude_request)
                claude_request['stream'] = True
                if is_claude_haiku_model(model):
                    claude_request, system_fold_meta = fold_claude_system_into_first_user_message(claude_request)
                    if system_fold_meta.get('action') != 'absent':
                        log(
                            f"{trace_prefix} Claude stream system fold "
                            f"action={system_fold_meta.get('action')} "
                            f"blocks={system_fold_meta.get('blocks')} "
                            f"chars={system_fold_meta.get('chars')}"
                        )
                if replay_spec:
                    claude_replay.consume_if_needed(replay_spec)
                    return StreamingResponse(
                        replay_anthropic_chat_stream(
                            raw_sse_text=replay_spec['raw_sse_text'],
                            request_data=claude_request,
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                            deps=build_anthropic_upstream_deps(),
                        ),
                        media_type='text/event-stream'
                    )
                return StreamingResponse(
                    forward_anthropic_chat_stream(
                        url=claude_url,
                        request_data=claude_request,
                        headers=claude_headers,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=build_anthropic_upstream_deps(),
                        cache_keepalive=prompt_cache_keepalive_cfg if prompt_cache_keepalive_enabled_for_model else None,
                    ),
                    media_type='text/event-stream'
                )

            try:
                claude_request = dict(claude_request)
                claude_request['stream'] = True
                if is_claude_haiku_model(model):
                    claude_request, system_fold_meta = fold_claude_system_into_first_user_message(claude_request)
                    if system_fold_meta.get('action') != 'absent':
                        log(
                            f"{trace_prefix} Claude collect system fold "
                            f"action={system_fold_meta.get('action')} "
                            f"blocks={system_fold_meta.get('blocks')} "
                            f"chars={system_fold_meta.get('chars')}"
                        )
                if replay_spec:
                    claude_replay.consume_if_needed(replay_spec)
                    full_content, model_name, usage, finish_reason, raw_response = await collect_anthropic_chat_completion_from_raw_sse(
                        raw_sse_text=replay_spec['raw_sse_text'],
                        request_data=claude_request,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        deps=build_anthropic_upstream_deps(),
                    )
                else:
                    full_content, model_name, usage, finish_reason, raw_response = await collect_anthropic_chat_completion(
                        url=claude_url,
                        request_data=claude_request,
                        headers=claude_headers,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=build_anthropic_upstream_deps(),
                    )
                save_request_log(
                    model,
                    data.get('messages', []),
                    full_content,
                    stream=False,
                    raw_sse=raw_response,
                    request_payload=claude_request,
                    inbound_request_payload=inbound_request,
                    debug_meta={
                        "inbound_openai_structure": inbound_structure,
                    },
                    trace_id=trace_id,
                )
                return JSONResponse({
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name or model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": usage or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })
            except Exception as e:
                log(f"{trace_prefix} Claude collect error: {e}")
                save_request_log(
                    model,
                    data.get('messages', []),
                    f"[ERROR] {e}",
                    stream=False,
                    request_payload=claude_request,
                    inbound_request_payload=inbound_request,
                    debug_meta={
                        "inbound_openai_structure": inbound_structure,
                    },
                    error_type="claude_upstream_error",
                    trace_id=trace_id,
                )
                return JSONResponse(
                    {"error": {"message": str(e), "type": "claude_upstream_error"}},
                    status_code=502
                )

        # GPT：对齐 Codex CLI，默认将 Chat Completions 入站转为 Responses 上游。
        outbound_data = data

        if model_policy.is_gpt_model(model):
            outbound_data = inject_gpt_usage_policies_system_message(data)
            log(f"{trace_prefix} gpt_usage_policies_system_injected first_system=yes")
            if not CODEX_API_KEY:
                if stream:
                    release_active_stream_caller(caller_key, trace_id)
                return JSONResponse(
                    {"error": {"message": "CODEX_API_KEY is missing", "type": "config_error"}},
                    status_code=500
                )

            if GPT_USE_RESPONSES:
                outbound_data = convert_chat_to_responses_request(outbound_data)
                outbound_data = dict(outbound_data)
                outbound_data['stream'] = True
                outbound_data.setdefault('store', False)
                if GPT_SERVICE_TIER:
                    outbound_data['service_tier'] = GPT_SERVICE_TIER
                outbound_data.setdefault('prompt_cache_key', build_gpt_prompt_cache_key(model, outbound_data, GPT_PROMPT_CACHE_KEY))
                if GPT_PROMPT_CACHE_RETENTION:
                    outbound_data.setdefault('prompt_cache_retention', GPT_PROMPT_CACHE_RETENTION)
                outbound_data.setdefault('reasoning', {})
                if isinstance(outbound_data.get('reasoning'), dict):
                    outbound_data['reasoning'].setdefault('effort', CODEX_REASONING_EFFORT)
                    if CODEX_REASONING_SUMMARY:
                        outbound_data['reasoning'].setdefault('summary', CODEX_REASONING_SUMMARY)

                codex_deps = build_codex_upstream_deps()
                log(
                    f"{trace_prefix} Route to GPT Responses upstream /v1/responses: "
                    f"{model} service_tier={outbound_data.get('service_tier', '-')} "
                    f"prompt_cache_key={outbound_data.get('prompt_cache_key', '-')} "
                    f"prompt_cache_retention={outbound_data.get('prompt_cache_retention', '-')} "
                    f"reasoning_effort={(outbound_data.get('reasoning') or {}).get('effort', '-')} "
                    f"input_items={len(outbound_data.get('input', []))} "
                    f"instructions={'yes' if outbound_data.get('instructions') else 'no'} "
                    f"since_enter={fmt_ms(route_t0)}"
                )

                if stream:
                    return StreamingResponse(
                        forward_codex_chat_stream(
                            url=CODEX_BASE_URL,
                            api_key=CODEX_API_KEY,
                            request_data=outbound_data,
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                            timeout=get_timeout_config(),
                            max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                            deps=codex_deps,
                        ),
                        media_type='text/event-stream'
                    )

                try:
                    full_content, usage, finish_reason = await collect_codex_chat_completion(
                        url=CODEX_BASE_URL,
                        api_key=CODEX_API_KEY,
                        request_data=outbound_data,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=codex_deps,
                    )
                    return JSONResponse({
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_content
                            },
                            "finish_reason": finish_reason
                        }],
                        "usage": usage or {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    })
                except Exception as e:
                    log(f"{trace_prefix} GPT Responses collect error: {e}")
                    save_request_log(
                        model,
                        data.get('messages', []),
                        f"[ERROR] {e}",
                        stream=False,
                        request_payload=outbound_data,
                        error_type="gpt_responses_upstream_error",
                        trace_id=trace_id,
                    )
                    return JSONResponse(
                        {"error": {"message": str(e), "type": "gpt_responses_upstream_error"}},
                        status_code=502
                    )

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {CODEX_API_KEY}',
            }
            outbound_data = dict(outbound_data)
            if GPT_SERVICE_TIER:
                outbound_data['service_tier'] = GPT_SERVICE_TIER
            log(
                f"{trace_prefix} Route to GPT upstream /v1/chat/completions: "
                f"{model} service_tier={outbound_data.get('service_tier', '-')} "
                f"since_enter={fmt_ms(route_t0)}"
            )

            if stream:
                outbound_data['stream'] = True
                return StreamingResponse(
                    forward_stream(
                        url=GPT_BASE_URL,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=build_openai_upstream_deps(),
                        enable_early_stop=False,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                    ),
                    media_type='text/event-stream'
                )

            try:
                full_content, model_name, usage, finish_reason, raw_sse = await collect_stream(
                    url=GPT_BASE_URL,
                    request_data=outbound_data,
                    headers=headers,
                    timeout=get_timeout_config(),
                    max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                    deps=build_openai_upstream_deps(),
                    enable_early_stop=False,
                    trace_id=trace_id,
                )

                save_request_log(
                    model,
                    data.get('messages', []),
                    full_content,
                    stream=False,
                    raw_sse=raw_sse,
                    request_payload=outbound_data,
                    trace_id=trace_id,
                )

                return JSONResponse({
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name or model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": usage or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })
            except Exception as e:
                log(f"{trace_prefix} GPT collect error: {e}")
                save_request_log(
                    model,
                    data.get('messages', []),
                    f"[ERROR] {e}",
                    stream=False,
                    request_payload=outbound_data,
                    error_type="gpt_upstream_error",
                    trace_id=trace_id,
                )
                return JSONResponse(
                    {"error": {"message": str(e), "type": "gpt_upstream_error"}},
                    status_code=502
                )

        if model_policy.is_codex_model(model):
            removed_fields = model_policy.apply_codex_reasoning(data)
            outbound_data = convert_chat_to_responses_request(data)
            if removed_fields:
                log(
                    "Codex responses shim: "
                    f"model={model}, "
                    f"input_items={len(outbound_data.get('input', []))}, "
                    f"instructions={'yes' if outbound_data.get('instructions') else 'no'}, "
                    f"removed={','.join(sorted(removed_fields))}"
                )
            else:
                log(
                    "Codex responses shim: "
                    f"model={model}, "
                    f"input_items={len(outbound_data.get('input', []))}, "
                    f"instructions={'yes' if outbound_data.get('instructions') else 'no'}"
                )
            if not CODEX_API_KEY:
                if stream:
                    release_active_stream_caller(caller_key, trace_id)
                return JSONResponse(
                    {"error": {"message": "CODEX_API_KEY is missing", "type": "config_error"}},
                    status_code=500
                )

            outbound_data = dict(outbound_data)
            outbound_data['stream'] = True
            codex_deps = build_codex_upstream_deps()
            log(f"{trace_prefix} Route to Codex upstream /v1/responses: {model} since_enter={fmt_ms(route_t0)}")

            if stream:
                return StreamingResponse(
                    forward_codex_chat_stream(
                        url=CODEX_BASE_URL,
                        api_key=CODEX_API_KEY,
                        request_data=outbound_data,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=codex_deps,
                    ),
                    media_type='text/event-stream'
                )

            full_content, usage, finish_reason = await collect_codex_chat_completion(
                url=CODEX_BASE_URL,
                api_key=CODEX_API_KEY,
                request_data=outbound_data,
                model=model,
                messages=data.get('messages', []),
                trace_id=trace_id,
                timeout=get_timeout_config(),
                max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                deps=codex_deps,
            )
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    },
                    "finish_reason": finish_reason
                }],
                "usage": usage or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            })

        # 未出现在 /v1/models 的自填 model id：按模型名族群选择 OpenAI-compatible 上游透传。
        passthrough_upstream = resolve_model_name_passthrough_upstream(model)
        if passthrough_upstream:
            route_name = passthrough_upstream.get('name') or 'openai-compatible-passthrough'
            route_reason = passthrough_upstream.get('selection_reason') or 'unknown'
            base_url = str(passthrough_upstream.get('base_url') or '').rstrip('/')
            target_url = f"{base_url}/chat/completions"
            outbound_data = copy.deepcopy(data)
            if should_apply_deepseek_drawing_context_filter(route_name, base_url):
                filter_stats = apply_drawing_context_filter(outbound_data)
                if filter_stats.get('blocks'):
                    log(
                        f"{trace_prefix} model_name_passthrough_drawing_context_filter "
                        f"name={route_name} messages={filter_stats['messages']} "
                        f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                        f"mode=closed_tags_and_bare_image_prompts"
                    )
            headers = {'Content-Type': 'application/json'}
            route_key = str(passthrough_upstream.get('api_key') or '').strip()
            if route_key:
                headers['Authorization'] = f'Bearer {route_key}'
            else:
                inbound_auth = request.headers.get('authorization')
                if inbound_auth:
                    headers['Authorization'] = inbound_auth
            log(
                f"{trace_prefix} model_name_passthrough "
                f"name={route_name} reason={route_reason} model={model} "
                f"url={target_url} since_enter={fmt_ms(route_t0)}"
            )
            if stream:
                outbound_data['stream'] = True
                return StreamingResponse(
                    forward_stream(
                        url=target_url,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=get_timeout_config(),
                        max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                        deps=build_openai_upstream_deps(),
                        enable_early_stop=False,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                    ),
                    media_type='text/event-stream'
                )
            try:
                full_content, model_name, usage, finish_reason, raw_sse = await collect_stream(
                    url=target_url,
                    request_data=outbound_data,
                    headers=headers,
                    timeout=get_timeout_config(),
                    max_raw_sse_bytes=MAX_RAW_SSE_BYTES,
                    deps=build_openai_upstream_deps(),
                    enable_early_stop=False,
                    trace_id=trace_id,
                )
                save_request_log(
                    model,
                    data.get('messages', []),
                    full_content,
                    stream=False,
                    raw_sse=raw_sse,
                    request_payload=outbound_data,
                    trace_id=trace_id,
                )
                return JSONResponse({
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name or model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                        },
                        "finish_reason": finish_reason,
                    }],
                    "usage": usage or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                })
            except Exception as e:
                log(f"{trace_prefix} model-name passthrough collect error upstream={route_name}: {e}")
                save_request_log(
                    model,
                    data.get('messages', []),
                    f"[ERROR] {e}",
                    stream=False,
                    request_payload=outbound_data,
                    error_type="model_name_passthrough_upstream_error",
                    trace_id=trace_id,
                )
                return JSONResponse(
                    {"error": {"message": str(e), "type": "model_name_passthrough_upstream_error"}},
                    status_code=502,
                )

        # 未匹配任何显式路由且无透传上游：返回清晰错误。
        if stream:
            release_active_stream_caller(caller_key, trace_id)
        err_msg = (
            f"No upstream route for model {model!r}. Configure it in runtime-flags.json "
            "openai_compatible_upstreams or use a built-in Claude/Gemini/GPT route."
        )
        log(f"{trace_prefix} no_route model={model} since_enter={fmt_ms(route_t0)}")
        save_request_log(
            model,
            data.get('messages', []),
            f"[NO_ROUTE] {err_msg}",
            stream=bool(stream),
            request_payload=data,
            error_type="no_route",
            trace_id=trace_id,
        )
        return JSONResponse(
            {"error": {"message": err_msg, "type": "no_route"}},
            status_code=404,
        )

    except httpx.TimeoutException:
        log(f"{trace_prefix} Request timeout")
        return JSONResponse(
            {"error": {"message": "Upstream timeout", "type": "timeout"}},
            status_code=504
        )
    except Exception as e:
        log(f"{trace_prefix} Exception: {e}")
        if isinstance(data, dict):
            save_request_log(
                data.get('model', ''),
                data.get('messages', []),
                f"[EXCEPTION] {e}",
                stream=bool(data.get('stream', False)),
                request_payload=data,
                error_type="proxy_error",
                trace_id=trace_id,
            )
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": {"message": str(e), "type": "proxy_error"}},
            status_code=500
        )


@app.get('/v1/models')
async def models(request: Request):
    """返回本地模型 + 动态上游模型目录。"""
    directory = await refresh_model_directory(
        force=True,
        request_authorization=request.headers.get('authorization', ''),
    )
    models_data = directory.get('models')
    if not isinstance(models_data, list):
        models_data = []
    return JSONResponse({"object": "list", "data": models_data})


@app.get('/health')
async def health():
    return JSONResponse({"status": "ok"})


@app.get('/admin/claude-replay')
async def admin_claude_replay_state():
    return JSONResponse({
        'ok': True,
        'control': claude_replay.build_state(),
        'entries': claude_replay.list_entries(),
    })


@app.post('/admin/claude-replay')
async def admin_claude_replay_update(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {'ok': False, 'error': 'Request body must be valid JSON'},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {'ok': False, 'error': 'Request body must be a JSON object'},
            status_code=400,
        )

    enabled = _coerce_bool(body.get('enabled'), False)
    mode = str(body.get('mode', 'always') or 'always').strip().lower()
    match_request = _coerce_bool(body.get('match_request'), True)
    raw_sse_path = str(body.get('raw_sse_path', '') or '').strip()
    input_json_path = str(body.get('input_json_path', '') or '').strip()

    if mode not in claude_replay.allowed_modes:
        return JSONResponse(
            {
                'ok': False,
                'error': f"mode must be one of: {', '.join(sorted(claude_replay.allowed_modes))}",
            },
            status_code=400,
        )

    resolved_raw_sse_path = claude_replay.resolve_path(raw_sse_path)
    resolved_input_json_path = claude_replay.resolve_path(input_json_path) if input_json_path else None

    if enabled and not resolved_raw_sse_path:
        return JSONResponse(
            {'ok': False, 'error': 'Enabled replay requires a readable raw_sse_path'},
            status_code=400,
        )

    if enabled and match_request and not resolved_input_json_path:
        resolved_input_json_path = claude_replay.derive_input_json_path(raw_sse_path)
        if resolved_input_json_path and not input_json_path:
            input_json_path = os.path.basename(resolved_input_json_path)

    if enabled and match_request and not resolved_input_json_path:
        return JSONResponse(
            {'ok': False, 'error': 'match_request=true requires a readable input_json_path'},
            status_code=400,
        )

    control = {
        'enabled': enabled,
        'mode': mode,
        'match_request': match_request,
        'raw_sse_path': raw_sse_path,
    }
    if input_json_path:
        control['input_json_path'] = input_json_path

    try:
        claude_replay.write_control(control)
    except Exception as e:
        log(f"Claude replay control write failed: {e}")
        return JSONResponse(
            {'ok': False, 'error': f'Failed to write control file: {e}'},
            status_code=500,
        )

    return JSONResponse({
        'ok': True,
        'control': claude_replay.build_state(),
        'entries': claude_replay.list_entries(),
    })


@app.get('/admin/claude-replay/log-output')
async def admin_claude_replay_log_output(raw_sse_path: str):
    output_txt_path = claude_replay.derive_output_txt_path(raw_sse_path)
    if not output_txt_path:
        return JSONResponse(
            {'ok': False, 'error': 'output txt not found for selected raw_sse_path'},
            status_code=404,
        )

    try:
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        return JSONResponse(
            {'ok': False, 'error': f'failed to read output txt: {e}'},
            status_code=500,
        )

    try:
        size_bytes = os.stat(output_txt_path).st_size
    except Exception:
        size_bytes = len(text.encode('utf-8', errors='ignore'))

    return JSONResponse({
        'ok': True,
        'raw_sse_path': raw_sse_path,
        'output_txt_path': os.path.basename(output_txt_path),
        'size_bytes': size_bytes,
        'text': text,
    })


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', '3002'))
    uvicorn.run(app, host='0.0.0.0', port=port)
