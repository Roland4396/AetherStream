from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from aetherstream.api.dependencies import RouteDependencies, build_route_dependencies
from aetherstream.features.request_injections import append_assistant_prefill_continuation
from aetherstream.streaming.json_keepalive import keepalive_json_response
from aetherstream.streaming.responses import DisconnectSafeStreamingResponse as StreamingResponse


def _is_pioneer_auto_model(model: Any) -> bool:
    return str(model or '').strip().lower() in {'pioneer/auto', 'anthropic/pioneer-auto'}


def _is_pioneer_account_pool_base(base_url: Any) -> bool:
    base = str(base_url or '').lower()
    return 'account-pool' in base or 'pioneer.ai' in base


async def anthropic_messages(request: Request, deps: RouteDependencies):
    """Anthropic Messages API 透传。"""
    data = None
    trace_id = getattr(request.state, 'trace_id', None) or request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    try:
        data = await request.json()
        if deps.should_strip_claude_cache_controls():
            data = deps.strip_claude_cache_controls(data)
        model = data.get('model', '')
        stream = data.get('stream', False)
        removed_fields = deps.model_policy.apply_claude_sampling_compat(data)

        if _is_pioneer_auto_model(model):
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_model", "message": "pioneer/auto route is disabled; call the target model directly"}},
                status_code=400,
            )

        if not deps.model_policy.is_model_allowed(model):
            deps.save_request_log(
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
            deps.log(
                "Anthropic passthrough: "
                f"removed={','.join(sorted(removed_fields))} "
                "for Claude sampling compatibility"
            )
        deps.log(f"Anthropic passthrough: model={model}, stream={stream}")

        # NewAPI 已移除：Anthropic Messages 透传改为直连 Claude 上游。
        msg_provider, msg_real_model, _ = deps.parse_claude_provider_prefix(model)
        msg_api_key, msg_base_url = deps.get_claude_upstream_for_provider(msg_provider)
        if msg_provider:
            model = msg_real_model
            data['model'] = model
            deps.log(f"Anthropic passthrough: claude_provider={msg_provider} real_model={model}")
        if msg_provider != 'codecli' and _is_pioneer_account_pool_base(msg_base_url):
            data.pop('models ', None)
        append_assistant_prefill_continuation(
            data,
            selected_model=model,
            trace_prefix=f"[TRACE {trace_id}]",
            route_label="anthropic_messages",
            log=deps.log,
        )
        if not msg_api_key:
            missing_label = f"CLAUDE2_API_KEY (codecli)" if msg_provider == 'codecli' else "CLAUDE_API_KEY"
            return JSONResponse(
                {"type": "error", "error": {"type": "config_error", "message": f"{missing_label} is missing"}},
                status_code=500,
            )
        claude_session_id = deps.extract_claude_session_id(data.get('metadata'))
        forward_headers = deps.build_claude_upstream_headers(
            session_id=claude_session_id,
            model=model,
            api_key=msg_api_key,
        )
        target_url = deps.build_claude_messages_url(base_url=msg_base_url)

        if stream:
            # 流式：透传
            return StreamingResponse(
                deps.forward_anthropic_messages_stream(
                    url=target_url,
                    request_data=data,
                    headers=forward_headers,
                    timeout=deps.get_timeout_config(),
                    deps=deps.build_anthropic_messages_deps(),
                    enable_early_stop=False,
                    trace_id=trace_id,
                ),
                media_type='text/event-stream'
            )

        # 非流式：内部仍走上游流式，再聚合回标准 Anthropic JSON
        try:
            resp_data, raw_sse = await deps.collect_anthropic_messages_response(
                url=target_url,
                request_data=data,
                headers=forward_headers,
                model=model,
                timeout=deps.get_timeout_config(),
                max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                deps=deps.build_anthropic_messages_deps(),
                trace_id=trace_id,
            )
            deps.save_request_log(
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
            deps.save_request_log(
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
        deps.log(f"Anthropic passthrough error: {e}")
        if isinstance(data, dict):
            deps.save_request_log(
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


MESSAGES_DEPENDENCY_NAMES = (
    'MAX_RAW_SSE_BYTES',
    'build_anthropic_messages_deps',
    'build_claude_messages_url',
    'build_claude_upstream_headers',
    'collect_anthropic_messages_response',
    'extract_claude_session_id',
    'forward_anthropic_messages_stream',
    'get_claude_upstream_for_provider',
    'get_timeout_config',
    'log',
    'model_policy',
    'parse_claude_provider_prefix',
    'save_request_log',
    'should_strip_claude_cache_controls',
    'strip_claude_cache_controls',
)


def register_routes(app, ctx: dict[str, Any]) -> RouteDependencies:
    deps = build_route_dependencies(ctx, MESSAGES_DEPENDENCY_NAMES)
    nonstream_keepalive_interval = float(ctx.get('NONSTREAM_KEEPALIVE_INTERVAL', 10.0))

    async def anthropic_messages_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return await anthropic_messages(request, deps)

        if not isinstance(payload, dict) or payload.get('stream', False):
            return await anthropic_messages(request, deps)

        return keepalive_json_response(
            lambda: anthropic_messages(request, deps),
            interval=nonstream_keepalive_interval,
            trace_id=getattr(request.state, 'trace_id', ''),
            log=deps.log,
        )

    app.post('/v1/messages')(anthropic_messages_endpoint)
    return deps
