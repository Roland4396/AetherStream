from __future__ import annotations

import copy
import json
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from aetherstream.api.dependencies import RouteDependencies, build_route_dependencies
from aetherstream.api.timed_routes import register_routes as register_timed_routes
from aetherstream.features.kimi_sampling import apply_kimi_sampling_compat
from aetherstream.features.replay import ReplayPreparationError
from aetherstream.features.scene_filter import remove_scene_messages
from aetherstream.features.stage_warning import wrap_stage_messages
from aetherstream.features.request_injections import (
    append_assistant_prefill_continuation,
    apply_direct_opus_note,
    apply_forced_opus_note,
    apply_pioneer_opus_note,
    is_kimi_model,
)
from aetherstream.features.terminal_tool import (
    inject_openai_chat_terminal_tool,
    prepend_anthropic_terminal_prompt,
    terminal_tool_enabled_for_model,
)
from aetherstream.streaming.json_keepalive import keepalive_json_response
from aetherstream.streaming.responses import DisconnectSafeStreamingResponse as StreamingResponse
from aetherstream.utils.coerce import coerce_bool


LATEST_WINS_LOCAL_MODELS = frozenset({
    'glm-5.2-local',
    'deepseek-v4-flash-local',
})


def _is_pioneer_auto_model(model: Any) -> bool:
    return str(model or '').strip().lower() in {'pioneer/auto', 'anthropic/pioneer-auto'}


def _claude_nonstream_to_stream_enabled(deps: RouteDependencies) -> bool:
    return coerce_bool(deps._runtime_lookup('claude', 'stream', 'nonstream_to_stream'), True)


async def _run_claude_nonstream_once(
    *, deps: RouteDependencies, protocol: str, model: str, url: str,
    headers: dict, request_data: dict, trace_id: str, runner,
) -> dict:
    # Share Gemini's detached, cross-release exact-request barrier. Include
    # endpoint/auth identity so identical bodies on different routes never mix.
    # Managed session IDs remain part of the key: a new turn after refusal must
    # not hit a response cached under the retired session.
    key = deps.build_exact_request_key({
        'scope': f'claude-nonstream:{protocol}',
        'url': url,
        'headers': headers,
        'request': request_data,
    })
    result, shared = await deps.run_exact_nonstream_once(
        dedupe_key=key, trace_id=trace_id,
        upstream_label=f'claude:{protocol}:{model}', runner=runner,
    )
    if shared:
        deps.log(f'[TRACE {trace_id}] nonstream_dedupe_return_shared model={model} key={key[:16]}')
    return result


async def _plain_claude_when_conversion_disabled(
    *, deps: RouteDependencies, model: str, request_data: dict, url: str,
    headers: dict, trace_id: str,
) -> JSONResponse | None:
    # Called only from existing non-stream request branches. Include channel
    # aliases such as [m3]claude-* and 「anti5」claude-* without changing
    # model routing/classification. Channel delimiters are not ASCII-only.
    if not (deps.model_policy.is_claude_model(model)
            or re.search(r'(?:^|[^\w])claude-', str(model).lower())):
        return None
    enabled = _claude_nonstream_to_stream_enabled(deps)
    deps.log(
        f"[TRACE {trace_id}] claude_nonstream_to_stream enabled={str(enabled).lower()} "
        f"incoming_stream=false upstream_stream={str(enabled).lower()} model={model}"
    )
    if enabled:
        return None  # Preserve the original streaming collector verbatim.
    payload = dict(request_data, stream=False)

    async def collect():
        content, _, _, _, raw, response = await deps.collect_chat_completions_nonstream(
            url=url, request_data=payload, headers=headers, timeout=deps.get_timeout_config(),
            deps=deps.build_chat_completions_upstream_deps(), trace_id=trace_id,
        )
        if not isinstance(response, dict) or isinstance(response.get('error'), dict):
            raise RuntimeError(f'Claude upstream non-stream error: {raw[:4000]}')
        if not isinstance(response.get('choices'), list) or not response['choices']:
            raise RuntimeError('Claude upstream non-stream response has no choices')
        return {'content': content, 'raw': raw, 'response': response}

    collected = await _run_claude_nonstream_once(
        deps=deps, protocol='chat', model=model, url=url, headers=headers,
        request_data=payload, trace_id=trace_id, runner=collect,
    )
    content, raw, response = collected['content'], collected['raw'], collected['response']
    deps.save_request_log(
        model, request_data.get('messages', []), content, stream=False,
        raw_sse=raw, request_payload=payload, trace_id=trace_id,
    )
    return JSONResponse(response)


def _drop_configured_request_fields(payload: dict[str, Any], route: dict[str, Any]) -> list[str]:
    fields = route.get('drop_request_fields')
    if not isinstance(fields, list):
        return []

    removed: list[str] = []
    for raw_field in fields:
        field = str(raw_field or '').strip()
        if field and field in payload:
            payload.pop(field, None)
            removed.append(field)
    return removed


def _apply_configured_reasoning(payload: dict[str, Any], route: dict[str, Any]) -> list[str]:
    changed: list[str] = []

    effort = route.get('effort')
    if isinstance(effort, str) and effort:
        if payload.get('effort') != effort:
            payload['effort'] = effort
            changed.append('effort')

    reasoning_effort = route.get('reasoning_effort')
    if isinstance(reasoning_effort, str) and reasoning_effort:
        if payload.get('reasoning_effort') != reasoning_effort:
            payload['reasoning_effort'] = reasoning_effort
            changed.append('reasoning_effort')

    enabled = route.get('enable_thinking')
    kwargs = payload.get('chat_template_kwargs')
    if not isinstance(kwargs, dict):
        kwargs = {}
    kwargs = dict(kwargs)
    if isinstance(enabled, bool) and kwargs.get('enable_thinking') is not enabled:
        kwargs['enable_thinking'] = enabled
        changed.append('chat_template_kwargs.enable_thinking')

    thinking = route.get('thinking')
    if isinstance(thinking, bool) and kwargs.get('thinking') is not thinking:
        kwargs['thinking'] = thinking
        changed.append('chat_template_kwargs.thinking')

    if isinstance(enabled, bool) or isinstance(thinking, bool):
        payload['chat_template_kwargs'] = kwargs
    return changed


def _upstream_http_status(error: BaseException, default: int = 502) -> int:
    """Recover an upstream HTTP status embedded by the OpenAI collector."""
    match = re.search(r"\bUpstream error:\s*(\d{3})\b", str(error))
    if match:
        status = int(match.group(1))
        if 400 <= status <= 599:
            return status
    return default


def _apply_configured_template_thinking(payload: dict[str, Any], route: dict[str, Any]) -> bool:
    """Compatibility wrapper for callers that only configure enable_thinking."""
    return bool(_apply_configured_reasoning(payload, route))


async def chat_completions(request: Request, deps: RouteDependencies):
    data = None
    inbound_request = None
    inbound_structure = None
    trace_id = getattr(request.state, 'trace_id', None) or request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    trace_prefix = f"[TRACE {trace_id}]"
    route_t0 = time.perf_counter()
    caller_key, caller_desc = deps.build_caller_fingerprint(request)
    local_supersede_event = None
    try:
        json_t0 = time.perf_counter()
        data = await request.json()
        inbound_request = copy.deepcopy(data)
        inbound_structure = deps.summarize_openai_messages(inbound_request.get('messages', []))
        deps.log(f"{trace_prefix} json_parsed elapsed={deps.fmt_ms(json_t0)} since_enter={deps.fmt_ms(route_t0)}")
        deps.log(
            f"{trace_prefix} inbound_openai_structure "
            f"{json.dumps(inbound_structure[:12], ensure_ascii=False)}"
        )
        model = data.get('model', '')
        stream = data.get('stream', False)
        terminal_tool_enabled = terminal_tool_enabled_for_model(model)
        injected = inject_openai_chat_terminal_tool(data) if terminal_tool_enabled else False
        deps.log(
            f"{trace_prefix} terminal_tool_injection protocol=openai_chat "
            f"action={'added' if injected else ('present' if terminal_tool_enabled else 'skipped_model')}"
        )
        removed_fields = deps.model_policy.apply_claude_sampling_compat(data)
        msg_count = len(data.get('messages', []))
        payload_sig = (
            f"model={model} stream={stream} msgs={msg_count} "
            f"cl={request.headers.get('content-length', '-')}"
        )
        if removed_fields:
            deps.log(
                f"{trace_prefix} claude_sampling_compat "
                f"removed={','.join(sorted(removed_fields))}"
            )
        deps.log(f"{trace_prefix} caller={caller_key} {caller_desc}")
        deps.log(f"{trace_prefix} request_meta {payload_sig}")

        if _is_pioneer_auto_model(model):
            if stream:
                deps.release_active_stream_caller(caller_key, trace_id)
            return JSONResponse(
                {"error": {"message": "pioneer/auto route is disabled; call the target model directly", "type": "invalid_model"}},
                status_code=400,
            )

        if stream:
            prev = deps.active_stream_registry.get(caller_key)
            if prev and prev.get('trace_id') != trace_id:
                prev_age = time.time() - prev.get('started_at', time.time())
                deps.log(
                    f"{trace_prefix} caller_overlap caller={caller_key} "
                    f"prev_trace={prev.get('trace_id')} prev_age={prev_age:.1f}s "
                    f"prev_model={prev.get('model')} prev_msgs={prev.get('msg_count')} "
                    f"note=new stream from same caller may cancel previous stream"
                )
            supersede_previous = bool(
                model in LATEST_WINS_LOCAL_MODELS
            )
            deps.active_stream_registry.register(
                caller_key,
                trace_id=trace_id,
                model=model,
                msg_count=msg_count,
                supersede_previous=supersede_previous,
            )
            if model in LATEST_WINS_LOCAL_MODELS:
                local_supersede_event = deps.active_stream_registry.cancellation_event(
                    caller_key,
                    trace_id,
                )

        replay_model = str(inbound_request.get('model') or model)
        replay_messages = inbound_request.get('messages')
        if not isinstance(replay_messages, list):
            replay_messages = []
        try:
            prepared_replay = deps.replay_service.prepare(
                model=replay_model,
                messages=replay_messages,
            )
        except ReplayPreparationError as replay_error:
            if stream:
                deps.release_active_stream_caller(caller_key, trace_id)
            deps.log(f"{trace_prefix} replay_prepare_error err={replay_error}")
            return JSONResponse(
                {
                    "error": {
                        "message": str(replay_error),
                        "type": "replay_invalid",
                    }
                },
                status_code=422,
            )

        if prepared_replay is not None:
            replay_record = prepared_replay.record
            deps.log(
                f"{trace_prefix} replay_intercept "
                f"format={replay_record.source_format} "
                f"source_complete={str(replay_record.source_complete).lower()} "
                f"file={prepared_replay.spec.get('raw_sse_path')} "
                f"since_enter={deps.fmt_ms(route_t0)}"
            )
            if stream:
                return StreamingResponse(
                    deps.replay_service.stream(
                        prepared_replay,
                        model=replay_model,
                        messages=replay_messages,
                        request_payload=inbound_request,
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                    ),
                    media_type='text/event-stream',
                )
            return JSONResponse(
                deps.replay_service.complete(
                    prepared_replay,
                    model=replay_model,
                    messages=replay_messages,
                    request_payload=inbound_request,
                    trace_id=trace_id,
                )
            )

        scene_removed, scene_invalid = remove_scene_messages(data.get('messages'))
        if scene_removed or scene_invalid:
            deps.log(
                f"{trace_prefix} ai_scene_filter removed={scene_removed} "
                f"invalid_skipped={scene_invalid}"
            )

        stage_wrapped, stage_invalid = wrap_stage_messages(data.get('messages'))
        if stage_wrapped or stage_invalid:
            deps.log(
                f"{trace_prefix} ai_stage_warning wrapped={stage_wrapped} "
                f"invalid_skipped={stage_invalid}"
            )

        if str(model).startswith("fake-slow-stream"):
            fake_chunks = int(data.get('fake_chunks') or 1200)
            fake_delay = float(data.get('fake_delay') or 0.5)
            fake_token = str(data.get('fake_token') or '假流')
            deps.log(
                f"{trace_prefix} Route to local fake slow stream "
                f"model={model} chunks={fake_chunks} delay={fake_delay} since_enter={deps.fmt_ms(route_t0)}"
            )
            if stream:
                return StreamingResponse(
                    deps.fake_slow_openai_stream(
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
                {"error": {"message": f"Model disabled by proxy: {model}", "type": "invalid_model"}},
                status_code=400
            )

        if deps.is_claude_opus55_model(model):
            # Apply before provider selection so aliases and all public
            # protocols get the same approved note. Shared markers dedupe
            # any existing route-specific Opus injection later in the path.
            apply_forced_opus_note(
                data,
                selected_model=model,
                trace_prefix=trace_prefix,
                route_label="opus55",
                log=deps.log,
            )

        if is_kimi_model(model):
            data = copy.deepcopy(data)
            removed = apply_kimi_sampling_compat(data, model)
            deps.log(
                f"{trace_prefix} kimi_compat model={model} "
                f"sampling_removed={','.join(removed) or '-'} early_stop=runtime_config"
            )
            apply_forced_opus_note(
                data,
                selected_model=model,
                trace_prefix=trace_prefix,
                route_label="kimi",
                log=deps.log,
            )

        is_free_model, free_real_model = deps.parse_free_provider_prefix(model)
        # Free-pool Claude models require the Anthropic Messages API. Leave
        # those for the Claude shim below; this branch is OpenAI-compatible.
        if is_free_model and not deps.model_policy.is_claude_model(model):
            free_upstream_key, free_upstream_base = deps.get_claude_upstream_for_provider('free')
            if not free_upstream_key:
                if stream:
                    deps.release_active_stream_caller(caller_key, trace_id)
                return JSONResponse(
                    {"error": {"message": "CLAUDE_API_KEY/free account pool key is missing", "type": "config_error"}},
                    status_code=500
                )

            target_url = deps.build_free_openai_chat_url(free_upstream_base)
            outbound_data = copy.deepcopy(data)
            outbound_data['model'] = free_real_model
            outbound_data.pop('models ', None)
            apply_pioneer_opus_note(
                outbound_data,
                selected_model=free_real_model,
                trace_prefix=trace_prefix,
                route_label="free_openai",
                runtime_lookup=deps._runtime_lookup,
                is_opus_model=deps.is_claude_opus_model,
                log=deps.log,
            )
            if deps.apply_openai_template_thinking_disabled(outbound_data, free_real_model):
                deps.log(
                    f"{trace_prefix} free_openai_disable_template_thinking "
                    f"model={model} real_model={free_real_model}"
                )
            if deps.is_claude_opus55_model(free_real_model):
                outbound_data, compat_meta = deps.apply_claude_model_compat_request(outbound_data)
                deps.log(
                    f"{trace_prefix} opus55_adaptive_compat route=free_openai "
                    f"model={model} removed={compat_meta.get('removed', '-')}"
                )
            append_assistant_prefill_continuation(
                outbound_data,
                selected_model=free_real_model,
                trace_prefix=trace_prefix,
                route_label="free_openai",
                log=deps.log,
            )
            headers = deps.build_free_openai_headers(free_upstream_key)
            free_is_gemini = deps.model_policy.is_gemini_model(free_real_model)

            if free_is_gemini:
                filter_stats = deps.apply_drawing_context_filter(outbound_data)
                if filter_stats.get('blocks'):
                    deps.log(
                        f"{trace_prefix} free_openai_gemini_drawing_context_filter "
                        f"model={model} real_model={free_real_model} "
                        f"messages={filter_stats['messages']} blocks={filter_stats['blocks']} "
                        f"chars={filter_stats['chars']} mode=closed_tags_and_bare_image_prompts"
                    )

            deps.log(
                f"{trace_prefix} Route to free OpenAI account-pool upstream "
                f"model={model} real_model={free_real_model} url={target_url} "
                f"stream_to_nonstream={str(bool(stream and free_is_gemini)).lower()} "
                f"since_enter={deps.fmt_ms(route_t0)}"
            )

            if stream:
                if free_is_gemini:
                    outbound_data['stream'] = False
                    return StreamingResponse(
                        deps.replay_chat_completions_nonstream_as_stream(
                            url=target_url,
                            request_data=outbound_data,
                            headers=headers,
                            timeout=deps.get_timeout_config(),
                            deps=deps.build_chat_completions_upstream_deps(),
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                        ),
                        media_type='text/event-stream'
                    )

                outbound_data['stream'] = True
                return StreamingResponse(
                    deps.forward_chat_completions_stream(
                        url=target_url,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=deps.build_chat_completions_upstream_deps(),
                        enable_early_stop=is_kimi_model(model),
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                    ),
                    media_type='text/event-stream'
                )

            try:
                plain_response = await _plain_claude_when_conversion_disabled(
                    deps=deps, model=free_real_model, request_data=outbound_data,
                    url=target_url, headers=headers, trace_id=trace_id,
                )
                if plain_response is not None:
                    return plain_response
                if free_is_gemini:
                    outbound_data['stream'] = False
                    dedupe_key = deps.build_exact_request_key(outbound_data)

                    async def _run_free_openai_nonstream():
                        collected = await deps.collect_chat_completions_nonstream(
                            url=target_url,
                            request_data=outbound_data,
                            headers=headers,
                            timeout=deps.get_timeout_config(),
                            deps=deps.build_chat_completions_upstream_deps(),
                            trace_id=trace_id,
                        )
                        return collected[5]

                    upstream_nonstream_payload, dedupe_shared = await deps.run_exact_nonstream_once(
                        dedupe_key=dedupe_key,
                        trace_id=trace_id,
                        upstream_label=f"free-openai:{free_real_model}",
                        runner=_run_free_openai_nonstream,
                    )
                    full_content = deps.extract_openai_chat_payload_content(upstream_nonstream_payload)
                    raw_response = json.dumps(upstream_nonstream_payload, ensure_ascii=False)
                    deps.save_request_log(
                        model,
                        data.get('messages', []),
                        full_content,
                        stream=False,
                        raw_sse=raw_response,
                        request_payload=outbound_data,
                        trace_id=trace_id,
                    )
                    if dedupe_shared:
                        deps.log(
                            f"{trace_prefix} nonstream_dedupe_return_shared "
                            f"key={dedupe_key[:16]} out_chars={len(full_content)}"
                        )
                    return JSONResponse(upstream_nonstream_payload)

                outbound_data['stream'] = True
                full_content, model_name, usage, finish_reason, raw_sse = await deps.collect_chat_completions_stream(
                    url=target_url,
                    request_data=outbound_data,
                    headers=headers,
                    timeout=deps.get_timeout_config(),
                    max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                    deps=deps.build_chat_completions_upstream_deps(),
                    enable_early_stop=is_kimi_model(model),
                    trace_id=trace_id,
                )
                deps.save_request_log(
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
                    "model": model_name or free_real_model,
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
                deps.log(f"{trace_prefix} free_openai collect error model={model} real_model={free_real_model}: {e}")
                deps.save_request_log(
                    model,
                    data.get('messages', []),
                    f"[ERROR] {e}",
                    stream=False,
                    request_payload=outbound_data,
                    error_type="free_openai_upstream_error",
                    trace_id=trace_id,
                )
                return JSONResponse(
                    {"error": {"message": str(e), "type": "free_openai_upstream_error"}},
                    status_code=502,
                )

        directory_route = await deps.resolve_openai_compatible_route(
            model,
            request_authorization=request.headers.get('authorization', ''),
        )
        if directory_route:
            route_name = directory_route.get('name') or 'openai-compatible'
            base_url = str(directory_route.get('base_url') or '').rstrip('/')
            target_url = f"{base_url}/chat/completions"
            outbound_data = copy.deepcopy(data)
            dropped_fields = _drop_configured_request_fields(outbound_data, directory_route)
            if dropped_fields:
                deps.log(
                    f"{trace_prefix} model-directory_request_compat "
                    f"name={route_name} model={model} "
                    f"removed={','.join(dropped_fields)}"
                )
            directory_is_pioneer_upstream = bool(directory_route.get('pioneer_upstream') or directory_route.get('pioneer_router'))
            if directory_is_pioneer_upstream:
                outbound_data.pop('models ', None)
                apply_pioneer_opus_note(
                    outbound_data,
                    selected_model=model,
                    trace_prefix=trace_prefix,
                    route_label="model_directory",
                    runtime_lookup=deps._runtime_lookup,
                    is_opus_model=deps.is_claude_opus_model,
                    log=deps.log,
                )
            has_reasoning_override = (
                bool(str(directory_route.get('effort') or '').strip())
                or bool(str(directory_route.get('reasoning_effort') or '').strip())
                or isinstance(directory_route.get('enable_thinking'), bool)
                or isinstance(directory_route.get('thinking'), bool)
            )
            if has_reasoning_override:
                changed = _apply_configured_reasoning(outbound_data, directory_route)
                deps.log(
                    f"{trace_prefix} model-directory_reasoning_override "
                    f"name={route_name} model={model} "
                    f"changed={','.join(changed) or '-'}"
                )
            elif deps.apply_openai_template_thinking_disabled(outbound_data, model):
                deps.log(
                    f"{trace_prefix} model-directory_disable_template_thinking "
                    f"name={route_name} model={model}"
                )
            route_forces_opus_note = bool(directory_route.get('inject_opus_note'))
            if route_forces_opus_note:
                apply_forced_opus_note(
                    outbound_data,
                    selected_model=model,
                    trace_prefix=trace_prefix,
                    route_label='model_directory_configured_opus',
                    log=deps.log,
                    log_context=f'name={route_name}',
                )
            if not route_forces_opus_note and deps.should_append_pro_opus46_last_user_note(route_name, base_url, model):
                apply_forced_opus_note(
                    outbound_data,
                    selected_model=model,
                    trace_prefix=trace_prefix,
                    route_label='pro_opus46',
                    log=deps.log,
                    log_context=f'name={route_name}',
                )
            if deps.should_apply_deepseek_drawing_context_filter(route_name, base_url):
                filter_stats = deps.apply_drawing_context_filter(outbound_data)
                if filter_stats.get('blocks'):
                    deps.log(
                        f"{trace_prefix} ds_drawing_context_filter "
                        f"name={route_name} messages={filter_stats['messages']} "
                        f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                        f"mode=closed_tags_and_bare_image_prompts"
                    )
            if deps.model_policy.is_gemini_model(model):
                filter_stats = deps.apply_drawing_context_filter(outbound_data)
                if filter_stats.get('blocks'):
                    deps.log(
                        f"{trace_prefix} pro_gemini_drawing_context_filter "
                        f"name={route_name} model={model} messages={filter_stats['messages']} "
                        f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                        f"mode=closed_tags_and_bare_image_prompts"
                    )
            # Opus 5.5 rejects disabled thinking. Do not let the general pro
            # latency policy erase its adaptive/effort settings.
            opus55 = deps.is_claude_opus55_model(model)
            no_reasoning_meta = {} if opus55 else deps.apply_pro_no_reasoning_payload(outbound_data, route_name, base_url)
            if no_reasoning_meta:
                deps.log(
                    f"{trace_prefix} pro_no_reasoning_payload "
                    f"name={route_name} model={model} "
                    f"reasoning_effort={no_reasoning_meta.get('reasoning_effort')} "
                    f"removed={','.join(no_reasoning_meta.get('removed') or []) or '-'}"
                )
            if opus55:
                outbound_data, compat_meta = deps.apply_claude_model_compat_request(outbound_data)
                deps.log(
                    f"{trace_prefix} opus55_adaptive_compat route=model_directory "
                    f"name={route_name} model={model} removed={compat_meta.get('removed', '-')}"
                )
            # Some Claude-compatible model ids are provider aliases (for
            # example ``agy-claude-opus-4-6``) and cannot be classified from
            # the model string alone.  An explicit route compatibility flag
            # is authoritative for those aliases.
            route_requires_user_ended_prefill = (
                directory_is_pioneer_upstream
                or deps.model_policy.is_claude_model(model)
                or bool(directory_route.get('inject_opus_note'))
            )
            if route_requires_user_ended_prefill:
                append_assistant_prefill_continuation(
                    outbound_data,
                    selected_model=model,
                    trace_prefix=trace_prefix,
                    route_label="model_directory",
                    log=deps.log,
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
            deps.log(
                f"{trace_prefix} Route to model-directory upstream "
                f"name={route_name} model={model} url={target_url} "
                f"pioneer_upstream={str(directory_is_pioneer_upstream).lower()} "
                f"since_enter={deps.fmt_ms(route_t0)}"
            )

            if stream:
                pro_gemini_nonstream_replay = bool(deps.model_policy.is_gemini_model(model))
                outbound_data['stream'] = False if pro_gemini_nonstream_replay else True
                if pro_gemini_nonstream_replay:
                    deps.log(
                        f"{trace_prefix} model-directory pro_gemini stream_to_nonstream_replay "
                        f"name={route_name} model={model}"
                    )
                    return StreamingResponse(
                        deps.replay_chat_completions_nonstream_as_stream(
                            url=target_url,
                            request_data=outbound_data,
                            headers=headers,
                            timeout=deps.get_timeout_config(),
                            deps=deps.build_chat_completions_upstream_deps(),
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                        ),
                        media_type='text/event-stream'
                    )

                return StreamingResponse(
                    deps.forward_chat_completions_stream(
                        url=target_url,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=deps.build_chat_completions_upstream_deps(),
                        enable_early_stop=is_kimi_model(model),
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                        supersede_event=local_supersede_event,
                    ),
                    media_type='text/event-stream'
                )

            try:
                plain_response = await _plain_claude_when_conversion_disabled(
                    deps=deps, model=model, request_data=outbound_data,
                    url=target_url, headers=headers, trace_id=trace_id,
                )
                if plain_response is not None:
                    return plain_response
                use_plain_non_stream = bool(deps.model_policy.is_gemini_model(model))
                upstream_nonstream_payload = None
                if use_plain_non_stream:
                    outbound_data['stream'] = False
                    dedupe_key = deps.build_exact_request_key(outbound_data)
                    deps.log(
                        f"{trace_prefix} model-directory Gemini nonstream passthrough "
                        f"name={route_name} model={model} key={dedupe_key[:16]}"
                    )

                    async def _run_plain_nonstream():
                        collected = await deps.collect_chat_completions_nonstream(
                            url=target_url,
                            request_data=outbound_data,
                            headers=headers,
                            timeout=deps.get_timeout_config(),
                            deps=deps.build_chat_completions_upstream_deps(),
                            trace_id=trace_id,
                        )
                        return collected[5]

                    upstream_nonstream_payload, dedupe_shared = await deps.run_exact_nonstream_once(
                        dedupe_key=dedupe_key,
                        trace_id=trace_id,
                        upstream_label=f"{route_name}:{model}",
                        runner=_run_plain_nonstream,
                    )
                    full_content = deps.extract_openai_chat_payload_content(upstream_nonstream_payload)
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
                        deps.log(
                            f"{trace_prefix} nonstream_dedupe_return_shared "
                            f"key={dedupe_key[:16]} out_chars={len(full_content)}"
                        )
                else:
                    # The collector sends stream=True on its own copy. Keep
                    # the saved upstream payload truthful as well.
                    outbound_data['stream'] = True
                    full_content, model_name, usage, finish_reason, raw_response = await deps.collect_chat_completions_stream(
                        url=target_url,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=deps.build_chat_completions_upstream_deps(),
                        enable_early_stop=is_kimi_model(model),
                        trace_id=trace_id,
                    )
                deps.save_request_log(
                    model,
                    data.get('messages', []),
                    full_content,
                    stream=False,
                    raw_sse="" if use_plain_non_stream else raw_response,
                    request_payload=outbound_data,
                    trace_id=trace_id,
                )
                if use_plain_non_stream and isinstance(upstream_nonstream_payload, dict):
                    deps.log(
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
                deps.log(f"{trace_prefix} model-directory collect error upstream={route_name}: {e}")
                deps.save_request_log(
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
                    status_code=_upstream_http_status(e),
                )

        # Gemini 模型：直接 HTTP 直连（不再依赖 gemini-proxy）
        if deps.model_policy.is_gemini_model(model):
            if not deps.GEMINI_ENABLED:
                return JSONResponse(
                    {"error": {"message": "Gemini is temporarily disabled by proxy", "type": "service_unavailable"}},
                    status_code=503
                )

            deps.log(f"{trace_prefix} Route to Gemini HTTP: {model} since_enter={deps.fmt_ms(route_t0)}")
            if not deps.GEMINI_API_KEY:
                return JSONResponse(
                    {"error": {"message": "GEMINI_API_KEY is missing", "type": "config_error"}},
                    status_code=500
                )
            data = copy.deepcopy(data)
            filter_stats = deps.apply_drawing_context_filter(data)
            if filter_stats.get('blocks'):
                deps.log(
                    f"{trace_prefix} gemini_drawing_context_filter "
                    f"model={model} messages={filter_stats['messages']} "
                    f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                    f"mode=closed_tags_and_bare_image_prompts"
                )
            gemini_config = deps.build_gemini_generate_content_config()
            gemini_deps = deps.build_gemini_generate_content_deps()

            deps.log(
                f"{trace_prefix} native_gemini stream_to_nonstream "
                f"model={model} downstream_stream={str(bool(stream)).lower()} "
                f"upstream_method=generateContent upstream_stream=false"
            )
            if stream:
                data['stream'] = False
                return StreamingResponse(
                    deps.forward_gemini_generate_content_stream(
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
                dedupe_key = deps.build_exact_request_key(data)

                async def _run_gemini_nonstream():
                    full_content_inner, usage_inner, finish_reason_inner = await deps.collect_gemini_generate_content(
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

                response_payload, dedupe_shared = await deps.run_exact_nonstream_once(
                    dedupe_key=dedupe_key,
                    trace_id=trace_id,
                    upstream_label=f"gemini-http:{model}",
                    runner=_run_gemini_nonstream,
                )
                full_content = deps.extract_openai_chat_payload_content(response_payload)
                deps.save_request_log(
                    model,
                    data.get('messages', []),
                    full_content,
                    stream=False,
                    request_payload=data,
                    trace_id=trace_id,
                )
                if dedupe_shared:
                    deps.log(
                        f"{trace_prefix} nonstream_dedupe_return_shared "
                        f"key={dedupe_key[:16]} out_chars={len(full_content)}"
                    )
                return JSONResponse(response_payload)
            except Exception as e:
                deps.log(f"Gemini collect error: {e}")
                deps.schedule_delayed_restart()
                deps.save_request_log(
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

        if deps.model_policy.is_claude_model(model):
            claude_provider, claude_real_model, claude_display = deps.parse_claude_provider_prefix(model)
            claude_upstream_key, claude_upstream_base = deps.get_claude_upstream_for_provider(claude_provider)
            if not claude_upstream_key:
                if stream:
                    deps.release_active_stream_caller(caller_key, trace_id)
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
                deps.log(f"{trace_prefix} claude_provider={claude_provider} real_model={model}")

            if deps.is_claude_haiku_model(model):
                data = copy.deepcopy(data)
                filter_stats = deps.apply_drawing_context_filter(data)
                if filter_stats.get('blocks'):
                    deps.log(
                        f"{trace_prefix} haiku_drawing_context_filter "
                        f"model={model} messages={filter_stats['messages']} "
                        f"blocks={filter_stats['blocks']} chars={filter_stats['chars']} "
                        f"mode=closed_tags_and_bare_image_prompts"
                    )

            if deps.should_strip_claude_cache_controls():
                data = deps.strip_claude_cache_controls(data)

            data = copy.deepcopy(data)
            apply_direct_opus_note(
                data,
                selected_model=model,
                trace_prefix=trace_prefix,
                route_label="claude_code",
                is_opus_model=deps.is_claude_opus_model,
                log=deps.log,
            )
            append_assistant_prefill_continuation(
                data,
                selected_model=model,
                trace_prefix=trace_prefix,
                route_label="claude_code",
                log=deps.log,
            )

            claude_metadata_user_id, claude_session_id, claude_session_mode, claude_session_ttl_sec, claude_session_key = deps.build_timed_claude_user_id(
                model=model,
                provider=claude_provider or 'default',
            )
            prompt_cache_cfg = deps.get_claude_prompt_caching_settings()
            prompt_cache_enabled_for_model = bool(
                prompt_cache_cfg.get('enabled') and deps.is_claude_prompt_cache_model(model)
            )
            prompt_cache_keepalive_cfg = deps.get_claude_cache_keepalive_settings()
            prompt_cache_keepalive_enabled_for_model = bool(
                prompt_cache_enabled_for_model and prompt_cache_keepalive_cfg.get('enabled')
            )
            prompt_cache_control = deps.build_claude_prompt_cache_control() if prompt_cache_enabled_for_model else None
            prompt_cache_strategy = (
                'sonnet_fill_table' if deps.is_claude_sonnet_model(model)
                else 'opus_roleplay_layered' if deps.is_claude_opus_model(model)
                else 'generic'
            )
            claude_request = deps.convert_chat_to_anthropic_messages_request(
                data,
                system_prefix=deps.build_claude_system_prefix(model),
                metadata_user_id=claude_metadata_user_id,
                default_max_tokens=deps.CLAUDE_DEFAULT_MAX_TOKENS,
                prompt_cache_control=prompt_cache_control if prompt_cache_enabled_for_model and prompt_cache_cfg.get('mode') == 'explicit' else None,
                prompt_cache_strategy=prompt_cache_strategy,
            )
            if terminal_tool_enabled:
                prepend_anthropic_terminal_prompt(claude_request)
            if prompt_cache_enabled_for_model and prompt_cache_cfg.get('mode') == 'automatic' and isinstance(prompt_cache_control, dict):
                claude_request['cache_control'] = dict(prompt_cache_control)
            claude_request = deps.apply_claude_output_settings(claude_request)
            claude_request, keyword_filter_stats = deps.apply_claude_keyword_filter(claude_request)
            claude_request, compat_meta = deps.apply_claude_client_compat_request(claude_request)
            claude_request, model_compat_meta = deps.apply_claude_model_compat_request(claude_request)
            for key, value in model_compat_meta.items():
                compat_meta[f'model_{key}'] = value
            if deps.should_strip_claude_cache_controls() and deps.is_claude_haiku_model(model):
                claude_request = deps.strip_claude_cache_controls(claude_request)
                compat_meta['cache_control'] = 'stripped'
            elif deps.should_strip_claude_cache_controls():
                compat_meta['cache_control'] = 'preserved_agent_system'
            claude_structure = deps.summarize_anthropic_request(claude_request)
            claude_cache_breakpoint_layout = deps.summarize_anthropic_cache_breakpoints(claude_request)
            claude_headers = deps.build_claude_upstream_headers(
                session_id=claude_session_id,
                model=claude_request.get('model'),
                api_key=claude_upstream_key,
            )
            claude_url = deps.build_claude_messages_url(base_url=claude_upstream_base)

            deps.log(
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
                deps.log(
                    f"{trace_prefix}Claude cache breakpoint layout: "
                    f"{json.dumps(claude_cache_breakpoint_layout, ensure_ascii=False, separators=(',', ':'))}"
                )
            if int(keyword_filter_stats.get('total_removed', 0) or 0) > 0:
                deps.log(
                    f"{trace_prefix} Claude keyword filter "
                    f"removed={json.dumps(keyword_filter_stats.get('removed_keywords', {}), ensure_ascii=False)} "
                    f"touched_paths={json.dumps(keyword_filter_stats.get('touched_paths', []), ensure_ascii=False)}"
                )
            deps.log(f"{trace_prefix} Route to Claude upstream /v1/messages: {model} since_enter={deps.fmt_ms(route_t0)}")

            if stream:
                claude_request = dict(claude_request)
                claude_request['stream'] = True
                if deps.is_claude_haiku_model(model):
                    claude_request, system_fold_meta = deps.fold_claude_system_into_first_user_message(claude_request)
                    if system_fold_meta.get('action') != 'absent':
                        deps.log(
                            f"{trace_prefix} Claude stream system fold "
                            f"action={system_fold_meta.get('action')} "
                            f"blocks={system_fold_meta.get('blocks')} "
                            f"chars={system_fold_meta.get('chars')}"
                        )
                claude_deps = deps.build_anthropic_messages_deps(claude_request)
                claude_deps.session_key = claude_session_key
                claude_deps.session_id = claude_session_id
                return StreamingResponse(
                    deps.forward_anthropic_messages_as_chat_stream(
                        url=claude_url,
                        request_data=claude_request,
                        headers=claude_headers,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=claude_deps,
                        cache_keepalive=prompt_cache_keepalive_cfg if prompt_cache_keepalive_enabled_for_model else None,
                    ),
                    media_type='text/event-stream'
                )

            try:
                claude_request = dict(claude_request)
                claude_request['stream'] = _claude_nonstream_to_stream_enabled(deps)
                deps.log(
                    f"{trace_prefix} claude_nonstream_to_stream "
                    f"enabled={str(claude_request['stream']).lower()} incoming_stream=false "
                    f"upstream_stream={str(claude_request['stream']).lower()} model={model}"
                )
                if deps.is_claude_haiku_model(model):
                    claude_request, system_fold_meta = deps.fold_claude_system_into_first_user_message(claude_request)
                    if system_fold_meta.get('action') != 'absent':
                        deps.log(
                            f"{trace_prefix} Claude collect system fold "
                            f"action={system_fold_meta.get('action')} "
                            f"blocks={system_fold_meta.get('blocks')} "
                            f"chars={system_fold_meta.get('chars')}"
                        )
                claude_deps = deps.build_anthropic_messages_deps(claude_request)
                claude_deps.session_key = claude_session_key
                claude_deps.session_id = claude_session_id

                async def collect_claude():
                    collected = await deps.collect_anthropic_messages_as_chat_completion(
                        url=claude_url,
                        request_data=claude_request,
                        headers=claude_headers,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=claude_deps,
                        upstream_stream=claude_request['stream'],
                    )
                    return {'collected': collected}

                collected_result = await _run_claude_nonstream_once(
                    deps=deps, protocol='messages', model=model, url=claude_url,
                    headers=claude_headers, request_data=claude_request,
                    trace_id=trace_id, runner=collect_claude,
                )
                full_content, model_name, usage, finish_reason, raw_response, tool_calls = collected_result['collected']
                deps.save_request_log(
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
                response_message = {
                    "role": "assistant",
                    "content": full_content,
                }
                if tool_calls:
                    response_message["tool_calls"] = tool_calls
                return JSONResponse({
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name or model,
                    "choices": [{
                        "index": 0,
                        "message": response_message,
                        "finish_reason": finish_reason
                    }],
                    "usage": usage or {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })
            except Exception as e:
                deps.log(f"{trace_prefix} Claude collect error: {e}")
                deps.save_request_log(
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

        # GPT：将 Chat Completions 入站转换为 OpenAI Responses 上游。
        outbound_data = data

        if deps.model_policy.is_gpt_model(model):
            outbound_data = deps.inject_gpt_usage_policies_system_message(data)
            deps.log(f"{trace_prefix} gpt_usage_policies_system_injected first_system=yes")
            # Apply the existing shared notes before Chat -> Responses conversion,
            # so both public entrypoints and both response modes retain them.
            apply_forced_opus_note(
                outbound_data,
                selected_model=model,
                trace_prefix=trace_prefix,
                route_label='gpt',
                log=deps.log,
            )
            if not deps.RESPONSES_API_KEY:
                if stream:
                    deps.release_active_stream_caller(caller_key, trace_id)
                return JSONResponse(
                    {"error": {"message": "RESPONSES_API_KEY is missing", "type": "config_error"}},
                    status_code=500
                )

            if deps.GPT_USE_RESPONSES:
                outbound_data = deps.convert_chat_to_responses_request(outbound_data)
                outbound_data = dict(outbound_data)
                outbound_data['stream'] = True
                outbound_data.setdefault('store', False)
                if deps.GPT_SERVICE_TIER:
                    outbound_data['service_tier'] = deps.GPT_SERVICE_TIER
                outbound_data.setdefault('prompt_cache_key', deps.build_gpt_prompt_cache_key(model, outbound_data, deps.GPT_PROMPT_CACHE_KEY))
                if deps.GPT_PROMPT_CACHE_RETENTION:
                    outbound_data.setdefault('prompt_cache_retention', deps.GPT_PROMPT_CACHE_RETENTION)
                reasoning = outbound_data.get('reasoning')
                outbound_data['reasoning'] = dict(reasoning) if isinstance(reasoning, dict) else {}
                configured_effort = str(deps.RESPONSES_REASONING_EFFORT or '').strip().lower()
                outbound_data['reasoning']['effort'] = 'max' if configured_effort == 'ultra' else configured_effort
                if configured_effort == 'ultra':
                    outbound_data['reasoning']['context'] = 'all_turns'
                if deps.RESPONSES_REASONING_SUMMARY:
                    outbound_data['reasoning'].setdefault('summary', deps.RESPONSES_REASONING_SUMMARY)

                responses_deps = deps.build_responses_upstream_deps()
                deps.log(
                    f"{trace_prefix} Route to GPT Responses upstream /v1/responses: "
                    f"{model} service_tier={outbound_data.get('service_tier', '-')} "
                    f"prompt_cache_key={outbound_data.get('prompt_cache_key', '-')} "
                    f"prompt_cache_retention={outbound_data.get('prompt_cache_retention', '-')} "
                    f"reasoning_effort={configured_effort or '-'} "
                    f"reasoning_wire_effort={(outbound_data.get('reasoning') or {}).get('effort', '-')} "
                    f"input_items={len(outbound_data.get('input', []))} "
                    f"instructions={'yes' if outbound_data.get('instructions') else 'no'} "
                    f"since_enter={deps.fmt_ms(route_t0)}"
                )

                if stream:
                    return StreamingResponse(
                        deps.forward_responses_as_chat_stream(
                            url=deps.RESPONSES_BASE_URL,
                            api_key=deps.RESPONSES_API_KEY,
                            request_data=outbound_data,
                            model=model,
                            messages=data.get('messages', []),
                            trace_id=trace_id,
                            caller_key=caller_key,
                            caller_desc=caller_desc,
                            timeout=deps.get_timeout_config(),
                            max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                            deps=responses_deps,
                        ),
                        media_type='text/event-stream'
                    )

                try:
                    full_content, usage, finish_reason = await deps.collect_responses_as_chat_completion(
                        url=deps.RESPONSES_BASE_URL,
                        api_key=deps.RESPONSES_API_KEY,
                        request_data=outbound_data,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=responses_deps,
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
                    deps.log(f"{trace_prefix} GPT Responses collect error: {e}")
                    deps.save_request_log(
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
                'Authorization': f'Bearer {deps.RESPONSES_API_KEY}',
            }
            outbound_data = dict(outbound_data)
            if deps.GPT_SERVICE_TIER:
                outbound_data['service_tier'] = deps.GPT_SERVICE_TIER
            deps.log(
                f"{trace_prefix} Route to GPT upstream /v1/chat/completions: "
                f"{model} service_tier={outbound_data.get('service_tier', '-')} "
                f"since_enter={deps.fmt_ms(route_t0)}"
            )

            if stream:
                outbound_data['stream'] = True
                return StreamingResponse(
                    deps.forward_chat_completions_stream(
                        url=deps.GPT_BASE_URL,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=deps.build_chat_completions_upstream_deps(),
                        enable_early_stop=is_kimi_model(model),
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                    ),
                    media_type='text/event-stream'
                )

            try:
                outbound_data['stream'] = True
                full_content, model_name, usage, finish_reason, raw_sse = await deps.collect_chat_completions_stream(
                    url=deps.GPT_BASE_URL,
                    request_data=outbound_data,
                    headers=headers,
                    timeout=deps.get_timeout_config(),
                    max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                    deps=deps.build_chat_completions_upstream_deps(),
                    enable_early_stop=is_kimi_model(model),
                    trace_id=trace_id,
                )

                deps.save_request_log(
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
                deps.log(f"{trace_prefix} GPT collect error: {e}")
                deps.save_request_log(
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

        if deps.model_policy.is_responses_model(model):
            removed_fields = deps.model_policy.apply_responses_compat(data)
            outbound_data = deps.convert_chat_to_responses_request(data)
            if removed_fields:
                deps.log(
                    "OpenAI Responses shim: "
                    f"model={model}, "
                    f"input_items={len(outbound_data.get('input', []))}, "
                    f"instructions={'yes' if outbound_data.get('instructions') else 'no'}, "
                    f"removed={','.join(sorted(removed_fields))}"
                )
            else:
                deps.log(
                    "OpenAI Responses shim: "
                    f"model={model}, "
                    f"input_items={len(outbound_data.get('input', []))}, "
                    f"instructions={'yes' if outbound_data.get('instructions') else 'no'}"
                )
            if not deps.RESPONSES_API_KEY:
                if stream:
                    deps.release_active_stream_caller(caller_key, trace_id)
                return JSONResponse(
                    {"error": {"message": "RESPONSES_API_KEY is missing", "type": "config_error"}},
                    status_code=500
                )

            outbound_data = dict(outbound_data)
            outbound_data['stream'] = True
            responses_deps = deps.build_responses_upstream_deps()
            deps.log(f"{trace_prefix} Route to OpenAI Responses upstream /v1/responses: {model} since_enter={deps.fmt_ms(route_t0)}")

            if stream:
                return StreamingResponse(
                    deps.forward_responses_as_chat_stream(
                        url=deps.RESPONSES_BASE_URL,
                        api_key=deps.RESPONSES_API_KEY,
                        request_data=outbound_data,
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=responses_deps,
                    ),
                    media_type='text/event-stream'
                )

            full_content, usage, finish_reason = await deps.collect_responses_as_chat_completion(
                url=deps.RESPONSES_BASE_URL,
                api_key=deps.RESPONSES_API_KEY,
                request_data=outbound_data,
                model=model,
                messages=data.get('messages', []),
                trace_id=trace_id,
                timeout=deps.get_timeout_config(),
                max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                deps=responses_deps,
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
        passthrough_upstream = deps.resolve_model_name_passthrough_upstream(model)
        if passthrough_upstream:
            route_name = passthrough_upstream.get('name') or 'openai-compatible-passthrough'
            route_reason = passthrough_upstream.get('selection_reason') or 'unknown'
            base_url = str(passthrough_upstream.get('base_url') or '').rstrip('/')
            target_url = f"{base_url}/chat/completions"
            outbound_data = copy.deepcopy(data)
            if deps.is_claude_opus55_model(model):
                outbound_data, compat_meta = deps.apply_claude_model_compat_request(outbound_data)
                deps.log(
                    f"{trace_prefix} opus55_adaptive_compat route=model_name_passthrough "
                    f"name={route_name} model={model} removed={compat_meta.get('removed', '-')}"
                )
            if deps.should_apply_deepseek_drawing_context_filter(route_name, base_url):
                filter_stats = deps.apply_drawing_context_filter(outbound_data)
                if filter_stats.get('blocks'):
                    deps.log(
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
            deps.log(
                f"{trace_prefix} model_name_passthrough "
                f"name={route_name} reason={route_reason} model={model} "
                f"url={target_url} since_enter={deps.fmt_ms(route_t0)}"
            )
            if stream:
                outbound_data['stream'] = True
                return StreamingResponse(
                    deps.forward_chat_completions_stream(
                        url=target_url,
                        request_data=outbound_data,
                        headers=headers,
                        timeout=deps.get_timeout_config(),
                        max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                        deps=deps.build_chat_completions_upstream_deps(),
                        enable_early_stop=is_kimi_model(model),
                        model=model,
                        messages=data.get('messages', []),
                        trace_id=trace_id,
                        caller_key=caller_key,
                        caller_desc=caller_desc,
                    ),
                    media_type='text/event-stream'
                )
            try:
                plain_response = await _plain_claude_when_conversion_disabled(
                    deps=deps, model=model, request_data=outbound_data,
                    url=target_url, headers=headers, trace_id=trace_id,
                )
                if plain_response is not None:
                    return plain_response
                outbound_data['stream'] = True
                full_content, model_name, usage, finish_reason, raw_sse = await deps.collect_chat_completions_stream(
                    url=target_url,
                    request_data=outbound_data,
                    headers=headers,
                    timeout=deps.get_timeout_config(),
                    max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES,
                    deps=deps.build_chat_completions_upstream_deps(),
                    enable_early_stop=is_kimi_model(model),
                    trace_id=trace_id,
                )
                deps.save_request_log(
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
                deps.log(f"{trace_prefix} model-name passthrough collect error upstream={route_name}: {e}")
                deps.save_request_log(
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
            deps.release_active_stream_caller(caller_key, trace_id)
        err_msg = (
            f"No upstream route for model {model!r}. Configure it in runtime-flags.json "
            "openai_compatible_upstreams or use a built-in Claude/Gemini/GPT route."
        )
        deps.log(f"{trace_prefix} no_route model={model} since_enter={deps.fmt_ms(route_t0)}")
        deps.save_request_log(
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
        deps.log(f"{trace_prefix} Request timeout")
        return JSONResponse(
            {"error": {"message": "Upstream timeout", "type": "timeout"}},
            status_code=504
        )
    except Exception as e:
        deps.log(f"{trace_prefix} Exception: {e}")
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
            {"error": {"message": str(e), "type": "proxy_error"}},
            status_code=500
        )


CHAT_DEPENDENCY_NAMES = (
    'CLAUDE_DEFAULT_MAX_TOKENS',
    'RESPONSES_API_KEY',
    'RESPONSES_BASE_URL',
    'RESPONSES_REASONING_EFFORT',
    'RESPONSES_REASONING_SUMMARY',
    'GEMINI_API_KEY',
    'GEMINI_ENABLED',
    'GPT_BASE_URL',
    'GPT_PROMPT_CACHE_KEY',
    'GPT_PROMPT_CACHE_RETENTION',
    'GPT_SERVICE_TIER',
    'GPT_USE_RESPONSES',
    'MAX_RAW_SSE_BYTES',
    '_runtime_lookup',
    'active_stream_registry',
    'apply_claude_client_compat_request',
    'apply_claude_keyword_filter',
    'apply_claude_model_compat_request',
    'apply_claude_output_settings',
    'apply_drawing_context_filter',
    'apply_openai_template_thinking_disabled',
    'apply_pro_no_reasoning_payload',
    'build_anthropic_messages_deps',
    'build_caller_fingerprint',
    'build_claude_messages_url',
    'build_claude_prompt_cache_control',
    'build_claude_system_prefix',
    'build_claude_upstream_headers',
    'build_responses_upstream_deps',
    'build_exact_request_key',
    'build_free_openai_chat_url',
    'build_free_openai_headers',
    'build_gemini_generate_content_config',
    'build_gemini_generate_content_deps',
    'build_gpt_prompt_cache_key',
    'build_chat_completions_upstream_deps',
    'build_timed_claude_user_id',
    'collect_anthropic_messages_as_chat_completion',
    'collect_responses_as_chat_completion',
    'collect_gemini_generate_content',
    'collect_chat_completions_nonstream',
    'collect_chat_completions_stream',
    'convert_chat_to_anthropic_messages_request',
    'convert_chat_to_responses_request',
    'extract_openai_chat_payload_content',
    'fake_slow_openai_stream',
    'fmt_ms',
    'fold_claude_system_into_first_user_message',
    'forward_anthropic_messages_as_chat_stream',
    'forward_responses_as_chat_stream',
    'forward_gemini_generate_content_stream',
    'replay_chat_completions_nonstream_as_stream',
    'forward_chat_completions_stream',
    'get_claude_cache_keepalive_settings',
    'get_claude_prompt_caching_settings',
    'get_claude_upstream_for_provider',
    'get_timeout_config',
    'inject_gpt_usage_policies_system_message',
    'is_claude_haiku_model',
    'is_claude_opus55_model',
    'is_claude_opus_model',
    'is_claude_prompt_cache_model',
    'is_claude_sonnet_model',
    'log',
    'model_policy',
    'parse_claude_provider_prefix',
    'parse_free_provider_prefix',
    'release_active_stream_caller',
    'replay_service',
    'resolve_model_name_passthrough_upstream',
    'resolve_openai_compatible_route',
    'run_exact_nonstream_once',
    'save_request_log',
    'schedule_delayed_restart',
    'should_append_pro_opus46_last_user_note',
    'should_apply_deepseek_drawing_context_filter',
    'should_strip_claude_cache_controls',
    'strip_claude_cache_controls',
    'summarize_anthropic_cache_breakpoints',
    'summarize_anthropic_request',
    'summarize_openai_messages',
)


def register_routes(app, ctx: dict[str, Any]) -> RouteDependencies:
    deps = build_route_dependencies(ctx, CHAT_DEPENDENCY_NAMES)
    nonstream_keepalive_interval = float(ctx.get('NONSTREAM_KEEPALIVE_INTERVAL', 10.0))

    async def chat_completions_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return await chat_completions(request, deps)

        if not isinstance(payload, dict) or payload.get('stream', False):
            return await chat_completions(request, deps)

        return keepalive_json_response(
            lambda: chat_completions(request, deps),
            interval=nonstream_keepalive_interval,
            trace_id=getattr(request.state, 'trace_id', ''),
            log=deps.log,
        )

    app.post('/v1/chat/completions')(chat_completions_endpoint)
    register_timed_routes(app, handler=chat_completions, deps=deps)
    return deps
