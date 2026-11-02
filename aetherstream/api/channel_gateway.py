"""Pool-selected, Stream-policy-owned protocol-oriented request pipeline.

No channel names, provider URLs, public-name rewriting or account selection
belong here. Feature settings live in Stream runtime-flags.json; this module executes the
resolved policy once and delegates only wire-format differences to adapters.
"""
from __future__ import annotations

import copy
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any

import httpx
import anyio
from fastapi.responses import JSONResponse

from aetherstream.features.kimi_sampling import apply_kimi_sampling_compat
from aetherstream.features.request_injections import _apply_opus_note, append_assistant_prefill_continuation
from aetherstream.features.scene_filter import remove_scene_messages
from aetherstream.features.stage_warning import wrap_stage_messages
from aetherstream.features.terminal_tool import inject_openai_chat_terminal_tool, prepend_anthropic_terminal_prompt
from aetherstream.features.output_tool import prepare_output_tool, record_marker
from aetherstream.features.replay import ReplayPreparationError
from aetherstream.features.normal_completion_guard import CompletionGuardWarning
from aetherstream.routing.account_channels import ChannelUnavailable
from aetherstream.routing.channel_policy import policy_scope, policy_stream
from aetherstream.streaming.modes import (completion, collect_chat_stream, replay_json_completion,
                                        upstream_stream_mode, pack_shared_result, unpack_shared_result)
from aetherstream.streaming.responses import DisconnectSafeStreamingResponse as StreamingResponse
from aetherstream.upstreams.openai_responses import collect_responses_nonstream
from aetherstream.upstreams.gemini_generate_content import _request_gemini_generate_content, _gemini_nonstream_fields


def apply_request_policy(data: dict, policy: dict, deps, *, trace_id: str) -> dict:
    """One ordered set of filters/injections/field policies for all channels."""
    # These two helpers are shared with the legacy compatibility entrypoint.
    from aetherstream.api.chat_routes import _apply_configured_reasoning, _drop_configured_request_fields
    payload = copy.deepcopy(data)
    model = str(payload.get('model') or '')
    trace = f'[TRACE {trace_id}]'
    filters = policy.get('filters', {})
    adapter = policy.get('adapter', {})
    request_cfg = policy.get('request', {})
    injections = policy.get('injections', {})
    if adapter.get('claude_sampling'):
        # Eligibility is declared by the Stream feature policy; the operation itself is generic.
        if 'temperature' in payload and 'top_p' in payload:
            payload.pop('top_p')
    if adapter.get('kimi_sampling'):
        # Fixed-sampling compatibility is still the one existing implementation.
        apply_kimi_sampling_compat(payload, model)
    if filters.get('scene'):
        remove_scene_messages(payload.get('messages'))
    if filters.get('stage'):
        wrap_stage_messages(payload.get('messages'))
    if filters.get('drawing'):
        deps.apply_drawing_context_filter(payload)
    if policy.get('terminal_tool'):
        inject_openai_chat_terminal_tool(payload)
    if injections.get('gpt_usage'):
        payload = deps.inject_gpt_usage_policies_system_message(payload)
    if injections.get('opus_note'):
        _apply_opus_note(
            payload, selected_model=model, trace_prefix=trace, route_label='channel',
            log=deps.log, illustration_enabled=injections.get('illustration', True), log_prefix='',
            math_enabled=injections.get('math_reminder', True),
        )
    if injections.get('assistant_prefill'):
        append_assistant_prefill_continuation(
            payload, selected_model=model, trace_prefix=trace, route_label='channel', log=deps.log,
        )
    _drop_configured_request_fields(payload, request_cfg)
    if any(isinstance(request_cfg.get(k), bool) for k in ('enable_thinking', 'thinking')) or any(
            request_cfg.get(k) for k in ('effort', 'reasoning_effort')):
        _apply_configured_reasoning(payload, request_cfg)
    elif adapter.get('template_thinking'):
        deps.apply_openai_template_thinking_disabled(payload, model)
    if adapter.get('no_reasoning') and not adapter.get('opus55_compat'):
        # The helper's legacy name detector is satisfied by a feature marker,
        # not by looking at this channel's identity or upstream hostname.
        deps.apply_pro_no_reasoning_payload(payload, 'pro-policy', '')
    if adapter.get('opus55_compat'):
        payload, _ = deps.apply_claude_model_compat_request(payload)
    if request_cfg.get('drop_legacy_models_key'):
        payload.pop('models ', None)
    return payload


@dataclass
class Prepared:
    plan: dict
    payload: dict
    headers: dict
    url: str
    transport_deps: Any
    cache_keepalive: dict | None = None
    gemini_config: Any = None
    output_tool: dict | None = None
    raw_response: str | None = None
    identity_payload: dict | None = None
    identity_plan: dict | None = None


def prepare_adapter(data: dict, plan: dict, deps, *, trace_id: str, upstream_stream: bool) -> Prepared:
    """Serialize one canonical Chat request; preserve its public model name."""
    model = str(data['model'])
    protocol = plan['transport']
    policy = plan['policy']
    base = str(plan['base_url']).rstrip('/')
    key = plan['api_key']
    if base.endswith('/v1'):
        base = base[:-3]
    payload = copy.deepcopy(data)
    payload['stream'] = upstream_stream
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
    if protocol == 'messages':
        session_cfg = policy.get('session', {})
        user_id, session_id, _, _, session_key = deps.build_timed_claude_user_id(
            model=session_cfg.get('model', model), provider=session_cfg.get('provider', 'default'),
        )
        cache = deps.get_claude_prompt_caching_settings()
        cache_enabled = bool(cache.get('enabled') and deps.is_claude_prompt_cache_model(model))
        keepalive = deps.get_claude_cache_keepalive_settings()
        cache_control = deps.build_claude_prompt_cache_control()
        if deps.should_strip_claude_cache_controls():
            payload = deps.strip_claude_cache_controls(payload)
        strategy = ('sonnet_fill_table' if deps.is_claude_sonnet_model(model)
                    else 'opus_roleplay_layered' if deps.is_claude_opus_model(model) else 'generic')
        payload = deps.convert_chat_to_anthropic_messages_request(
            payload, system_prefix=deps.build_claude_system_prefix(model), metadata_user_id=user_id,
            default_max_tokens=int(policy.get('adapter', {}).get('max_tokens', deps.CLAUDE_DEFAULT_MAX_TOKENS)),
            prompt_cache_control=cache_control if cache_enabled and cache.get('mode') == 'explicit' else None,
            prompt_cache_strategy=strategy,
        )
        if policy.get('terminal_tool'):
            prepend_anthropic_terminal_prompt(payload)
        if cache_enabled and cache.get('mode') == 'automatic' and isinstance(cache_control, dict):
            payload['cache_control'] = dict(cache_control)
        payload = deps.apply_claude_output_settings(payload)
        payload, _ = deps.apply_claude_keyword_filter(payload)
        payload, _ = deps.apply_claude_client_compat_request(payload)
        payload, _ = deps.apply_claude_model_compat_request(payload)
        if deps.is_claude_haiku_model(model):
            if deps.should_strip_claude_cache_controls():
                payload = deps.strip_claude_cache_controls(payload)
            payload, _ = deps.fold_claude_system_into_first_user_message(payload)
        payload['stream'] = upstream_stream
        td = deps.build_anthropic_messages_deps(payload)
        td.session_key, td.session_id = session_key, session_id
        headers = deps.build_claude_upstream_headers(session_id=session_id, model=model, api_key=key)
        return Prepared(plan, payload, headers, base + '/v1/messages?beta=true', td,
                        keepalive if cache_enabled and keepalive.get('enabled') else None)
    if protocol == 'responses':
        cfg = policy.get('adapter', {}).get('responses', {})
        payload = deps.convert_chat_to_responses_request(payload)
        payload['stream'] = upstream_stream
        payload.setdefault('store', False)
        if cfg.get('service_tier'):
            payload['service_tier'] = cfg['service_tier']
        payload.setdefault('prompt_cache_key', deps.build_gpt_prompt_cache_key(
            model, payload, str(cfg.get('prompt_cache_key') or '')))
        if cfg.get('prompt_cache_retention'):
            payload.setdefault('prompt_cache_retention', cfg['prompt_cache_retention'])
        reasoning = payload.get('reasoning')
        reasoning = dict(reasoning) if isinstance(reasoning, dict) else {}
        effort = str(cfg.get('reasoning_effort') or '').strip().lower()
        if effort:
            reasoning['effort'] = 'max' if effort == 'ultra' else effort
            if effort == 'ultra':
                reasoning['context'] = 'all_turns'
        if cfg.get('reasoning_summary'):
            reasoning.setdefault('summary', cfg['reasoning_summary'])
        if reasoning:
            payload['reasoning'] = reasoning
        return Prepared(plan, payload, headers, base + '/v1/responses', deps.build_responses_upstream_deps())
    if protocol == 'gemini':
        config = replace(deps.build_gemini_generate_content_config(), base_url=base, api_key=key)
        return Prepared(plan, payload, headers, base, deps.build_gemini_generate_content_deps(), gemini_config=config)
    if protocol == 'chat':
        config = policy.get('runtime', {}).get('output_tool')
        identity_payload = copy.deepcopy(payload) if config else None
        output_tool = prepare_output_tool(payload, config, log=deps.log,
                                         trace_prefix=f'[TRACE {trace_id}] ') if config else None
        identity_plan = None
        if config is not None:
            # This changes the delivery envelope, not the visible request.
            # Preserve exact-request locks/cache keys across rollout and toggles.
            identity_plan = copy.deepcopy(plan)
            identity_plan['policy']['runtime'].pop('output_tool', None)
        return Prepared(plan, payload, headers, base + '/v1/chat/completions',
                        deps.build_chat_completions_upstream_deps(), output_tool=output_tool,
                        identity_payload=identity_payload,
                        identity_plan=identity_plan)
    raise ChannelUnavailable('Unsupported transport', 400)


def forward_prepared(prepared: Prepared, deps, *, data: dict, trace_id: str,
                     caller_key: str = '', caller_desc: str = '', supersede_event=None):
    plan = prepared.plan
    common = dict(url=prepared.url, request_data=prepared.payload, model=data['model'],
                  messages=data.get('messages', []), trace_id=trace_id, caller_key=caller_key,
                  caller_desc=caller_desc, timeout=deps.get_timeout_config(),
                  max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES, deps=prepared.transport_deps)
    if plan['transport'] == 'chat':
        return deps.forward_chat_completions_stream(
            **common, headers=prepared.headers,
            enable_early_stop=bool(plan['policy'].get('early_stop', {}).get('enabled')),
            supersede_event=supersede_event,
            **({'output_tool': prepared.output_tool} if prepared.output_tool is not None else {}),
        )
    if plan['transport'] == 'messages':
        return deps.forward_anthropic_messages_as_chat_stream(
            **common, headers=prepared.headers, cache_keepalive=prepared.cache_keepalive,
        )
    if plan['transport'] == 'responses':
        return deps.forward_responses_as_chat_stream(**common, api_key=plan['api_key'], request_headers=prepared.headers)
    raise ChannelUnavailable('Selected protocol has no native streaming transport', 400)


async def collect_prepared(prepared: Prepared, deps, *, data: dict, trace_id: str,
                           upstream_stream: bool) -> dict:
    model = str(data['model'])
    if upstream_stream:
        return await collect_chat_stream(forward_prepared(prepared, deps, data=data, trace_id=trace_id), model=model)
    common = dict(url=prepared.url, request_data=prepared.payload, timeout=deps.get_timeout_config(),
                  deps=prepared.transport_deps, trace_id=trace_id)
    protocol = prepared.plan['transport']
    if protocol == 'chat':
        collected = await deps.collect_chat_completions_nonstream(
            **common, headers=prepared.headers,
            **({'output_tool': prepared.output_tool} if prepared.output_tool is not None else {}))
        if prepared.output_tool is not None:
            prepared.raw_response = record_marker(prepared.output_tool) + collected[4]
        return collected[5]  # Preserve the complete JSON, including tool calls.
    if protocol == 'messages':
        content, actual, usage, finish, _, calls = await deps.collect_anthropic_messages_as_chat_completion(
            **common, headers=prepared.headers, model=model, messages=data.get('messages', []),
            max_raw_sse_bytes=deps.MAX_RAW_SSE_BYTES, upstream_stream=False,
        )
        return completion(actual or model, content, usage, finish, calls)
    if protocol == 'responses':
        content, usage, finish, _, calls = await collect_responses_nonstream(
            **common, api_key=prepared.plan['api_key'], model=model, messages=data.get('messages', []), save_log=False,
            request_headers=prepared.headers,
        )
        return completion(model, content, usage, finish, calls)
    if protocol == 'gemini':
        native = await _request_gemini_generate_content(
            model=model, openai_request=prepared.payload, config=prepared.gemini_config,
            deps=prepared.transport_deps, trace_id=trace_id,
        )
        content, usage, finish, calls = _gemini_nonstream_fields(native, include_thoughts=prepared.gemini_config.include_thoughts)
        return completion(model, content, usage, finish, calls)
    raise ChannelUnavailable('Unsupported transport', 400)


async def chat_completions(request, deps):
    trace_id = getattr(request.state, 'trace_id', None) or request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    request.state.trace_id = trace_id
    caller_key, caller_desc = deps.build_caller_fingerprint(request)
    handed_off = registered = False
    inbound = None
    try:
        inbound = await request.json()
        if not isinstance(inbound, dict) or not isinstance(inbound.get('model'), str):
            raise ChannelUnavailable('Expected a JSON object with a model name', 400)
        inbound = copy.deepcopy(inbound)
        model = inbound['model']
        stream = inbound.get('stream') is True
        replay = deps.replay_service.prepare(model=model, messages=inbound.get('messages', []))
        if replay is not None:
            if stream:
                return StreamingResponse(deps.replay_service.stream(
                    replay, model=model, messages=inbound.get('messages', []), request_payload=inbound,
                    trace_id=trace_id, caller_key=caller_key, caller_desc=caller_desc,
                ), media_type='text/event-stream')
            return JSONResponse(deps.replay_service.complete(
                replay, model=model, messages=inbound.get('messages', []), request_payload=inbound, trace_id=trace_id,
            ))
        plan = getattr(request.state, 'account_channel_plan', None)
        if not isinstance(plan, dict) or plan.get('model') != model:
            plan = await deps.resolve_account_route(model, request_authorization=request.headers['authorization']) if request.headers.get('authorization') else await deps.resolve_account_route(model)
        policy = plan['policy']
        deps.normal_completion_guard.check(plan, trace_id=trace_id, log=deps.log)
        # Policy resolution does not advance round-robin; no account has been
        # selected yet. The subsequent inference request is the only attempt.
        deps.log(f'[TRACE {trace_id}] channel_policy channel={plan["channel"]} route={plan["route"]} '
                 f'transport={plan["transport"]} revision={plan["revision"]} model={model}')
        with policy_scope(policy, inbound):
            if stream:
                deps.active_stream_registry.register(
                    caller_key, trace_id=trace_id, model=model,
                    msg_count=len(inbound.get('messages', [])), supersede_previous=bool(policy.get('latest_wins')),
                )
                registered = True
            supersede = deps.active_stream_registry.cancellation_event(caller_key, trace_id) if (
                registered and policy.get('latest_wins')) else None
            if plan['transport'] == 'local':
                if stream:
                    body = deps.fake_slow_openai_stream(
                        model=model, trace_id=trace_id, chunks=int(inbound.get('fake_chunks') or 1200),
                        delay=float(inbound.get('fake_delay') or 0.5), token=str(inbound.get('fake_token') or '假流'),
                    )
                    handed_off = True
                    return StreamingResponse(_owned_stream(body, policy, inbound, deps, caller_key, trace_id),
                        media_type='text/event-stream',
                        on_close=lambda: deps.release_active_stream_caller(caller_key, trace_id))
                return JSONResponse(completion(model, str(inbound.get('fake_token') or '假流')))
            data = apply_request_policy(inbound, policy, deps, trace_id=trace_id)
            mode = upstream_stream_mode(policy, stream)
            mode_info = plan.get('request_mode', {})
            deps.log(f'[TRACE {trace_id}] request_mode client_stream={str(stream).lower()} '
                     f'upstream_stream={str(mode).lower()} '
                     f'mode={policy.get("streaming", {}).get("force_upstream", "auto")} '
                     f'source={mode_info.get("source", "request-policy")} '
                     f'channel={plan["channel"]} upstream_model={plan.get("upstream_model", model)}')
            prepared = prepare_adapter(data, plan, deps, trace_id=trace_id, upstream_stream=mode)
            if policy.get('adapter', {}).get('caller_authorization') and request.headers.get('authorization'):
                prepared.headers['X-ACC-Passthrough-Authorization'] = request.headers['authorization']

            async def collect():
                # This coroutine may run in a JSON keepalive/dedupe task. Bind
                # an immutable request policy explicitly rather than a global.
                with policy_scope(policy, inbound), deps.normal_completion_guard.operation(plan, trace_id=trace_id, log=deps.log):
                    result = await collect_prepared(prepared, deps, data=data, trace_id=trace_id, upstream_stream=mode)
                    deps.save_request_log(
                        model, inbound.get('messages', []), deps.extract_openai_chat_payload_content(result),
                        stream=stream, request_payload=prepared.payload, inbound_request_payload=inbound,
                        raw_sse=prepared.raw_response or json.dumps(result, ensure_ascii=False), trace_id=trace_id,
                    )
                    return result

            if stream:
                if mode:
                    body = forward_prepared(prepared, deps, data=data, trace_id=trace_id,
                                            caller_key=caller_key, caller_desc=caller_desc, supersede_event=supersede)
                    body = deps.normal_completion_guard.stream(body, plan, trace_id=trace_id, log=deps.log)
                else:
                    def replay_error(error, exception_type):
                        deps.log(f'[TRACE {trace_id}] nonstream_replay_error '
                                 f'status={error.get("status", "-")} exception={exception_type or "upstream_payload"}')
                        deps.save_request_log(
                            model, inbound.get('messages', []), '[ERROR] ' + str(error.get('message', 'Upstream error')),
                            stream=True, request_payload=prepared.payload, inbound_request_payload=inbound,
                            raw_sse='data: ' + json.dumps({'error': error}, ensure_ascii=False) + '\n\n',
                            error_type='channel_upstream_error', trace_id=trace_id,
                        )
                    body = replay_json_completion(collect, model=model, on_error=replay_error)
                response = StreamingResponse(
                    _owned_stream(body, policy, inbound, deps, caller_key, trace_id), media_type='text/event-stream',
                    on_close=lambda: deps.release_active_stream_caller(caller_key, trace_id),
                )
                response.channel_managed = True
                deadline = float(policy.get('streaming', {}).get('deadline_seconds') or 0)
                if deadline:
                    from aetherstream.api.timed_routes import TimedSSEResponse
                    # Optional policy deadline is independent from channel
                    # guessing; cleanup remains the same actual-disconnect path.
                    response = TimedSSEResponse(response, deadline=time.monotonic() + deadline,
                                                seconds=deadline, trace_id=trace_id, deps=deps)
                handed_off = True
                return response
            # Existing detached/shared exact-request semantics are retained for
            # JSON clients. SSE cancellation never enters this barrier.
            dedupe_key = deps.build_exact_request_key({
                'scope': 'acc-channel-json-v1', 'route': plan['route'],
                'policy': prepared.identity_plan['policy'] if prepared.identity_plan is not None else policy,
                'url': prepared.url, 'headers': prepared.headers,
                'request': prepared.identity_payload if prepared.identity_payload is not None else prepared.payload,
            })
            if policy.get('streaming', {}).get('exact_dedupe'):
                identity = await deps.build_account_dedupe(
                    prepared.identity_plan or plan,
                    prepared.identity_payload if prepared.identity_payload is not None else prepared.payload,
                    prepared.headers)
                dedupe_key, shape = identity['key'], identity['shape']
                async def collect_shared():
                    return pack_shared_result(await collect(), shape)
                cached, shared = await deps.run_exact_nonstream_once(
                    dedupe_key=dedupe_key, trace_id=trace_id, upstream_label=f'acc:{plan["route"]}:{model}', runner=collect_shared,
                )
                result = unpack_shared_result(cached, shape, model)
            else:
                result, shared = await collect(), False
            if shared:
                deps.log(f'[TRACE {trace_id}] nonstream_dedupe_return_shared scope=acc key={dedupe_key[:16]}')
            return JSONResponse(result)
    except CompletionGuardWarning as exc:
        return JSONResponse({'error': {'type': 'normal_completion_guard', 'code': 'confirmation_required',
                                      'message': str(exc), 'retry_required': True, 'upstream_sent': False}}, status_code=409)
    except (ChannelUnavailable, ReplayPreparationError) as exc:
        status = exc.status if isinstance(exc, ChannelUnavailable) else 422
        return JSONResponse({'error': {'type': 'channel_routing_error', 'message': str(exc)}}, status_code=status)
    except httpx.TimeoutException:
        return JSONResponse({'error': {'type': 'upstream_timeout', 'message': 'Upstream timeout'}}, status_code=504)
    except Exception as exc:
        deps.log(f'[TRACE {trace_id}] channel_request_failed type={type(exc).__name__}')
        if isinstance(inbound, dict):
            deps.save_request_log(inbound.get('model', ''), inbound.get('messages', []), '[ERROR] ' + str(exc),
                                  stream=bool(inbound.get('stream')), request_payload=inbound,
                                  error_type='channel_upstream_error', trace_id=trace_id)
        return JSONResponse({'error': {'type': 'channel_upstream_error', 'message': str(exc)}}, status_code=502)
    finally:
        if registered and not handed_off:
            deps.release_active_stream_caller(caller_key, trace_id)


async def _owned_stream(body, policy, inbound, deps, caller_key, trace_id):
    scoped = policy_stream(body, policy, inbound)
    try:
        async for chunk in scoped:
            yield chunk
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            try:
                await scoped.athrow(asyncio.CancelledError())
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        raise
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await scoped.aclose()
            finally:
                deps.release_active_stream_caller(caller_key, trace_id)
