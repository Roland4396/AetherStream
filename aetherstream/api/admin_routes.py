from __future__ import annotations

import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from aetherstream.api.dependencies import RouteDependencies, build_route_dependencies


async def admin_claude_replay_state(deps: RouteDependencies):
    return JSONResponse({
        'ok': True,
        'control': deps.claude_replay.build_state(),
        'entries': deps.claude_replay.list_entries(),
    })


async def admin_claude_replay_update(request: Request, deps: RouteDependencies):
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

    enabled = deps._coerce_bool(body.get('enabled'), False)
    mode = str(body.get('mode', 'always') or 'always').strip().lower()
    match_request = deps._coerce_bool(body.get('match_request'), True)
    raw_sse_path = str(body.get('raw_sse_path', '') or '').strip()
    input_json_path = str(body.get('input_json_path', '') or '').strip()

    if mode not in deps.claude_replay.allowed_modes:
        return JSONResponse(
            {
                'ok': False,
                'error': f"mode must be one of: {', '.join(sorted(deps.claude_replay.allowed_modes))}",
            },
            status_code=400,
        )

    resolved_raw_sse_path = deps.claude_replay.resolve_path(raw_sse_path)
    resolved_input_json_path = deps.claude_replay.resolve_path(input_json_path) if input_json_path else None

    if enabled and not resolved_raw_sse_path:
        return JSONResponse(
            {'ok': False, 'error': 'Enabled replay requires a readable raw_sse_path'},
            status_code=400,
        )

    if enabled and match_request and not resolved_input_json_path:
        resolved_input_json_path = deps.claude_replay.derive_input_json_path(raw_sse_path)
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
        deps.claude_replay.write_control(control)
    except Exception as e:
        deps.log(f"Replay control write failed: {e}")
        return JSONResponse(
            {'ok': False, 'error': f'Failed to write control file: {e}'},
            status_code=500,
        )

    return JSONResponse({
        'ok': True,
        'control': deps.claude_replay.build_state(),
        'entries': deps.claude_replay.list_entries(),
    })


async def admin_claude_replay_log_output(raw_sse_path: str, deps: RouteDependencies):
    output_txt_path = deps.claude_replay.derive_output_txt_path(raw_sse_path)
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


ADMIN_DEPENDENCY_NAMES = (
    '_coerce_bool',
    'claude_replay',
    'log',
)


def register_routes(app, ctx: dict[str, Any]) -> RouteDependencies:
    deps = build_route_dependencies(ctx, ADMIN_DEPENDENCY_NAMES)

    async def replay_state_endpoint():
        return await admin_claude_replay_state(deps)

    async def replay_update_endpoint(request: Request):
        return await admin_claude_replay_update(request, deps)

    async def replay_log_output_endpoint(raw_sse_path: str):
        return await admin_claude_replay_log_output(raw_sse_path, deps)

    app.get('/admin/claude-replay')(replay_state_endpoint)
    app.post('/admin/claude-replay')(replay_update_endpoint)
    app.get('/admin/claude-replay/log-output')(replay_log_output_endpoint)
    # Generic names for new clients; keep the Claude-prefixed paths for the
    # existing SillyTavern extension.
    app.get('/admin/replay')(replay_state_endpoint)
    app.post('/admin/replay')(replay_update_endpoint)
    app.get('/admin/replay/log-output')(replay_log_output_endpoint)
    return deps
