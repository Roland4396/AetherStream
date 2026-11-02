"""One JSON keepalive/policy-entry implementation for all public protocols."""
from fastapi.responses import JSONResponse

from aetherstream.routing.account_channels import ChannelUnavailable
from aetherstream.streaming.json_keepalive import keepalive_json_response


async def protocol_entry(request, deps, handler, *, interval=10.0):
    try:
        payload = await request.json()
    except Exception:
        return await handler()
    if not isinstance(payload, dict) or payload.get('stream', False):
        return await handler()
    if getattr(deps, 'account_routing_enabled', lambda: False)():
        try:
            plan = await deps.resolve_account_route(str(payload.get('model') or ''), request_authorization=request.headers.get('authorization') or ('Bearer ' + request.headers['x-api-key'] if request.headers.get('x-api-key') else ''))
        except ChannelUnavailable as exc:
            return JSONResponse({'error': {'type': 'channel_routing_error', 'message': str(exc)}}, status_code=exc.status)
        request.state.account_channel_plan = plan
        if not plan['policy'].get('streaming', {}).get('json_keepalive', True):
            # Disabling whitespace heartbeats must not disable disconnect
            # observation. Exact-dedupe work, when opted in, retains its own
            # explicitly tracked cross-request lifecycle barrier.
            interval = float('inf')
    return keepalive_json_response(handler, interval=interval,
                                   trace_id=getattr(request.state, 'trace_id', ''), log=deps.log)
