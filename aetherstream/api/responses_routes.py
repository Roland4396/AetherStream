from __future__ import annotations

from typing import Any

from fastapi import Request

from aetherstream.api.chat_routes import CHAT_DEPENDENCY_NAMES
from aetherstream.api.dependencies import RouteDependencies, build_route_dependencies
from aetherstream.api.protocol_gateway import route_protocol_request
from aetherstream.streaming.json_keepalive import keepalive_json_response


def register_routes(app, ctx: dict[str, Any]) -> RouteDependencies:
    chat_deps = build_route_dependencies(ctx, CHAT_DEPENDENCY_NAMES)
    nonstream_keepalive_interval = float(ctx.get("NONSTREAM_KEEPALIVE_INTERVAL", 10.0))

    async def responses_endpoint(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return await route_protocol_request(request, chat_deps, protocol="responses")
        if not isinstance(payload, dict) or payload.get("stream", False):
            return await route_protocol_request(request, chat_deps, protocol="responses")
        return keepalive_json_response(
            lambda: route_protocol_request(request, chat_deps, protocol="responses"),
            interval=nonstream_keepalive_interval,
            trace_id=getattr(request.state, "trace_id", ""),
            log=chat_deps.log,
        )

    app.post("/v1/responses")(responses_endpoint)
    return chat_deps

