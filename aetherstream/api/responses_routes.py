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
        from aetherstream.api.request_entry import protocol_entry
        return await protocol_entry(
            request, chat_deps, lambda: route_protocol_request(request, chat_deps, protocol="responses"), interval=nonstream_keepalive_interval,
        )

    app.post("/v1/responses")(responses_endpoint)
    return chat_deps

