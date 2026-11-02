from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from aetherstream.api.dependencies import RouteDependencies, build_route_dependencies
from aetherstream.routing.account_channels import ChannelUnavailable


async def models(request: Request, deps: RouteDependencies):
    """返回本地模型 + 动态上游模型目录。"""
    try:
        directory = await deps.refresh_model_directory(
            force=True,
            request_authorization=request.headers.get('authorization', ''),
        )
    except ChannelUnavailable as exc:
        return JSONResponse({'error': {'type': 'channel_routing_error', 'message': str(exc)}}, status_code=exc.status)
    models_data = directory.get('models')
    if not isinstance(models_data, list):
        models_data = []
    return JSONResponse({"object": "list", "data": models_data})


async def health(deps: RouteDependencies):
    return JSONResponse({"status": "ok"})


SYSTEM_DEPENDENCY_NAMES = ('refresh_model_directory',)


def register_routes(app, ctx: dict[str, Any]) -> RouteDependencies:
    deps = build_route_dependencies(ctx, SYSTEM_DEPENDENCY_NAMES)

    async def models_endpoint(request: Request):
        return await models(request, deps)

    async def health_endpoint():
        return await health(deps)

    app.get('/v1/models')(models_endpoint)
    app.get('/health')(health_endpoint)
    return deps
