"""Private request-mode controls; metadata-only, never inference or GPU work."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from aetherstream.api.quota_keeper_routes import authorized
from aetherstream.routing.request_mode import LABELS, configured_models, resolve
from aetherstream.routing.policy_config import RegistryError
from aetherstream.runtime.request_modes import ConfigConflict, RequestModeStore


HEADERS = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}


def register_routes(app, *, runtime_flags, token_file: str, backup_dir: str,
                    catalog, resolve_route, log):
    store = RequestModeStore(runtime_flags, backup_dir=backup_dir)

    async def directory():
        try:
            result = await catalog()
            return [{'id': item['id'], 'channel': item.get('owned_by', '')}
                    for item in result.get('models', [])
                    if isinstance(item, dict) and isinstance(item.get('id'), str)
                    and isinstance(item.get('owned_by', ''), str)], None
        except Exception:
            return [], '模型目录暂时不可用；已有渠道设置仍可查看和修改。'

    def error(message, status):
        return JSONResponse({'ok': False, 'error': message}, status_code=status, headers=HEADERS)

    async def state_payload():
        flags, revision = store.snapshot()
        models, catalog_error = await directory()
        config = flags['channel_policies']
        ids = set(config.get('channels', {})) | {m['channel'] for m in models if m['channel']}
        ids.update(cid for rule in config.get('model_rules', [])
                   if 'force_upstream' in rule.get('policy', {}).get('streaming', {})
                   for cid in rule.get('channels', []))
        channels = []
        for cid in sorted(ids):
            entry = {'id': cid, **resolve(config, cid, ''),
                     'models': [resolve(config, cid, name) for name in configured_models(config, cid)]}
            channels.append(entry)
        return {'ok': True, 'revision': revision, 'choices': LABELS, 'channels': channels,
                'catalog': models, 'catalog_error': catalog_error,
                'default_mode': 'auto', 'precedence': ['channel', 'exact_model'],
                'scope': 'upstream-request-only'}

    @app.get('/admin/request-modes', response_class=HTMLResponse)
    async def page():
        # This shell contains no settings or credentials. Every data endpoint
        # requires the separate private admin token; nothing is put in a URL.
        html = Path(__file__).with_name('request_modes.html').read_text()
        return HTMLResponse(html, headers={**HEADERS, 'Content-Security-Policy':
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"})

    @app.get('/admin/request-modes/state')
    async def state(request: Request):
        if not authorized(request, token_file):
            return error('Unauthorized', 401)
        try:
            return JSONResponse(await state_payload(), headers=HEADERS)
        except RegistryError as exc:
            return error('请求方式配置校验失败：' + str(exc), 503)
        except (OSError, ValueError, KeyError) as exc:
            log(f'Request-mode config unavailable: {type(exc).__name__}')
            return error('请求方式配置不可用；请检查服务器配置。', 503)

    @app.get('/admin/request-modes/resolve')
    async def preview(request: Request, model: str = ''):
        if not authorized(request, token_file):
            return error('Unauthorized', 401)
        if not model.strip() or len(model) > 512:
            return error('请输入有效的公开模型名。', 400)
        try:
            selected = await resolve_route(model)
            flags, revision = store.snapshot()
            info = resolve(flags['channel_policies'], selected['channel'], selected['upstream_model'])
            return JSONResponse({'ok': True, 'revision': revision, 'public_model': model,
                                 'route': selected['route'], **info}, headers=HEADERS)
        except Exception as exc:
            log(f'Request-mode preview failed: {type(exc).__name__}')
            return error('无法解析该模型；未发起推理请求。', 400)

    @app.post('/admin/request-modes/state')
    async def update(request: Request):
        if not authorized(request, token_file):
            return error('Unauthorized', 401)
        try:
            raw = await request.body()
            if len(raw) > 8192:
                return error('设置请求过大。', 413)
            body = json.loads(raw)
            if not isinstance(body, dict) or set(body) != {'revision', 'channel', 'model', 'mode'}:
                return error('仅允许修改一个渠道或模型的请求方式。', 400)
            store.update(**body)
            return JSONResponse(await state_payload(), headers=HEADERS)
        except ConfigConflict:
            return error('设置已被修改，请重新读取后再保存。', 409)
        except (ValueError, KeyError, TypeError):
            return error('请求方式或配置无效，未保存。', 400)
        except OSError as exc:
            log(f'Request-mode save failed: {type(exc).__name__}')
            return error('此实例无法写入设置，请检查配置文件挂载权限。', 503)

    return store
