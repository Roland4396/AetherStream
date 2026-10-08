"""Private, separately authenticated quota-keeper observability and recovery."""

import hmac
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

from aetherstream.features.quota_keeper.state import StateError


def authorized(request: Request, token_file: str) -> bool:
    try:
        expected = Path(token_file).read_text().strip() if token_file else ''
    except OSError:
        return False
    scheme, _, provided = request.headers.get('authorization', '').partition(' ')
    return (len(expected) >= 32 and scheme.lower() == 'bearer' and
            hmac.compare_digest(expected.encode(), provided.encode()))


def register_routes(app, *, keeper, admin_token_file: str):
    async def status(request: Request):
        if not authorized(request, admin_token_file):
            return JSONResponse({'ok': False, 'error': 'Unauthorized'}, status_code=401)
        return JSONResponse({'ok': True, **keeper.status()})

    async def resume(credential_id: int, request: Request):
        if not authorized(request, admin_token_file):
            return JSONResponse({'ok': False, 'error': 'Unauthorized'}, status_code=401)
        if credential_id <= 0:
            return JSONResponse({'ok': False, 'error': 'Invalid credential id'}, status_code=400)
        try:
            result = await keeper.resume(credential_id)
        except (StateError, OSError):
            return JSONResponse({'ok': False, 'error': 'Keeper state unavailable'}, status_code=503)
        if result == 'busy':
            return JSONResponse({'ok': False, 'error': 'Keeper is busy; retry later'}, status_code=409)
        if result == 'not_found':
            return JSONResponse({'ok': False, 'error': 'Credential not found in keeper state'}, status_code=404)
        return JSONResponse({'ok': True, 'result': result, 'credential_id': credential_id})

    app.get('/admin/quota-keeper')(status)
    app.post('/admin/quota-keeper/credentials/{credential_id}/resume')(resume)
