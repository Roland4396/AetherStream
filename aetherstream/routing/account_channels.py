"""Pool-owned channel selection client; Stream owns the feature policy."""
from __future__ import annotations

import copy
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from aetherstream.routing import policy_config


class ChannelUnavailable(RuntimeError):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


def settings(lookup) -> dict:
    value = lookup('account_routing')
    return value if isinstance(value, dict) else {}


def enabled(lookup) -> bool:
    return settings(lookup).get('enabled') is True


def connection(lookup) -> tuple[str, str]:
    cfg = settings(lookup)
    base = str(cfg.get('base_url') or '').rstrip('/')
    parts = urlsplit(base)
    if parts.scheme not in {'http', 'https'} or not parts.netloc or parts.username or parts.query or parts.fragment:
        raise ChannelUnavailable('Invalid ACC endpoint configuration')
    key = str(cfg.get('api_key') or os.environ.get(str(cfg.get('api_key_env') or ''), '')).strip()
    if not key:
        raise ChannelUnavailable('ACC service credential is missing')
    return base, key


async def _get(lookup, path: str, *, params: dict | None = None, request_authorization: str = "") -> dict:
    base, key = connection(lookup)
    try:
        headers = {'Authorization': 'Bearer ' + key}
        if request_authorization:
            headers['X-ACC-Passthrough-Authorization'] = request_authorization
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.get(base + path, params=params, headers=headers)
        if response.status_code != 200:
            raise ChannelUnavailable('ACC model is not configured' if response.status_code == 404
                                     else 'ACC channel metadata is unavailable',
                                     404 if response.status_code == 404 else 503)
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError('Expected an object')
        return payload
    except ChannelUnavailable:
        raise
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        # Never include a credential or a provider's response body in errors.
        raise ChannelUnavailable('ACC channel metadata is unavailable') from exc


async def resolve(lookup, model: str, request_authorization: str = "") -> dict[str, Any]:
    result = await _get(lookup, '/internal/channels/resolve', params={'model': model}, request_authorization=request_authorization)
    if (result.get('model') != model or result.get('transport') not in
            {'chat', 'messages', 'responses', 'gemini', 'local'}
            or not result.get('revision')):
        raise ChannelUnavailable('ACC returned an invalid route contract')
    try:
        result = policy_config.enrich_selection(lookup, result)
    except (ValueError, TypeError, KeyError):
        raise ChannelUnavailable('Invalid local Stream feature policy; no inference was sent')
    if not isinstance(result.get('policy'), dict):
        raise ChannelUnavailable('Local Stream feature policy is missing')
    base, key = connection(lookup)
    # This is always ACC's own endpoint and service credential, not a vendor's.
    return {**result, 'base_url': base, 'api_key': key}


async def catalog(lookup, request_authorization: str = "") -> dict:
    result = await _get(lookup, '/v1/models', request_authorization=request_authorization)
    if not isinstance(result.get('data'), list):
        raise ChannelUnavailable('ACC returned an invalid model catalog')
    models = []
    for item in result['data']:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str):
            raise ChannelUnavailable('ACC returned an invalid model entry')
        # Deliberately whitelist the public projection a second time.
        models.append({'id': item['id'], 'object': 'model', 'owned_by': str(item.get('owned_by') or 'acc')})
    return {'models': copy.deepcopy(models), 'routes': {}, 'revision': result.get('revision')}


async def dedupe_key(lookup, plan: dict, payload: dict, headers: dict) -> dict:
    if plan.get('_local_dedupe'):
        return policy_config.dedupe_identity(plan, payload, headers)
    base, key = connection(lookup)
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.post(base + '/internal/channels/dedupe-key',
                headers={'Authorization': 'Bearer ' + key, **{k: v for k, v in headers.items() if k.lower() == 'x-acc-passthrough-authorization'}},
                json={'model': plan['model'], 'request': payload, 'headers': headers})
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or len(str(result.get('key') or '')) != 64:
            raise ValueError('invalid digest')
        return result
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ChannelUnavailable('ACC coalescing identity is unavailable; no inference was sent') from exc
