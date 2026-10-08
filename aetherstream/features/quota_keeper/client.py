"""Async private gproxy admin transport and fixed tiny Claude generation."""

import json
from pathlib import Path
import time
import uuid

import httpx

MODEL = 'claude-opus-4-6-thinking'
GENERATION_URL = 'https://daily-cloudcode-pa.googleapis.com/v1internal:generateContent'


class ProbeError(Exception):
    pass


def read_credentials(path: str) -> dict:
    config = {}
    for line in Path(path).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip().strip('\"').strip("'")
    return {'username': config['GPROXY_ADMIN_USER'], 'password': config['GPROXY_ADMIN_PASSWORD']}


class GproxyClient:
    def __init__(self, base_url: str, credentials_file: str):
        self.base = base_url.rstrip('/') + '/admin/api/'
        url = httpx.URL(self.base)
        if url.scheme not in ('http', 'https') or not url.host:
            raise ValueError('Gproxy base URL requires an HTTP(S) host')
        # Match HTTPX's Host normalization (default ports, IPv6, IDNA).
        # Cookies authenticate unsafe management calls, so v4 requires an
        # explicit same-origin header; this is never sent to inference.
        self.origin = url.scheme + '://' + url.netloc.decode('ascii')
        self.credentials_file = credentials_file
        self.session = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(45, connect=5),
        )
        try:
            await self.api('login', read_credentials(self.credentials_file))
        except BaseException:
            await self.session.aclose()
            raise
        return self

    async def __aexit__(self, *args):
        await self.session.aclose()

    async def api(self, path: str, data=None):
        response = await self.session.request(
            'GET' if data is None else 'POST', self.base + path, json=data,
            headers={'Origin': self.origin} if data is not None else None,
        )
        response.raise_for_status()
        return response.json()

    async def cached(self, credential_id: int):
        return await self.api(f'credentials/{credential_id}/quota')

    async def probe(self, credential_id: int):
        start = time.time()
        result = await self.api(f'credentials/{credential_id}/quota-probe?force=true', {})
        snapshot = result.get('snapshot', {})
        source = next((s for s in snapshot.get('sources', [])
                       if s.get('capability', {}).get('id') == 'subscription'), {})
        if not result.get('raw') or source.get('error') or (
                source.get('observed_at_ms') or 0) < int(start * 1000) - 2000:
            raise ProbeError('No fresh successful subscription quota')
        return snapshot

    async def generate(self, credential_id: int):
        secret = (await self.api(f'credentials/{credential_id}/reveal', {}))['secret']
        if isinstance(secret, str):
            secret = json.loads(secret)
        body = {'model': MODEL, 'project': secret['project_id'],
                'user_prompt_id': uuid.uuid4().hex,
                'request': {'model': MODEL,
                    'contents': [{'role': 'user', 'parts': [{'text': 'Reply with OK only.'}]}],
                    'generationConfig': {'maxOutputTokens': 64,
                        'thinkingConfig': {'includeThoughts': False, 'thinkingBudget': 0}}}}
        # Never use the chat router, another account, another model, or retries.
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False,
                                     timeout=httpx.Timeout(90, connect=10)) as session:
            response = await session.post(GENERATION_URL, json=body, headers={
                'Authorization': 'Bearer ' + secret['access_token'],
                'Content-Type': 'application/json', 'Accept': 'application/json',
                'User-Agent': 'antigravity/cli/1.0.6 linux/amd64',
            })
        secret = None
        result = {'http_status': response.status_code, 'ok': False}
        if not response.is_success:
            return result
        raw = response.json()
        data = raw.get('response', raw)
        reply = ''.join(part.get('text', '') for candidate in data.get('candidates', [])
                        for part in candidate.get('content', {}).get('parts', [])
                        if not part.get('thought'))
        # Store metrics only, never credentials, headers or arbitrary response text.
        usage = data.get('usageMetadata') or {}
        result.update(ok=bool(reply.strip()), response_model=data.get('modelVersion'),
                      usage={key: usage[key] for key in (
                          'promptTokenCount', 'candidatesTokenCount', 'totalTokenCount')
                          if isinstance(usage.get(key), (int, float))})
        return result
