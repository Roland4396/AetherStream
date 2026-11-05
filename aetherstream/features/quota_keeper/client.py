"""Private GProxy transport for legacy Claude and native tiny Argon keepalive."""

import json
import re
import math
from datetime import datetime
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


    async def argon_probe(self, credential_id):
        """Native diagnostic preserves Argon's independent bucket (legacy quota drops it)."""
        started = time.time()
        result = await self.api('credentials/' + native_id('credentials', credential_id)
                                + '/quota-diagnostics', {})
        if not isinstance(result, dict) or not isinstance(result.get('snapshot'), dict):
            raise ProbeError('Invalid Argon diagnostic envelope')
        observed = result['snapshot'].get('observedAtMs')
        if (result.get('error') or isinstance(observed, bool)
                or not isinstance(observed, (int, float)) or not math.isfinite(observed)
                or observed < int(started * 1000) - 2000
                or observed > int(time.time() * 1000) + 2000):
            raise ProbeError('No fresh successful Argon diagnostic')
        models, buckets, catalogue = [], [], False
        responses = result.get('responses')
        if not isinstance(responses, list):
            raise ProbeError('Invalid Argon diagnostic responses')
        for response in responses:
            if not isinstance(response, dict):
                raise ProbeError('Invalid Argon diagnostic response')
            body = response.get('body')
            if response.get('status') != 200 or not isinstance(body, dict):
                continue
            if isinstance(body.get('models'), dict):
                catalogue = True
                models.extend(name for name, data in body['models'].items()
                              if name in ARGON_MODELS and isinstance(data, dict)
                              and data.get('disabled') is not True)
            for group in body.get('groups', []) if isinstance(body.get('groups'), list) else []:
                items = group.get('buckets', []) if isinstance(group, dict) else []
                if not isinstance(items, list):
                    raise ProbeError('Invalid Argon diagnostic buckets')
                for bucket in items:
                    if isinstance(bucket, dict) and bucket.get('bucketId') == ARGON_BUCKET:
                        buckets.append(bucket)
        if not catalogue:
            raise ProbeError('No Argon catalogue evidence')
        if not models:
            return {'available': False, 'models': [], 'observed_at_ms': observed}
        if len(buckets) != 1:
            raise ProbeError('Missing or ambiguous independent Argon weekly bucket')
        bucket = buckets[0]
        if bucket.get('disabled') is True:
            return {'available': False, 'models': models, 'observed_at_ms': observed}
        raw = bucket.get('remainingFraction')
        try:
            if isinstance(raw, bool):
                raise ValueError('boolean quota')
            remaining = float(raw)
            reset = datetime.fromisoformat(str(bucket['resetTime']).replace('Z', '+00:00'))
            if reset.tzinfo is None or not math.isfinite(remaining) or not 0 <= remaining <= 1:
                raise ValueError('invalid quota')
            end = reset.timestamp()
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ProbeError('Invalid Argon weekly reading') from None
        return {'available': True, 'models': sorted(set(models)), 'remaining': remaining,
                'end': end, 'observed_at_ms': observed}

    async def generate_argon(self, member, model):
        if model not in ARGON_MODELS:
            raise ValueError('Unsupported Argon keepalive model')
        # Existing GProxy admin generation: exact provider + credential, one
        # attempt, no failover, ping input and native maxOutputTokens=16.
        # It uses GProxy's account/session/TLS identity, not the legacy direct
        # Google client, and bypasses Stream's ACT/Kunlun story transforms.
        raw = await self.api('models/test', {
            'providerId': native_id('providers', member['provider_id']),
            'credentialId': native_id('credentials', member['id']), 'model': model})
        from .argon import safe_result
        return safe_result({'http_status': raw.get('status', 0),
                            'ok': raw.get('ok') is True, 'usage': raw.get('usage')})


ARGON_BUCKET = 'gemini-argon-wtu-weekly'
ARGON_MODELS = frozenset('gemini-4-argon-' + level for level in ('low', 'medium', 'high'))


def native_id(family, value):
    if (isinstance(value, bool) or value is None
            or (isinstance(value, int) and value <= 0)
            or family not in ('credentials', 'providers')):
        raise ValueError('Invalid credential/provider identity')
    text = str(value)
    if text.isdigit():
        if int(text) <= 0:
            raise ValueError('Invalid credential/provider identity')
        return 'v3-' + family + '-' + text
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', text):
        return text
    raise ValueError('Invalid credential/provider identity')
