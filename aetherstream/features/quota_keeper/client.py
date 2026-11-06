"""Private GProxy transport for legacy Claude and native tiny Argon keepalive."""

import json
import re
import math
from datetime import datetime
from pathlib import Path
import time

import httpx

MODEL = 'claude-opus-4-6-thinking'


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

    async def probe(self, credential_id):
        """Fresh native Claude quota + catalogue, never the legacy generation path."""
        observed, models, buckets = await self.diagnostic(credential_id)
        entries = []
        for name in ('3p-5h', '3p-weekly'):
            found = [b for b in buckets if b.get('bucketId') == name]
            if len(found) != 1:
                raise ProbeError('Missing or ambiguous Claude quota')
            value = bucket_reading(found[0])
            duration = 18000 if name == '3p-5h' else 604800
            if value['end'] > observed / 1000 + duration + 300:
                raise ProbeError('Invalid Claude quota duration')
            entries.append({'id': name, 'source_id': 'subscription',
                'label': 'antigravity_disabled' if found[0].get('disabled') is True else None,
                'value': {'kind': 'window', 'used_percent': (1 - value['remaining']) * 100,
                          'period_end': value['end']}})
        return {'sources': [{'capability': {'id': 'subscription'},
                             'observed_at_ms': observed, 'error': None}],
                'entries': entries, 'available_models': sorted(models)}

    async def generate(self, credential_id, *, provider_id, model=MODEL):
        # Native GProxy transport retains account/session identity. Exact
        # provider + credential, ping and maxOutputTokens=16, one core attempt.
        # This never reveals a secret, calls Google directly or uses Stream transforms.
        from .claude import model_name
        model_name(model)
        return await self.model_test(provider_id, credential_id, model)

    async def model_test(self, provider_id, credential_id, model):
        raw = await self.api('models/test', {
            'providerId': native_id('providers', provider_id),
            'credentialId': native_id('credentials', credential_id), 'model': model})
        if not isinstance(raw, dict):
            raise ProbeError('Invalid generation response')
        from .argon import safe_result
        # raw.model is merely the requested-name echo, NOT response identity.
        return safe_result({'http_status': raw.get('status', 0),
            'ok': raw.get('ok') is True and not raw.get('error'), 'usage': raw.get('usage')})

    async def diagnostic(self, credential_id):
        """Fresh, lean native evidence; never retain raw experiments or account labels."""
        started = time.time()
        result = await self.api('credentials/' + native_id('credentials', credential_id)
                                + '/quota-diagnostics', {})
        if not isinstance(result, dict) or not isinstance(result.get('snapshot'), dict):
            raise ProbeError('Invalid quota diagnostic envelope')
        observed = result['snapshot'].get('observedAtMs')
        if (result.get('error') or isinstance(observed, bool)
                or not isinstance(observed, (int, float)) or not math.isfinite(observed)
                or observed < int(started * 1000) - 2000
                or observed > int(time.time() * 1000) + 2000):
            raise ProbeError('No fresh successful quota diagnostic')
        responses = result.get('responses')
        if not isinstance(responses, list):
            raise ProbeError('Invalid quota diagnostic responses')
        models, buckets, catalogue = set(), [], False
        for response in responses:
            if not isinstance(response, dict) or response.get('status') != 200:
                raise ProbeError('Failed quota diagnostic response')
            body = response.get('body')
            if not isinstance(body, dict):
                raise ProbeError('Invalid quota diagnostic body')
            if isinstance(body.get('models'), dict):
                catalogue = True
                models.update(name for name, value in body['models'].items()
                              if isinstance(value, dict) and value.get('disabled') is not True)
            groups = body.get('groups', [])
            if not isinstance(groups, list):
                raise ProbeError('Invalid quota diagnostic groups')
            for group in groups:
                items = group.get('buckets', []) if isinstance(group, dict) else []
                if not isinstance(items, list):
                    raise ProbeError('Invalid quota diagnostic buckets')
                buckets.extend(b for b in items if isinstance(b, dict))
        if not catalogue:
            raise ProbeError('No catalogue evidence')
        return observed, models, buckets

    async def argon_probe(self, credential_id):
        observed, models, buckets = await self.diagnostic(credential_id)
        models = sorted(models & ARGON_MODELS)
        if not models:
            return {'available': False, 'models': [], 'observed_at_ms': observed}
        found = [b for b in buckets if b.get('bucketId') == ARGON_BUCKET]
        if len(found) != 1:
            raise ProbeError('Missing or ambiguous independent Argon weekly bucket')
        if found[0].get('disabled') is True:
            return {'available': False, 'models': models, 'observed_at_ms': observed}
        return {'available': True, 'models': models, 'observed_at_ms': observed,
                **bucket_reading(found[0])}

    async def generate_argon(self, member, model):
        if model not in ARGON_MODELS:
            raise ValueError('Unsupported Argon keepalive model')
        return await self.model_test(member['provider_id'], member['id'], model)


def bucket_reading(bucket):
    try:
        raw = bucket.get('remainingFraction')
        if isinstance(raw, bool):
            raise ValueError('boolean quota')
        remaining = float(raw)
        reset = datetime.fromisoformat(str(bucket['resetTime']).replace('Z', '+00:00'))
        if reset.tzinfo is None or not math.isfinite(remaining) or not 0 <= remaining <= 1:
            raise ValueError('invalid quota')
        return {'remaining': remaining, 'end': reset.timestamp()}
    except (ValueError, TypeError, KeyError, OverflowError):
        raise ProbeError('Invalid quota bucket') from None


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
