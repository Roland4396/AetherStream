"""Request-owned recovery of an explicitly expired Gproxy subscription login.

Only the trusted account-pool route's pre-generation 401 is eligible. This is
not a general inference retry or a background token keeper. Gproxy remains the
owner of credentials and performs the refresh through its existing admin API.
"""
import json
import time

import httpx

from aetherstream.features.quota_keeper.client import GproxyClient


def _origin(url):
    parsed = httpx.URL(url)
    return parsed.scheme, parsed.host, parsed.port


class GproxyOAuthRecovery:
    def __init__(self, *, base_url, credentials_file, trusted_url, shared_state,
                 log, client_factory=None):
        self.trusted_origin = _origin(trusted_url)
        self.shared_state = shared_state
        self.log = log
        self.client_factory = client_factory or (
            lambda: GproxyClient(base_url, credentials_file))
        # This pool route is explicitly bound to the existing Claude provider.
        # Never guess an account from the requested model or refresh a whole pool.
        self.routes = {'gproxy_claude_subscription_1': 'claude'}
        self.key_prefix = 'gproxy-oauth:' + str(_origin(base_url)) + ':'

    async def recover(self, response, url, trace_prefix=''):
        account = response.headers.get('x-account-pool-id', '')
        provider_name = self.routes.get(account)
        if (response.status_code != 401 or not provider_name
                or _origin(url) != self.trusted_origin or self.shared_state is None):
            return False
        content = await response.aread()
        if len(content) > 65536:
            return False
        try:
            body = json.loads(content)
        except (ValueError, TypeError):
            return False
        error = body.get('error') if isinstance(body, dict) else None
        if (not isinstance(error, dict)
                or error.get('type') != 'authentication_error'
                or 'oauth access token has expired' not in str(error.get('message', '')).lower()):
            return False

        async def refresh_selected():
            async with self.client_factory() as client:
                providers = await client.api('providers')
                matched = [p for p in providers if p.get('name') == provider_name
                           and p.get('channel') == 'claudecode' and p.get('enabled')]
                if len(matched) != 1:
                    return False
                rows = await client.api(f"providers/{matched[0]['id']}/credentials")
                credentials = [c for c in rows if c.get('enabled')
                               and c.get('auth_kind') == 'oauth'
                               and c.get('status', 'active') == 'active']
                # With several usable credentials we cannot identify the owner
                # of the rejected token from the pool header. Fail closed.
                if len(credentials) != 1:
                    self.log(f'{trace_prefix}gproxy_oauth_recovery_skip reason=ambiguous_credential')
                    return False
                credential = credentials[0]
                cid = credential['id']
                native_id = f'v3-credentials-{cid}' if isinstance(cid, int) else str(cid)
                current = await client.api(f'credentials/{native_id}')
                expires = current.get('expiresAtMs')
                if isinstance(expires, int) and not isinstance(expires, bool) and expires > int(time.time() * 1000) + 60000:
                    # A concurrent request/native refresh already renewed it.
                    return True
                refreshed = await client.api(f'credentials/{native_id}/refresh?force=true', {})
                expires = refreshed.get('expiresAtMs')
                ok = (refreshed.get('status') == 'active'
                      and isinstance(expires, int) and not isinstance(expires, bool)
                      and expires > int(time.time() * 1000) + 60000)
                self.log(f'{trace_prefix}gproxy_oauth_recovery account={account} credential_id={cid} ok={str(ok).lower()}')
                return ok

        try:
            # Existing kernel-backed cross-process lock. The short-lived value
            # caches a non-secret auth result, not an inference response/lease.
            result, _ = await self.shared_state.exact_response(
                self.key_prefix + provider_name, 2.0, refresh_selected)
            return bool(result)
        except Exception as exc:
            # Do not log exception messages, URLs, headers or token responses.
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            self.log(f'{trace_prefix}gproxy_oauth_recovery_failed error_type={type(exc).__name__} http_status={status}')
            return False
