"""Bounded public errors for request-owned upstream operations."""
from __future__ import annotations

import json

import httpx


class UpstreamHTTPError(RuntimeError):
    """Keep the HTTP status without requiring callers to parse exception text."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        error = {}
        try:
            parsed = json.loads(body) if len(body) <= 65536 else None
            if isinstance(parsed, dict) and isinstance(parsed.get('error'), dict):
                error = parsed['error']
        except (ValueError, TypeError):
            pass
        message = error.get('message')
        self.public_error = {
            'type': str(error.get('type') or 'upstream_error')[:128],
            'status': status_code,
            'code': status_code,
            'message': (message if isinstance(message, str) and message else body or 'Upstream request failed')[:4000],
        }
        for key, limit in (('code', 128), ('param', 512)):
            value = error.get(key)
            if isinstance(value, str):
                self.public_error[key] = value[:limit]
            elif (value is None and key in error) or type(value) is int:
                self.public_error[key] = value
        # Preserve compatibility with legacy HTTP-status readers.
        super().__init__(f'Upstream error: {status_code} - {body[:4000]}')


def public_stream_error(error: Exception) -> dict:
    if isinstance(error, UpstreamHTTPError):
        return dict(error.public_error)
    if isinstance(error, httpx.HTTPStatusError):
        try:
            body = error.response.text
        except httpx.ResponseNotRead:
            body = 'Upstream request failed'
        return UpstreamHTTPError(error.response.status_code, body).public_error
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return {'type': 'upstream_timeout', 'status': 504, 'code': 504,
                'message': 'Upstream request timed out'}
    # Transport/internal exception strings can contain URLs or credentials.
    return {'type': 'upstream_error', 'status': 502, 'code': 502,
            'message': 'Upstream request failed'}
