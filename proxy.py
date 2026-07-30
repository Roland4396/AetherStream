"""Backward-compatible ASGI entrypoint.

Keep `uvicorn proxy:app` working while the implementation lives in the
`aetherstream` package.
"""

from aetherstream.observability.logging import install_uvicorn_access_log_filter

install_uvicorn_access_log_filter()

from aetherstream.api.app import app

__all__ = ["app"]
