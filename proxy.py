"""Backward-compatible ASGI entrypoint.

Keep `uvicorn proxy:app` working while the implementation lives in the
`aetherstream` package.
"""

from aetherstream.api.app import app

__all__ = ["app"]
