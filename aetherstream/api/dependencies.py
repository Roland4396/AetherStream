from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class RouteDependencies:
    """Validated dependency container captured by an API route.

    Route modules use this container instead of mutating their module globals
    from ``app.py``. Attributes remain replaceable so focused tests can swap a
    service without rebuilding the FastAPI application.
    """

    def __init__(self, source: Mapping[str, Any], names: Iterable[str]) -> None:
        dependency_names = tuple(dict.fromkeys(names))
        missing = [name for name in dependency_names if name not in source]
        if missing:
            joined = ', '.join(sorted(missing))
            raise RuntimeError(f'Missing route dependencies: {joined}')

        self._dependency_names = dependency_names
        for name in dependency_names:
            setattr(self, name, source[name])

    @property
    def names(self) -> tuple[str, ...]:
        return self._dependency_names


def build_route_dependencies(
    source: Mapping[str, Any],
    names: Iterable[str],
) -> RouteDependencies:
    return RouteDependencies(source, names)
