"""Configurable early-stop tag matching."""

from collections.abc import Callable
from typing import Any

from aetherstream.utils.coerce import coerce_bool, coerce_string_list


DEFAULT_EARLY_STOP_TAGS = ['<!--ST0P_PROXY_', '<disclaimer>', '<closing_leaf>']


class EarlyStopMatcher:
    def __init__(
        self,
        *,
        lookup: Callable[..., Any],
        env_enabled: bool,
        env_tags: str | None,
        env_case_sensitive: bool,
        default_tags: list[str] | None = None,
    ):
        self._lookup = lookup
        self._env_enabled = env_enabled
        self._env_tags = env_tags
        self._env_case_sensitive = env_case_sensitive
        self._default_tags = list(default_tags or DEFAULT_EARLY_STOP_TAGS)

    def settings(self) -> dict[str, object]:
        enabled = coerce_bool(
            self._lookup('early_stop', 'enabled'),
            self._env_enabled,
        )
        env_tags = coerce_string_list(
            self._env_tags,
            default=self._default_tags,
        )
        tags = coerce_string_list(
            self._lookup('early_stop', 'tags'),
            default=env_tags,
        )
        case_sensitive = coerce_bool(
            self._lookup('early_stop', 'case_sensitive'),
            self._env_case_sensitive,
        )
        return {
            'enabled': enabled,
            'tags': tags,
            'case_sensitive': case_sensitive,
        }

    def find(self, text: str) -> int:
        cfg = self.settings()
        if not cfg.get('enabled') or not isinstance(text, str) or not text:
            return -1
        tags = cfg.get('tags')
        if not isinstance(tags, list) or not tags:
            return -1

        haystack = text if cfg.get('case_sensitive') else text.lower()
        positions = []
        for tag in tags:
            if not isinstance(tag, str) or not tag:
                continue
            needle = tag if cfg.get('case_sensitive') else tag.lower()
            pos = haystack.find(needle)
            if pos >= 0:
                positions.append(pos)
        return min(positions) if positions else -1

    def has(self, text: str) -> bool:
        return self.find(text) >= 0
