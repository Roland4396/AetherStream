"""Configurable early-stop tag matching."""

from collections.abc import Callable
import re
from typing import Any
from urllib.parse import urlsplit

from aetherstream.utils.coerce import coerce_bool, coerce_string_list


DEFAULT_EARLY_STOP_TAGS = ['<!--ST0P_PROXY_', '<disclaimer>', '<closing_leaf>']


def request_uses_content_blocks(request: dict | None) -> bool:
    """Scope the guard to requests that already use the content-block contract."""
    if not isinstance(request, dict):
        return False

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for key in ('content', 'text'):
                if key in value:
                    yield from strings(value[key])

    opened = closed = False
    for value in (request.get('system'), request.get('messages', [])):
        for text in strings(value):
            opened |= bool(re.search(r'<content(?:\s[^<>]*|)>', text))
            closed |= '</content>' in text
            if opened and closed:
                return True
    return False


_CONTENT_STRUCTURE = re.compile(
    r'<!--.*?(?:-->|\Z)|```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`'
    r'|(?P<tag></?(?:content|thinking|think)(?:\s[^<>]*|)>)',
    re.DOTALL,
)


def _content_completed(text: str, before: int) -> bool:
    """Require a real paired body, not an example in thoughts/code/comments."""
    content_depth = thinking_depth = 0
    completed = False
    for match in _CONTENT_STRUCTURE.finditer(text, 0, before):
        tag = match.group('tag')
        if tag is None:
            continue
        closing = tag.startswith('</')
        name = re.match(r'</?(\w+)', tag).group(1)
        if name in ('thinking', 'think'):
            thinking_depth = max(0, thinking_depth + (-1 if closing else 1))
        elif not thinking_depth:
            if closing and content_depth:
                content_depth -= 1
                completed |= content_depth == 0
            elif not closing:
                content_depth += 1
    return completed and content_depth == 0 and thinking_depth == 0


def bind_content_guard(
    find: Callable[[str], int], request: dict | None,
    *, log: Callable[[str], None] | None = None,
) -> Callable[[str], int]:
    """Skip premature disclaimer matches without losing later valid matches.

    The legacy disclaimer-body marker must obey the same guard; otherwise it
    would stop the stream immediately after the ignored opening tag.
    Other configured stop tags and requests without this contract are unchanged.
    """
    if not request_uses_content_blocks(request):
        return find
    reported: set[int] = set()

    def guarded(text: str) -> int:
        offset = 0
        while offset < len(text):
            relative = find(text[offset:])
            if relative < 0:
                return -1
            pos = offset + relative
            tail = text[pos:].lower()
            is_disclaimer = tail.startswith(('<disclaimer>', '[ai_system detected:'))
            if not is_disclaimer or _content_completed(text, pos):
                return pos
            if log is not None and pos not in reported:
                reported.add(pos)
                log(f'early_stop_disclaimer_ignored reason=content_not_complete position={pos}')
            offset = pos + 1
        return -1

    return guarded


def is_himodels_upstream(response: Any, url: str) -> bool:
    """Identify HiModels even when it is selected behind account-pool."""
    account = str(response.headers.get('x-account-pool-id') or '').strip().lower()
    host = (urlsplit(url).hostname or '').lower()
    return (
        account == 'himodels' or account.startswith(('himodels_', 'himodels-'))
        or host == 'himodels.ai' or host.endswith('.himodels.ai')
    )


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
