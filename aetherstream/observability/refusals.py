"""Request-local refusal diagnostics; never infer a reason from prompt/output prose.

Only upstream protocol fields are inspected. This module neither changes the
request/response nor controls retries, session retirement, or early stopping.
"""
import json
import re
import unicodedata


_DETAIL_LIMITS = {
    'type': 120, 'category': 120, 'explanation': 4096,
    'reason': 1024, 'message': 4096, 'code': 120,
}
_BEARER = re.compile(r'(?i)\bBearer\s+[^\s,;"\']+')
_SECRET_ASSIGNMENT = re.compile(
    r'(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|'
    r'password|cookie)\b([\s"\']*[:=][\s"\']*)([^\s,;"\']+)'
)
_SECRET_TOKEN = re.compile(
    r'\b(?:sk-[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)'
)


def _safe_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ''
    value = _BEARER.sub('Bearer [REDACTED]', value)
    value = _SECRET_ASSIGNMENT.sub(r'\1\2[REDACTED]', value)
    value = _SECRET_TOKEN.sub('[REDACTED]', value)
    # Bound a single log record and prevent terminal/control-character injection.
    value = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in value)
    value = ' '.join(value.split())
    return value if len(value) <= limit else value[:limit] + ' [truncated]'


class AnthropicRefusalDiagnostics:
    def __init__(self) -> None:
        self.stop_reason = ''
        self.refused = False
        self.details: dict[str, str] = {}
        self.source = ''

    def observe(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get('type')
        if event_type is not None and (
            not isinstance(event_type, str) or event_type not in {
                'message', 'message_start', 'message_delta', 'message_stop',
            }
        ):
            return
        # No recursive walk: text, thinking, tool input, headers and credentials
        # are not diagnostic sources, even when they contain matching keywords.
        for path, node in (('response', event), ('message', event.get('message')),
                           ('delta', event.get('delta'))):
            if not isinstance(node, dict):
                continue
            reason = node.get('stop_reason')
            if isinstance(reason, str) and reason:
                self.stop_reason = _safe_text(reason, 120)
            details = node.get('stop_details')
            explicit_refusal = isinstance(details, dict) and details.get('type') == 'refusal'
            if reason != 'refusal' and not explicit_refusal:
                continue
            self.refused = True
            if not self.source:
                self.source = path + '.stop_reason'
            if isinstance(details, dict):
                clean = {key: text for key, limit in _DETAIL_LIMITS.items()
                         if (text := _safe_text(details.get(key), limit))}
                if clean:
                    self.details.update(clean)
                    self.source = path + '.stop_details'

    def metadata(self) -> dict:
        if not self.refused:
            return {}
        explanation = next((self.details[k] for k in ('explanation', 'message', 'reason')
                            if self.details.get(k)), '上游未提供具体原因')
        return {
            'stop_reason': self.stop_reason or 'refusal',
            'category': self.details.get('category', 'upstream_unspecified'),
            'explanation': explanation,
            'source': self.source,
            'stop_details': dict(self.details),
        }

    def log_suffix(self) -> str:
        info = self.metadata()
        if not info:
            return ''
        fields = {
            'upstream_stop_reason': info['stop_reason'],
            'refusal_category': info['category'],
            'refusal_explanation': info['explanation'],
            'refusal_source': info['source'],
            'refusal_details': info['stop_details'],
        }
        return ''.join(' ' + key + '=' + json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                       for key, value in fields.items())

    @classmethod
    def from_raw(cls, raw: str) -> 'AnthropicRefusalDiagnostics':
        result = cls()
        if not raw:
            return result
        stripped = raw.lstrip()
        if stripped.startswith('{'):
            try:
                result.observe(json.loads(stripped))
                return result
            except (ValueError, RecursionError):
                pass
        for line in raw.splitlines():
            if not line.startswith('data:'):
                continue
            try:
                result.observe(json.loads(line[5:].strip()))
            except (ValueError, RecursionError):
                continue
        return result
