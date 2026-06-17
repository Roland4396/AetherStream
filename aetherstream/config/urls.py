"""URL normalization helpers for upstream API bases."""


def normalize_openai_chat_base_url(raw_url: str) -> str:
    base = (raw_url or '').strip().rstrip('/')
    if not base:
        return 'https://api.openai.com/v1/chat/completions'

    if base.endswith('/v1'):
        return f'{base}/chat/completions'

    return base


def normalize_codex_responses_base_url(raw_url: str) -> str:
    base = (raw_url or '').strip().rstrip('/')
    if not base:
        return 'https://api.openai.com/v1/responses'

    if base.endswith('/v1'):
        return f'{base}/responses'

    return base
