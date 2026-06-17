"""Project-specific OpenAI-compatible pro-route request adjustments."""

from typing import Any


def is_pro_openai_compatible_route(route_name: Any, base_url: Any) -> bool:
    route = str(route_name or '').strip().lower()
    base = str(base_url or '').strip().lower()
    # Keep project-specific OpenAI-compatible route detection configurable by
    # name/base URL instead of hard-coding private domains.
    return route.startswith('pro-') or 'openai-compatible-pro' in base


def should_append_pro_opus46_last_user_note(route_name: Any, base_url: Any, model: Any) -> bool:
    model_name = str(model or '').strip().lower()
    return is_pro_openai_compatible_route(route_name, base_url) and (
        'claude-opus-4-6' in model_name or 'claude-opus-4-8' in model_name
    )


def apply_pro_no_reasoning_payload(outbound_data: dict, route_name: Any, base_url: Any) -> dict[str, Any]:
    """Force pro OpenAI-compatible requests to avoid upstream reasoning.

    This is request-side latency protection: the pro channel may disconnect long
    silent requests after about 10 minutes, so filtering reasoning in the response
    is not enough. Keep this limited to the pro route to avoid breaking other
    OpenAI-compatible providers.
    """
    if not is_pro_openai_compatible_route(route_name, base_url):
        return {}

    meta: dict[str, Any] = {
        'reasoning_effort': 'none',
        'removed': [],
    }
    for key in (
        'reasoning',
        'thinking',
        'include_reasoning',
        'return_reasoning',
        'reasoning_content',
        'include_thoughts',
        'return_thoughts',
    ):
        if key in outbound_data:
            outbound_data.pop(key, None)
            meta['removed'].append(key)

    outbound_data['reasoning_effort'] = 'none'
    return meta
