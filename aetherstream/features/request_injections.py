"""Project request-injection policies shared by every provider route."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aetherstream.features.opus_notes import (
    PRO_OPUS_LAST_USER_APPEND_MARKER,
    PRO_OPUS_LAST_USER_APPEND_TEXT,
    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
    PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
)
from aetherstream.transforms.requests import (
    append_to_last_user_message,
    insert_after_latest_human_message,
)


ASSISTANT_PREFILL_CONTINUATION_TEXT = '''开始吧。JSON 必须合法并符合 schema；可选字段没值就省略，不要写 undefined/null；next_action 中会开始或继续明确性行为的选项必须加 corruption_add:true，多个可同时标注。
<char_cg_assets>
sex_action char/action 必须原样选择；无匹配省略。
chiaki: cowgirl, group_handjob, missionary2, missionary, suspended
mai: blowjob, die_sex, missionary, standing
neina: blowjob, missionary, suspended
nene: blowjob, doggy, hand_doggy, hands_up_sex, happy_missionary, missionary, suspended
rina: blowjob, cowgirl, dogeza, doggy, footjob
rii: blowjob, cowgirl, doggy, footjob2, footjob, suspended
rinka: blowjob, cowgirl, dogeza, footjob, missionary, standing_sex, suspended
sasha: blowjob, doggy, missionary, sit_blowjob, suspended
shino: cowgirl, footjob, missionary
</char_cg_assets>'''


def _coerce_runtime_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {'', '0', 'false', 'no', 'off'}


def append_assistant_prefill_continuation(
    request_data: dict,
    *,
    selected_model: Any,
    trace_prefix: str,
    route_label: str,
    log: Callable[[str], None],
) -> bool:
    """Turn a final assistant prefill into a provider-safe user-ended request."""
    messages = request_data.get('messages')
    if not isinstance(messages, list) or not messages:
        return False
    last = messages[-1]
    if not isinstance(last, dict) or str(last.get('role') or '').strip().lower() != 'assistant':
        return False

    messages.append({
        'role': 'user',
        'content': ASSISTANT_PREFILL_CONTINUATION_TEXT,
    })
    log(f"{trace_prefix} {route_label}_assistant_prefill_continue_user_appended model={selected_model}")
    return True


def _apply_opus_note(
    request_data: dict,
    *,
    selected_model: Any,
    trace_prefix: str,
    route_label: str,
    log: Callable[[str], None],
    illustration_enabled: bool,
    log_prefix: str,
    log_context: str = '',
) -> None:
    messages = request_data.get('messages', [])
    feature_prefix = f'{log_prefix}_' if log_prefix else ''
    context_suffix = f' {log_context.strip()}' if log_context.strip() else ''
    if illustration_enabled:
        illustrated = insert_after_latest_human_message(
            messages,
            PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
            PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
        )
        placement = 'latest_human_message'
        if not illustrated:
            illustrated = append_to_last_user_message(
                messages,
                PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
                PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
            )
            placement = 'last_user_fallback'
        action = 'inserted' if illustrated else 'skipped'
        log(
            f"{trace_prefix} {route_label}_{feature_prefix}latest_human_illustration "
            f"{action} placement={placement} model={selected_model}{context_suffix}"
        )
    else:
        log(
            f"{trace_prefix} {route_label}_{feature_prefix}latest_human_illustration "
            f"disabled_by_runtime_flag model={selected_model}{context_suffix}"
        )

    injected = append_to_last_user_message(
        messages,
        PRO_OPUS_LAST_USER_APPEND_TEXT,
        PRO_OPUS_LAST_USER_APPEND_MARKER,
    )
    action = 'appended' if injected else 'skipped'
    log(
        f"{trace_prefix} {route_label}_{feature_prefix}last_user_note "
        f"{action} model={selected_model}{context_suffix}"
    )


def apply_pioneer_opus_note(
    request_data: dict,
    *,
    selected_model: Any,
    trace_prefix: str,
    route_label: str,
    runtime_lookup: Callable[..., Any],
    is_opus_model: Callable[[Any], bool],
    log: Callable[[str], None],
) -> None:
    if not is_opus_model(selected_model):
        return

    try:
        enabled_value = runtime_lookup('injections', 'pioneer_opus_note', 'enabled')
        illustration_value = runtime_lookup('injections', 'pioneer_opus_note', 'illustration_enabled')
    except Exception as lookup_error:
        log(
            f"{trace_prefix} {route_label}_pioneer_opus_note runtime_flag_error "
            f"error={lookup_error} model={selected_model}"
        )
        enabled_value = None
        illustration_value = None

    if not _coerce_runtime_bool(enabled_value, True):
        log(f"{trace_prefix} {route_label}_pioneer_opus_note disabled_by_runtime_flag model={selected_model}")
        return

    _apply_opus_note(
        request_data,
        selected_model=selected_model,
        trace_prefix=trace_prefix,
        route_label=route_label,
        log=log,
        illustration_enabled=_coerce_runtime_bool(illustration_value, True),
        log_prefix='pioneer_opus',
    )


def apply_direct_opus_note(
    request_data: dict,
    *,
    selected_model: Any,
    trace_prefix: str,
    route_label: str,
    is_opus_model: Callable[[Any], bool],
    log: Callable[[str], None],
) -> bool:
    if not is_opus_model(selected_model):
        log(f"{trace_prefix} {route_label}_last_user_note skipped_non_opus model={selected_model}")
        return False
    _apply_opus_note(
        request_data,
        selected_model=selected_model,
        trace_prefix=trace_prefix,
        route_label=route_label,
        log=log,
        illustration_enabled=True,
        log_prefix='',
    )
    return True


def apply_forced_opus_note(
    request_data: dict,
    *,
    selected_model: Any,
    trace_prefix: str,
    route_label: str,
    log: Callable[[str], None],
    log_context: str = '',
) -> None:
    """Apply the shared Opus note after a route-specific eligibility check."""
    _apply_opus_note(
        request_data,
        selected_model=selected_model,
        trace_prefix=trace_prefix,
        route_label=route_label,
        log=log,
        illustration_enabled=True,
        log_prefix='',
        log_context=log_context,
    )
