from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class AnthropicMessagesDeps:
    log: Callable[[str], None]
    save_request_log: Callable[..., None]
    build_openai_sse_error: Callable[[int, str, str], bytes]
    has_stop_tag: Callable[[str], bool]
    find_stop_tag: Callable[[str], int]
    fmt_ms: Callable[[float, float | None], str]
    release_caller: Callable[[str, str], None]
    shared_runtime_state: object = None
    header_keepalive_enabled: bool = False
    header_keepalive_interval_sec: float = 3.0
    stream_idle_timeout_enabled: bool = False
    stream_idle_timeout_sec: float = 4.0
    recover_oauth: Callable[..., Awaitable[bool]] | None = None
    session_key: str | None = None
    session_id: str | None = None
    retire_session: Callable[[str, str], bool] | None = None

    def retire_refused_session(self, stop_reason: str | None, trace_prefix: str = "") -> bool:
        if stop_reason != "refusal" or not self.session_key or not self.session_id:
            return False
        retire = self.retire_session
        if retire is None and self.shared_runtime_state is not None:
            retire = self.shared_runtime_state.retire_session
        if retire is None:
            return False
        try:
            retired = retire(self.session_key, self.session_id)
        except Exception as exc:
            self.log(f"{trace_prefix}claude_session_retire_failed reason=upstream_refusal error={type(exc).__name__}")
            return False
        self.log(
            f"{trace_prefix}claude_session_retired reason=upstream_refusal "
            f"session_key={self.session_key} session={self.session_id[:8]} "
            f"retired={str(retired).lower()}"
        )
        return retired
