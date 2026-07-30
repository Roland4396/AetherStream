from dataclasses import dataclass
from typing import Callable


@dataclass
class AnthropicMessagesDeps:
    log: Callable[[str], None]
    save_request_log: Callable[..., None]
    build_openai_sse_error: Callable[[int, str, str], bytes]
    has_stop_tag: Callable[[str], bool]
    find_stop_tag: Callable[[str], int]
    fmt_ms: Callable[[float, float | None], str]
    release_caller: Callable[[str, str], None]
    header_keepalive_enabled: bool = False
    header_keepalive_interval_sec: float = 3.0
    stream_idle_timeout_enabled: bool = False
    stream_idle_timeout_sec: float = 4.0
