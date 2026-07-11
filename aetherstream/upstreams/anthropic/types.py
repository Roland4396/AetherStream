from dataclasses import dataclass
from typing import Callable


@dataclass
class AnthropicUpstreamDeps:
    log: Callable[[str], None]
    save_request_log: Callable[..., None]
    build_openai_sse_error: Callable[[int, str, str], bytes]
    has_stop_tag: Callable[[str], bool]
    find_stop_tag: Callable[[str], int]
    fmt_ms: Callable[[float, float | None], str]
    release_caller: Callable[[str, str], None]
