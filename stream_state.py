import time
from typing import Callable


class ActiveStreamRegistry:
    def __init__(self, *, log: Callable[[str], None]):
        self._log = log
        self._active: dict[str, dict] = {}

    def get(self, caller_key: str) -> dict | None:
        return self._active.get(caller_key)

    def register(
        self,
        caller_key: str,
        *,
        trace_id: str,
        model: str,
        msg_count: int,
    ) -> dict | None:
        previous = self._active.get(caller_key)
        self._active[caller_key] = {
            "trace_id": trace_id,
            "started_at": time.time(),
            "model": model,
            "msg_count": msg_count,
        }
        return previous

    def release(self, caller_key: str, trace_id: str) -> None:
        if not caller_key:
            return
        current = self._active.get(caller_key)
        if current and current.get("trace_id") == trace_id:
            self._active.pop(caller_key, None)
            self._log(f"[TRACE {trace_id}] caller_release caller={caller_key}")
