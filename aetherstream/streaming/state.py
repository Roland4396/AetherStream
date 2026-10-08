import asyncio
import time
from typing import Callable


class ActiveStreamRegistry:
    def __init__(self, *, log: Callable[[str], None], shared_state=None):
        self.shared_state = shared_state
        self._monitors = {}
        self._shared_traces = {}
        self._log = log
        self._active: dict[str, dict] = {}

    def get(self, caller_key: str) -> dict | None:
        if self.shared_state is not None:
            return self.shared_state.stream_get(caller_key)
        return self._active.get(caller_key)

    def register(
        self,
        caller_key: str,
        *,
        trace_id: str,
        model: str,
        msg_count: int,
        supersede_previous: bool = False,
    ) -> dict | None:
        previous = self._active.get(caller_key)
        if supersede_previous and previous and previous.get("model") == model:
            cancel_event = previous.get("cancel_event")
            if isinstance(cancel_event, asyncio.Event):
                cancel_event.set()
                self._log(
                    f"[TRACE {trace_id}] caller_supersede caller={caller_key} "
                    f"prev_trace={previous.get('trace_id')} model={model}"
                )
        self._active[caller_key] = {
            "trace_id": trace_id,
            "started_at": time.time(),
            "model": model,
            "msg_count": msg_count,
            "cancel_event": asyncio.Event(),
        }
        if self.shared_state is not None:
            metadata = {k: v for k, v in self._active[caller_key].items() if k != 'cancel_event'}
            previous = self.shared_state.stream_register(caller_key, metadata, supersede_previous)
            self._shared_traces[trace_id] = caller_key
            # Only latest-wins models need polling. Other streams remain untouched.
            if model in {'glm-5.2-local', 'deepseek-v4-flash-local'}:
                event = self._active[caller_key]['cancel_event']
                task = asyncio.create_task(self._watch_cancel(trace_id, event))
                self._monitors[trace_id] = task
                task.add_done_callback(lambda done: self._monitors.pop(trace_id, None))
        return previous

    async def _watch_cancel(self, trace_id, event):
        while not event.is_set():
            if await asyncio.to_thread(self.shared_state.stream_cancelled, trace_id):
                event.set()
                self._log(f"[TRACE {trace_id}] caller_supersede cross_release=true")
                return
            await asyncio.sleep(.2)

    def _release_shared(self, trace_id):
        caller = self._shared_traces.pop(trace_id, None)
        if caller is not None:
            self.shared_state.stream_release(caller, trace_id)
        task = self._monitors.get(trace_id)
        if task:
            task.cancel()

    def cancellation_event(self, caller_key: str, trace_id: str) -> asyncio.Event | None:
        current = self._active.get(caller_key)
        if not current or current.get("trace_id") != trace_id:
            return None
        event = current.get("cancel_event")
        return event if isinstance(event, asyncio.Event) else None

    def release(self, caller_key: str, trace_id: str) -> None:
        self._release_shared(trace_id)
        if not caller_key:
            return
        current = self._active.get(caller_key)
        if current and current.get("trace_id") == trace_id:
            self._active.pop(caller_key, None)
            self._log(f"[TRACE {trace_id}] caller_release caller={caller_key}")

    def release_trace(self, trace_id: str) -> None:
        """Release a request at the HTTP lifecycle boundary.

        Provider generators also release by caller key. This trace-based fallback
        is deliberately idempotent and covers providers that do not know about
        the registry, early JSON errors, and local diagnostic streams.
        """
        if not trace_id:
            return
        self._release_shared(trace_id)
        for caller_key, current in list(self._active.items()):
            if current.get("trace_id") != trace_id:
                continue
            self._active.pop(caller_key, None)
            self._log(f"[TRACE {trace_id}] caller_release_trace caller={caller_key}")

    def __len__(self) -> int:
        return len(self._active)
