from filelock import FileLock

import hashlib
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime

from fastapi import Request

from aetherstream.features.replay import ReplayPreparationError, parse_replay_record
from aetherstream.observability.refusals import AnthropicRefusalDiagnostics


class SuccessfulHealthAccessFilter(logging.Filter):
    """Hide routine successful access lines without masking failures."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True

        method = str(args[1]).upper()
        path = str(args[2]).partition('?')[0]
        try:
            status_code = int(args[4])
        except (TypeError, ValueError):
            return True

        successful = 200 <= status_code < 400
        routine_request = (
            (method == 'GET' and path in {'/health', '/ready', '/admin/runtime', '/admin/quota-keeper'})
            or (
                method == 'POST'
                and path in {'/v1/chat/completions', '/v1/responses', '/v1/messages'}
            )
        )
        return not (successful and routine_request)


def install_uvicorn_access_log_filter() -> None:
    access_logger = logging.getLogger('uvicorn.access')
    if any(isinstance(item, SuccessfulHealthAccessFilter) for item in access_logger.filters):
        return
    access_logger.addFilter(SuccessfulHealthAccessFilter())


class ProxyLogger:
    def __init__(self, *, debug: bool, log_dir: str, prefix: str = "[STREAM-PROXY]"):
        self.debug = debug
        self.log_dir = log_dir
        self.prefix = prefix
        self.verbose_trace = os.environ.get("STREAM_PROXY_VERBOSE_TRACE", "false").lower() == "true"
        os.makedirs(self.log_dir, exist_ok=True)
        self.counter_file = os.path.join(self.log_dir, "counter.txt")
        self._write_lock = FileLock(os.path.join(self.log_dir, '.request-log.lock'))

    def log(self, msg: str) -> None:
        if self.debug and self._should_emit(msg):
            print(f"{self.prefix} {msg}", file=sys.stderr, flush=True)

    def _should_emit(self, msg: str) -> bool:
        if self.verbose_trace:
            return True

        lower = msg.lower()

        if (
            "asgi_done reason=app_returned" in lower
            and "disconnect_seen=false" in lower
            and "send_error=-" in lower
            and "receive_error=-" in lower
        ):
            return False

        # Always keep failures and disconnects visible.
        if any(token in lower for token in ("error", "exception", "cancelled")):
            return True

        # Default runtime logs: request arrived, request routed, request finished.
        high_signal_tokens = (
            "inbound method=",
            "route to ",
            "request_start",
            "_headers status=",
            "first_data",
            "done reason=",
            "collect_done",
            "nonstream_dedupe",
            "nonstream_replay",
            "nonstream_keepalive",
            "claude_session_retire",
            "claude_nonstream_to_stream ",
            "glm52_official_thinking ",
            "return_shape",
            "saved log #",
            "ai_stage_warning ",
            "ai_scene_filter ",
            "quota_keeper ",
            "runtime_lifecycle ",
            "early_stop_disclaimer_ignored ",
        )
        if any(token in lower for token in high_signal_tokens):
            return True

        # Everything else is verbose trace noise unless explicitly enabled.
        return False

    def fmt_ms(self, start: float, end: float | None = None) -> str:
        if end is None:
            end = time.perf_counter()
        return f"{(end - start) * 1000:.1f}ms"

    def build_caller_fingerprint(self, request: Request) -> tuple[str, str]:
        client_host = request.client.host if request.client else "?"
        client_port = request.client.port if request.client else "?"
        ua = request.headers.get("user-agent", "-")
        origin = request.headers.get("origin", "-")
        referer = request.headers.get("referer", "-")
        x_request_id = request.headers.get("x-request-id", "-")
        x_forwarded_for = request.headers.get("x-forwarded-for", "-")

        raw = f"{client_host}|{ua}|{origin}|{referer}|{x_forwarded_for}"
        caller_key = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
        caller_desc = (
            f"client={client_host}:{client_port} "
            f"ua={self._clip(ua, 90)} origin={self._clip(origin, 80)} "
            f"referer={self._clip(referer, 90)} xreq={self._clip(x_request_id, 40)} "
            f"xff={self._clip(x_forwarded_for, 40)}"
        )
        return caller_key, caller_desc

    def save_request_log(
        self,
        model: str,
        messages: list,
        response: str,
        stream: bool,
        raw_sse: str = "",
        request_payload: dict | None = None,
        inbound_request_payload: dict | None = None,
        debug_meta: dict | None = None,
        error_type: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        with self._write_lock:
            self._save_request_log_locked(
                model=model,
                messages=messages,
                response=response,
                stream=stream,
                raw_sse=raw_sse,
                request_payload=request_payload,
                inbound_request_payload=inbound_request_payload,
                debug_meta=debug_meta,
                error_type=error_type,
                trace_id=trace_id,
            )

    def _save_request_log_locked(
        self,
        *,
        model: str,
        messages: list,
        response: str,
        stream: bool,
        raw_sse: str = "",
        request_payload: dict | None = None,
        inbound_request_payload: dict | None = None,
        debug_meta: dict | None = None,
        error_type: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        log_id = self._get_next_log_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reasoning_response = self._extract_reasoning_response(raw_sse, model)
        refusal = AnthropicRefusalDiagnostics.from_raw(raw_sse)
        refusal_info = refusal.metadata()

        input_payload = {
            "time": timestamp,
            "model": model,
            "stream": stream,
            "messages": messages,
        }
        if trace_id:
            input_payload["trace_id"] = trace_id
        if request_payload is not None:
            input_payload["request"] = request_payload
        if inbound_request_payload is not None:
            input_payload["inbound_request"] = inbound_request_payload
        if isinstance(debug_meta, dict) and debug_meta:
            input_payload["debug"] = debug_meta
        if error_type:
            input_payload["error_type"] = error_type
        if refusal_info:
            input_payload["upstream_refusal"] = refusal_info

        input_file = os.path.join(self.log_dir, f"{log_id:02d}_input.json")
        self._write_text_atomic(
            input_file,
            json.dumps(input_payload, ensure_ascii=False, indent=2),
        )

        output_file = os.path.join(self.log_dir, f"{log_id:02d}_output.txt")
        output_lines = [
            f"Time: {timestamp}",
            f"Model: {model}",
            f"Stream: {stream}",
        ]
        if trace_id:
            output_lines.append(f"TraceId: {trace_id}")
        if error_type:
            output_lines.append(f"ErrorType: {error_type}")
        if refusal_info:
            output_lines.extend([
                f"UpstreamStopReason: {refusal_info['stop_reason']}",
                f"RefusalCategory: {refusal_info['category']}",
                f"RefusalExplanation: {refusal_info['explanation']}",
                f"RefusalSource: {refusal_info['source']}",
                'RefusalDetails: ' + json.dumps(refusal_info['stop_details'], ensure_ascii=False),
            ])
        output_lines.append(f"Length: {len(response)}")
        if reasoning_response:
            output_lines.extend([
                f"ReasoningLength: {len(reasoning_response)}",
                "=" * 50,
                "--- reasoning_content ---",
                reasoning_response,
                "--- content ---",
                response,
            ])
        else:
            output_lines.extend([
                "=" * 50,
                response,
            ])
        self._write_text_atomic(output_file, "\n".join(output_lines))

        raw_file = os.path.join(self.log_dir, f"{log_id:02d}_raw_sse.txt")
        if raw_sse:
            raw_lines = [f"Time: {timestamp}", f"Model: {model}"]
            if trace_id:
                raw_lines.append(f"TraceId: {trace_id}")
            raw_lines.extend(["=" * 50, raw_sse])
            self._write_text_atomic(raw_file, "\n".join(raw_lines))
        else:
            try:
                os.unlink(raw_file)
            except FileNotFoundError:
                pass

        input_size_obj = request_payload if request_payload is not None else {"messages": messages}
        trace_part = f" trace={trace_id}" if trace_id else ""
        self.log(
            f"Saved log #{log_id}{trace_part}: "
            f"input={len(json.dumps(input_size_obj, ensure_ascii=False))} bytes, "
            f"output={len(response)} bytes, reasoning={len(reasoning_response)} bytes, "
            f"raw_sse={len(raw_sse)} bytes{refusal.log_suffix()}"
        )

    @staticmethod
    def _extract_reasoning_response(raw_sse: str, model: str) -> str:
        if not raw_sse:
            return ""
        try:
            record = parse_replay_record(raw_sse, fallback_model=model)
        except (ReplayPreparationError, TypeError, ValueError, json.JSONDecodeError):
            return ""
        return record.reasoning_content

    def _get_next_log_id(self) -> int:
        try:
            with open(self.counter_file, "r", encoding="utf-8") as f:
                counter = int(f.read().strip())
        except Exception:
            counter = 0
        next_id = (counter % 10) + 1
        self._write_text_atomic(self.counter_file, str(next_id))
        return next_id

    @staticmethod
    def _write_text_atomic(path: str, text: str) -> None:
        temp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(temp_path, path)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    @staticmethod
    def _clip(value: str, limit: int = 120) -> str:
        if not isinstance(value, str):
            return "-"
        if len(value) <= limit:
            return value
        return value[:limit] + "..."
