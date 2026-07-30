import json
import logging
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aetherstream.observability.logging import (
    ProxyLogger,
    SuccessfulHealthAccessFilter,
    install_uvicorn_access_log_filter,
)


def build_access_record(method: str, path: str, status_code: int) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=('127.0.0.1:1234', method, path, '1.1', status_code),
        exc_info=None,
    )


class SuccessfulHealthAccessFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = SuccessfulHealthAccessFilter()

    def test_suppresses_successful_health_get(self):
        self.assertFalse(self.filter.filter(build_access_record('GET', '/health', 200)))
        self.assertFalse(self.filter.filter(build_access_record('GET', '/health?probe=1', 204)))

    def test_keeps_health_failures(self):
        self.assertTrue(self.filter.filter(build_access_record('GET', '/health', 500)))

    def test_keeps_other_requests(self):
        self.assertTrue(self.filter.filter(build_access_record('POST', '/health', 200)))
        self.assertTrue(self.filter.filter(build_access_record('GET', '/v1/models', 200)))

    def test_suppresses_successful_chat_completion(self):
        self.assertFalse(self.filter.filter(build_access_record('POST', '/v1/chat/completions', 200)))
        self.assertTrue(self.filter.filter(build_access_record('POST', '/v1/chat/completions', 500)))

    def test_install_is_idempotent(self):
        logger = logging.getLogger('uvicorn.access')
        original_filters = list(logger.filters)
        try:
            logger.filters = [
                item for item in logger.filters
                if not isinstance(item, SuccessfulHealthAccessFilter)
            ]
            install_uvicorn_access_log_filter()
            install_uvicorn_access_log_filter()
            installed = [
                item for item in logger.filters
                if isinstance(item, SuccessfulHealthAccessFilter)
            ]
            self.assertEqual(len(installed), 1)
        finally:
            logger.filters = original_filters


class ProxyLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp.name)
        self.logger = ProxyLogger(debug=False, log_dir=str(self.log_dir))

    def tearDown(self):
        self.temp.cleanup()

    def _save(self, index: int, raw_sse: str = "") -> None:
        self.logger.save_request_log(
            model=f"model-{index}",
            messages=[{"role": "user", "content": str(index)}],
            response=f"response-{index}",
            stream=bool(index % 2),
            raw_sse=raw_sse,
            request_payload={"index": index},
            trace_id=f"trace-{index}",
        )

    def test_reused_slot_removes_stale_raw_sse(self):
        self._save(0, raw_sse="data: old")
        self.assertTrue((self.log_dir / "01_raw_sse.txt").exists())

        for index in range(1, 11):
            self._save(index)

        self.assertFalse((self.log_dir / "01_raw_sse.txt").exists())
        payload = json.loads((self.log_dir / "01_input.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "model-10")

    def test_concurrent_writes_leave_complete_paired_files(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self._save, range(40)))

        input_files = sorted(self.log_dir.glob("[0-9][0-9]_input.json"))
        self.assertEqual(len(input_files), 10)
        for input_file in input_files:
            payload = json.loads(input_file.read_text(encoding="utf-8"))
            slot = input_file.name[:2]
            output = (self.log_dir / f"{slot}_output.txt").read_text(encoding="utf-8")
            self.assertIn(f"Model: {payload['model']}\n", output)
            self.assertIn(f"TraceId: {payload['trace_id']}\n", output)

    def test_suppresses_normal_asgi_completion_but_keeps_failure(self):
        normal = (
            "[TRACE abc] asgi_done reason=app_returned status=200 "
            "disconnect_seen=False send_error=- receive_error=-"
        )
        failed = normal.replace("send_error=-", "send_error=BrokenPipeError")

        self.assertFalse(self.logger._should_emit(normal))
        self.assertTrue(self.logger._should_emit(failed))


if __name__ == "__main__":
    unittest.main()
