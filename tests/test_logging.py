import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aetherstream.observability.logging import ProxyLogger


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


if __name__ == "__main__":
    unittest.main()
