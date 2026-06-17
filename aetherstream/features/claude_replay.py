"""Claude raw-SSE replay control and log listing helpers."""

import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

from aetherstream.transforms.requests import extract_text_from_chat_content
from aetherstream.utils.coerce import coerce_bool


class ClaudeReplayStore:
    allowed_modes = frozenset({'always', 'once', 'sticky'})
    raw_file_re = re.compile(r'^\d+_raw_sse\.txt$')

    def __init__(self, *, log_dir: str, control_file: str, log: Callable[[str], None]):
        self.log_dir = log_dir
        self.control_file = control_file
        self._log = log

    def resolve_path(self, path_value: str | None) -> str | None:
        if not path_value:
            return None

        raw = path_value.strip()
        if not raw:
            return None

        candidates: list[str] = []
        if os.path.isabs(raw):
            candidates.append(raw)
            candidates.append(os.path.join(self.log_dir, os.path.basename(raw)))
        else:
            candidates.append(os.path.join(self.log_dir, raw))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    def load_spec(self, *, model: str, messages: list) -> dict | None:
        if not os.path.isfile(self.control_file):
            return None

        try:
            with open(self.control_file, 'r', encoding='utf-8') as f:
                spec = json.load(f)
        except Exception as e:
            self._log(f'Claude replay control read failed: {e}')
            return None

        if not isinstance(spec, dict):
            self._log('Claude replay control ignored: content is not an object')
            return None

        if spec.get('enabled', True) is not True:
            return None

        raw_sse_path = self.resolve_path(spec.get('raw_sse_path'))
        if not raw_sse_path:
            self._log('Claude replay control ignored: raw_sse_path missing or unreadable')
            return None

        input_json_path = self.resolve_path(spec.get('input_json_path'))
        if not input_json_path and raw_sse_path.endswith('_raw_sse.txt'):
            derived = raw_sse_path[:-len('_raw_sse.txt')] + '_input.json'
            if os.path.isfile(derived):
                input_json_path = derived

        if spec.get('match_request', True) and input_json_path:
            try:
                with open(input_json_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
            except Exception as e:
                self._log(f'Claude replay control ignored: failed to read input_json_path: {e}')
                return None

            saved_model = saved.get('model')
            saved_messages = saved.get('messages')
            if saved_model != model or saved_messages != messages:
                self._log(
                    'Claude replay control skipped: current request does not match saved input '
                    f'model={model} saved_model={saved_model}'
                )
                return None

        try:
            with open(raw_sse_path, 'r', encoding='utf-8') as f:
                raw_sse_text = f.read()
        except Exception as e:
            self._log(f'Claude replay control ignored: failed to read raw_sse_path: {e}')
            return None

        return {
            'mode': str(spec.get('mode', 'sticky')).lower(),
            'raw_sse_path': raw_sse_path,
            'raw_sse_text': raw_sse_text,
        }

    def consume_if_needed(self, spec: dict | None) -> None:
        if not spec:
            return
        if spec.get('mode') != 'once':
            return
        try:
            control = self.read_control()
            control['enabled'] = False
            control['last_consumed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            control['last_consumed_raw_sse_path'] = os.path.basename(str(spec.get('raw_sse_path') or ''))
            self.write_control(control)
            self._log(f'Claude replay control consumed once and disabled: {self.control_file}')
        except FileNotFoundError:
            pass
        except Exception as e:
            self._log(f'Claude replay control disable failed: {e}')

    def default_control(self) -> dict[str, object]:
        return {
            'enabled': False,
            'mode': 'always',
            'match_request': True,
            'raw_sse_path': '',
            'input_json_path': '',
        }

    def read_control(self) -> dict[str, object]:
        control = self.default_control()
        if not os.path.isfile(self.control_file):
            return control

        try:
            with open(self.control_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        except Exception as e:
            self._log(f'Claude replay control state read failed: {e}')
            return control

        if not isinstance(loaded, dict):
            return control

        mode = str(loaded.get('mode', control['mode']) or control['mode']).strip().lower()
        if mode not in self.allowed_modes:
            mode = str(control['mode'])

        control['enabled'] = coerce_bool(loaded.get('enabled'), bool(control['enabled']))
        control['mode'] = mode
        control['match_request'] = coerce_bool(loaded.get('match_request'), bool(control['match_request']))
        control['raw_sse_path'] = str(loaded.get('raw_sse_path', '') or '').strip()
        control['input_json_path'] = str(loaded.get('input_json_path', '') or '').strip()
        return control

    def derive_input_json_path(self, raw_sse_path: str | None) -> str | None:
        if not raw_sse_path:
            return None
        resolved = self.resolve_path(raw_sse_path)
        if not resolved or not resolved.endswith('_raw_sse.txt'):
            return None
        derived = resolved[:-len('_raw_sse.txt')] + '_input.json'
        if os.path.isfile(derived):
            return derived
        return None

    def derive_output_txt_path(self, raw_sse_path: str | None) -> str | None:
        if not raw_sse_path:
            return None
        resolved = self.resolve_path(raw_sse_path)
        if not resolved or not resolved.endswith('_raw_sse.txt'):
            return None
        derived = resolved[:-len('_raw_sse.txt')] + '_output.txt'
        if os.path.isfile(derived):
            return derived
        return None

    def read_log_counter(self) -> int | None:
        try:
            with open(os.path.join(self.log_dir, 'counter.txt'), 'r', encoding='utf-8') as f:
                value = f.read().strip()
        except Exception:
            return None

        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    def build_state(self) -> dict[str, object]:
        control = self.read_control()
        resolved_raw_sse_path = self.resolve_path(str(control.get('raw_sse_path') or ''))

        input_json_path = str(control.get('input_json_path') or '').strip()
        resolved_input_json_path = self.resolve_path(input_json_path)
        if not resolved_input_json_path:
            resolved_input_json_path = self.derive_input_json_path(str(control.get('raw_sse_path') or ''))

        effective_input_json_path = ''
        if resolved_input_json_path:
            effective_input_json_path = os.path.basename(resolved_input_json_path)

        return {
            **control,
            'control_file': self.control_file,
            'log_dir': self.log_dir,
            'raw_sse_exists': bool(resolved_raw_sse_path),
            'input_json_exists': bool(resolved_input_json_path),
            'resolved_raw_sse_path': resolved_raw_sse_path,
            'resolved_input_json_path': resolved_input_json_path,
            'effective_input_json_path': effective_input_json_path,
            'current_counter': self.read_log_counter(),
        }

    def list_entries(self, limit: int = 50) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []

        try:
            file_names = os.listdir(self.log_dir)
        except Exception as e:
            self._log(f'Claude replay entries list failed: {e}')
            return entries

        for file_name in file_names:
            if not self.raw_file_re.match(file_name):
                continue

            raw_sse_path = os.path.join(self.log_dir, file_name)
            try:
                raw_stat = os.stat(raw_sse_path)
            except Exception:
                continue

            input_file_name = file_name[:-len('_raw_sse.txt')] + '_input.json'
            input_json_path = os.path.join(self.log_dir, input_file_name)
            input_exists = os.path.isfile(input_json_path)

            entry: dict[str, object] = {
                'raw_sse_path': file_name,
                'raw_sse_size_bytes': raw_stat.st_size,
                'mtime_ms': int(raw_stat.st_mtime * 1000),
                'input_json_path': input_file_name if input_exists else '',
                'input_json_exists': input_exists,
                'output_txt_path': '',
                'output_txt_exists': False,
                'output_txt_size_bytes': 0,
                'model': '',
                'time': '',
                'stream': None,
                'message_count': 0,
                'first_user_preview': '',
            }

            output_txt_path = self.derive_output_txt_path(file_name)
            if output_txt_path:
                try:
                    output_stat = os.stat(output_txt_path)
                except Exception:
                    output_stat = None
                entry['output_txt_path'] = os.path.basename(output_txt_path)
                entry['output_txt_exists'] = True
                entry['output_txt_size_bytes'] = int(output_stat.st_size) if output_stat else 0

            if input_exists:
                try:
                    with open(input_json_path, 'r', encoding='utf-8') as f:
                        input_payload = json.load(f)
                except Exception as e:
                    entry['input_error'] = str(e)
                else:
                    if isinstance(input_payload, dict):
                        messages = input_payload.get('messages')
                        entry['model'] = str(input_payload.get('model', '') or '')
                        entry['time'] = str(input_payload.get('time', '') or '')
                        entry['stream'] = input_payload.get('stream')
                        entry['message_count'] = len(messages) if isinstance(messages, list) else 0
                        entry['first_user_preview'] = self.extract_first_user_preview(messages)

            entries.append(entry)

        entries.sort(
            key=lambda item: (
                int(item.get('mtime_ms') or 0),
                str(item.get('raw_sse_path') or ''),
            ),
            reverse=True,
        )
        return entries[:limit]

    def extract_first_user_preview(self, messages: Any, limit: int = 180) -> str:
        if not isinstance(messages, list):
            return ''

        for message in messages:
            if not isinstance(message, dict) or message.get('role') != 'user':
                continue
            text = extract_text_from_chat_content(message.get('content'))
            if not text:
                continue
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > limit:
                return text[:limit] + '...'
            return text

        return ''

    def write_control(self, control: dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.control_file), exist_ok=True)
        temp_path = f'{self.control_file}.tmp-{os.getpid()}-{int(time.time() * 1000)}'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(control, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(temp_path, self.control_file)
