"""Provider-independent replay control and log listing helpers."""

import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from aetherstream.features.replay import ReplayPreparationError, parse_replay_record
from aetherstream.transforms.requests import extract_text_from_chat_content
from aetherstream.utils.coerce import coerce_bool


class ReplayStore:
    allowed_modes = frozenset({'always', 'once', 'sticky'})
    raw_file_re = re.compile(r'^\d+_raw_sse\.txt$')

    def __init__(self, *, log_dir: str, control_file: str, log: Callable[[str], None]):
        self.log_dir = log_dir
        self.control_file = control_file
        self._log = log
        self._control_lock = threading.RLock()

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

    @staticmethod
    def _read_output_text(output_txt_path: str | None) -> str:
        if not output_txt_path:
            return ''
        try:
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        except Exception:
            return ''
        for index, line in enumerate(lines[:12]):
            if line and set(line) == {'='}:
                return '\n'.join(lines[index + 1:])
        return '\n'.join(lines)

    def lookup(self, *, model: str, messages: list) -> dict[str, Any]:
        """Return a structured replay lookup result.

        Disabled and request-mismatch results are normal fall-through states.
        Invalid means replay was explicitly enabled but cannot be executed; the
        caller should fail closed instead of unexpectedly spending an upstream
        request.
        """
        with self._control_lock:
            if not os.path.isfile(self.control_file):
                return {'status': 'disabled'}

            try:
                control = self.read_control(strict=True)
            except ReplayPreparationError as control_error:
                reason = str(control_error)
                self._log(f'Replay control invalid: {reason}')
                return {'status': 'invalid', 'reason': reason}
            if not bool(control.get('enabled')):
                return {'status': 'disabled'}

            raw_sse_path = self.resolve_path(str(control.get('raw_sse_path') or ''))
            if not raw_sse_path:
                reason = 'replay raw_sse_path is missing or unreadable'
                self._log(f'Replay control invalid: {reason}')
                return {'status': 'invalid', 'reason': reason}

            input_json_path = self.resolve_path(str(control.get('input_json_path') or ''))
            if not input_json_path:
                input_json_path = self.derive_input_json_path(raw_sse_path)

            if bool(control.get('match_request')):
                if not input_json_path:
                    reason = 'match_request=true but input_json_path is missing or unreadable'
                    self._log(f'Replay control invalid: {reason}')
                    return {'status': 'invalid', 'reason': reason}
                try:
                    with open(input_json_path, 'r', encoding='utf-8') as f:
                        saved = json.load(f)
                except Exception as e:
                    reason = f'failed to read replay input_json_path: {e}'
                    self._log(f'Replay control invalid: {reason}')
                    return {'status': 'invalid', 'reason': reason}

                saved_model = saved.get('model') if isinstance(saved, dict) else None
                saved_messages = saved.get('messages') if isinstance(saved, dict) else None
                if saved_model != model or saved_messages != messages:
                    self._log(
                        'Replay control skipped: current request does not match saved input '
                        f'model={model} saved_model={saved_model}'
                    )
                    return {'status': 'mismatch'}

            try:
                with open(raw_sse_path, 'r', encoding='utf-8') as f:
                    raw_sse_text = f.read()
            except Exception as e:
                reason = f'failed to read replay raw_sse_path: {e}'
                self._log(f'Replay control invalid: {reason}')
                return {'status': 'invalid', 'reason': reason}

            output_txt_path = self.derive_output_txt_path(raw_sse_path)
            return {
                'status': 'ready',
                'spec': {
                    'mode': str(control.get('mode') or 'always').lower(),
                    'raw_sse_path': raw_sse_path,
                    'raw_sse_text': raw_sse_text,
                    'input_json_path': input_json_path,
                    'output_txt_path': output_txt_path,
                    'output_text': self._read_output_text(output_txt_path),
                },
            }

    def load_spec(self, *, model: str, messages: list) -> dict | None:
        """Backward-compatible wrapper for older route integrations."""
        result = self.lookup(model=model, messages=messages)
        return result.get('spec') if result.get('status') == 'ready' else None

    def consume_if_needed(self, spec: dict | None) -> bool:
        """Atomically claim a prepared replay when it is configured as once."""
        if not spec:
            return False
        if spec.get('mode') != 'once':
            return True
        try:
            with self._control_lock:
                control = self.read_control(strict=True)
                current_path = self.resolve_path(str(control.get('raw_sse_path') or ''))
                expected_path = str(spec.get('raw_sse_path') or '')
                if not bool(control.get('enabled')) or not current_path:
                    return False
                if os.path.realpath(current_path) != os.path.realpath(expected_path):
                    self._log('Replay once consume skipped: control changed after preparation')
                    return False
                control['enabled'] = False
                control['last_consumed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                control['last_consumed_raw_sse_path'] = os.path.basename(expected_path)
                self.write_control(control)
            self._log(f'Replay control consumed once and disabled: {self.control_file}')
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            self._log(f'Replay control disable failed: {e}')
            return False

    def default_control(self) -> dict[str, object]:
        return {
            'enabled': False,
            'mode': 'always',
            'match_request': True,
            'raw_sse_path': '',
            'input_json_path': '',
        }

    def read_control(self, *, strict: bool = False) -> dict[str, object]:
        control = self.default_control()
        if not os.path.isfile(self.control_file):
            return control

        try:
            with self._control_lock:
                with open(self.control_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
        except Exception as e:
            self._log(f'Replay control state read failed: {e}')
            if strict:
                raise ReplayPreparationError(f'failed to read replay control: {e}') from e
            return control

        if not isinstance(loaded, dict):
            if strict:
                raise ReplayPreparationError('replay control must be a JSON object')
            return control

        mode = str(loaded.get('mode', control['mode']) or control['mode']).strip().lower()
        if mode not in self.allowed_modes:
            if strict:
                allowed = ', '.join(sorted(self.allowed_modes))
                raise ReplayPreparationError(f'replay control mode must be one of: {allowed}')
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
        control_error = ''
        try:
            control = self.read_control(strict=True)
        except ReplayPreparationError as error:
            control = self.default_control()
            control_error = str(error)
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
            'control_error': control_error,
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
            self._log(f'Replay entries list failed: {e}')
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
                'replay_usable': False,
                'replay_format': '',
                'replay_source_complete': False,
                'replay_snapshot': False,
                'replay_output_chars': 0,
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

            try:
                with open(raw_sse_path, 'r', encoding='utf-8') as f:
                    raw_sse_text = f.read()
                output_text = self._read_output_text(output_txt_path)
                record = parse_replay_record(
                    raw_sse_text,
                    fallback_model=str(entry.get('model') or 'unknown'),
                    fallback_output_text=output_text,
                )
            except (OSError, ReplayPreparationError) as replay_error:
                entry['replay_error'] = str(replay_error)
            else:
                entry['replay_usable'] = True
                entry['replay_format'] = record.source_format
                entry['replay_source_complete'] = record.source_complete
                entry['replay_snapshot'] = not record.source_complete
                entry['replay_output_chars'] = len(record.text)

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
        with self._control_lock:
            os.makedirs(os.path.dirname(self.control_file), exist_ok=True)
            temp_path = f'{self.control_file}.tmp-{os.getpid()}-{int(time.time() * 1000)}'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(control, f, ensure_ascii=False, indent=2)
                f.write('\n')
            os.replace(temp_path, self.control_file)


# Keep imports used by existing deployments and extensions working.
ClaudeReplayStore = ReplayStore
