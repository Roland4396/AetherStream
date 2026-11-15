"""Caller-created final-reply envelopes: offline saved-log replay only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from aetherstream.api import app as app_module
from aetherstream.features.claude_replay import ReplayStore
from aetherstream.features.replay import ReplayPreparationError, ReplayService, parse_replay_record


TEXT = '<build>离线核验</build>\n<content>完整正文🙂，引号"与反斜杠\\。</content>\n<UpdateVariable>{"key":"value"}</UpdateVariable>'
MODEL = '[Rim]gemini-4-argon'


def event(delta=None, finish=None, usage=None):
    return {'id': 'offline', 'model': MODEL, 'choices': [
        {'index': 0, 'delta': delta or {}, 'finish_reason': finish}], 'usage': usage}


def call(name=None, args='', index=0):
    function = {'arguments': args}
    if name is not None:
        function['name'] = name
    return {'index': index, 'type': 'function', 'function': function}


def wire(events, marker='[FINISH_REASON: tool_calls]'):
    return ''.join('data: ' + json.dumps(e, ensure_ascii=False) + '\n\n' for e in events) + marker + '\n'


def tool_events(text=TEXT, name='actual_content', parts=None, finish='tool_calls'):
    args = json.dumps({'content': text}, ensure_ascii=True)
    width = parts or len(args)
    values = [event({'role': 'assistant', 'content': '', 'tool_calls': [call(name)]})]
    values += [event({'content': '', 'tool_calls': [call(args=args[i:i+width])]})
               for i in range(0, len(args), width)]
    values += [event(usage={'prompt_tokens': 7, 'completion_tokens': 11, 'total_tokens': 18}), event(finish=finish)]
    return values


def completion(text=TEXT, name='actual_content'):
    return {'model': MODEL, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '',
            'tool_calls': [call(name, json.dumps({'content': text}))]}, 'finish_reason': 'tool_calls'}]}


class CallerToolReplayParserTests(unittest.TestCase):
    def test_exact_observed_empty_name_then_full_arguments_shape(self):
        record = parse_replay_record(wire(tool_events()), fallback_model=MODEL)
        self.assertEqual(record.text, TEXT)
        self.assertEqual(''.join(d.content for d in record.deltas), TEXT)
        self.assertEqual(record.finish_reason, 'stop')
        self.assertTrue(record.source_complete)
        self.assertEqual(record.usage['total_tokens'], 18)

    def test_multi_event_arguments_and_unicode_escapes_are_lossless(self):
        for width in (1, 2, 7, 31, 99999):
            with self.subTest(width=width):
                record = parse_replay_record(wire(tool_events(parts=width)), fallback_model=MODEL)
                self.assertEqual(record.text, TEXT)
                self.assertEqual(record.finish_reason, 'stop')

    def test_split_function_name_is_recognized(self):
        events = [event({'tool_calls': [call('actual_')]}),
                  event({'tool_calls': [call('content', json.dumps({'content': TEXT}))]}),
                  event(finish='tool_calls')]
        self.assertEqual(parse_replay_record(wire(events), fallback_model=MODEL).text, TEXT)

    def test_namespaced_exact_name_is_recognized(self):
        self.assertEqual(parse_replay_record(wire(tool_events(name='functions:actual_content')),
                                           fallback_model=MODEL).text, TEXT)

    def test_unmarked_json_completion_uses_same_contract(self):
        record = parse_replay_record(json.dumps(completion()), fallback_model=MODEL)
        self.assertEqual((record.text, record.finish_reason), (TEXT, 'stop'))
        self.assertEqual(record.source_format, 'openai_json')

    def test_nested_log_headers_do_not_hide_the_tool(self):
        header = 'Time: offline\nModel: test\nTraceId: offline\n' + '=' * 50 + '\n'
        self.assertEqual(parse_replay_record(header + header + wire(tool_events()),
                                           fallback_model=MODEL).text, TEXT)

    def test_empty_output_diagnostics_never_override_recovered_text(self):
        for snapshot in ('[empty response]', '[empty replay response]', 'unrelated old output'):
            with self.subTest(snapshot=snapshot):
                record = parse_replay_record(wire(tool_events()), fallback_model=MODEL,
                                             fallback_output_text=snapshot)
                self.assertEqual(record.text, TEXT)

    def test_non_envelope_tools_are_not_interpreted_as_final_text(self):
        for name in ('search', 'actual_content_extra', 'other_actual_content', 'ACTUAL_CONTENT'):
            with self.subTest(name=name), self.assertRaisesRegex(ReplayPreparationError, 'unsupported tool call'):
                parse_replay_record(wire(tool_events(name=name)), fallback_model=MODEL)

    def test_mixed_unrelated_tool_is_not_silently_discarded(self):
        events = tool_events()
        events.insert(-1, event({'tool_calls': [call('search', '{"content":"not prose"}', 1)]}))
        with self.assertRaisesRegex(ReplayPreparationError, 'unsupported tool call'):
            parse_replay_record(wire(events), fallback_model=MODEL)

    def test_plain_reply_containing_a_tool_example_stays_plain(self):
        text = json.dumps(completion())
        record = parse_replay_record(wire([event({'content': text}), event(finish='stop')]), fallback_model=MODEL)
        self.assertEqual(record.text, text)

    def test_invalid_arguments_are_readable_preparation_errors(self):
        for args in ('{"content":"unfinished', '{"other":"no content"}', '{"content":123}'):
            with self.subTest(args=args), self.assertRaisesRegex(ReplayPreparationError, 'invalid output-tool arguments'):
                parse_replay_record(wire([event({'tool_calls': [call('actual_content', args)]}),
                                          event(finish='tool_calls')]), fallback_model=MODEL)

    def test_invalid_json_completion_is_a_preparation_error(self):
        payload = completion()
        payload['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{broken'
        with self.assertRaises(ReplayPreparationError):
            parse_replay_record(json.dumps(payload), fallback_model=MODEL)

    def test_multiple_final_reply_calls_are_rejected(self):
        events = tool_events()
        events.insert(-1, event({'tool_calls': [call('actual_content', '{"content":"duplicate"}', 1)]}))
        with self.assertRaises(ReplayPreparationError):
            parse_replay_record(wire(events), fallback_model=MODEL)

    def test_cancelled_partial_snapshot_does_not_fabricate_completion(self):
        raw = wire([event({'tool_calls': [call('actual_content', '{"content":"partial')]})],
                   '[CANCELLED: offline]')
        record = parse_replay_record(raw, fallback_model=MODEL)
        self.assertEqual(record.text, 'partial')
        self.assertFalse(record.source_complete)

    def test_length_finish_keeps_its_partial_output_and_reason(self):
        raw = wire([event({'tool_calls': [call('actual_content', '{"content":"partial')]}),
                    event(finish='length')], '[FINISH_REASON: length]')
        record = parse_replay_record(raw, fallback_model=MODEL)
        self.assertEqual((record.text, record.finish_reason), ('partial', 'length'))


class CallerToolReplayRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ReplayStore(log_dir=str(self.root), control_file=str(self.root/'control.json'), log=lambda _: None)
        (self.root/'04_raw_sse.txt').write_text(wire(tool_events(parts=7)))
        (self.root/'04_output.txt').write_text('Time: offline\n' + '='*50 + '\n[empty response]')
        self.store.write_control({'enabled': True, 'mode': 'once', 'match_request': False,
                                  'raw_sse_path': '04_raw_sse.txt'})
        self.saved = []
        self.service = ReplayService(store=self.store, log=lambda _: None,
            save_request_log=lambda *a, **k: self.saved.append((a,k)),
            release_caller=app_module.release_active_stream_caller)
        self.client = TestClient(app_module.app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def request(self, stream):
        return {'model': MODEL, 'stream': stream, 'messages': [{'role': 'user', 'content': 'offline only'}],
                'tool_choice': 'required', 'tools': [{'type': 'function', 'function': {'name': 'actual_content'}}]}

    def test_stream_replay_returns_all_text_and_one_done_without_inference(self):
        with patch.object(app_module.chat_route_dependencies, 'replay_service', self.service):
            response = self.client.post('/v1/chat/completions', json=self.request(True))
        self.assertEqual(response.status_code, 200)
        chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
        text = ''.join(c.get('delta', {}).get('content', '') for d in chunks for c in d.get('choices', []))
        self.assertEqual(text, TEXT)
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(response.text.count('data: [DONE]'), 1)
        self.assertFalse(self.store.read_control()['enabled'])
        self.assertEqual(self.saved[0][0][2], TEXT)
        self.assertEqual(len(app_module.active_stream_registry), 0)

    def test_json_replay_returns_same_text_without_inference(self):
        with patch.object(app_module.chat_route_dependencies, 'replay_service', self.service):
            response = self.client.post('/v1/chat/completions', json=self.request(False))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['choices'][0]['message']['content'], TEXT)
        self.assertFalse(self.store.read_control()['enabled'])

    def test_list_preview_reports_unmarked_source_as_usable(self):
        entries = self.store.list_entries()
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]['replay_usable'])

    def test_invalid_tool_fails_before_consuming_once_or_calling_provider(self):
        (self.root/'04_raw_sse.txt').write_text(wire(tool_events(name='search')))
        with patch.object(app_module.chat_route_dependencies, 'replay_service', self.service):
            response = self.client.post('/v1/chat/completions', json=self.request(True))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error']['type'], 'replay_invalid')
        self.assertTrue(self.store.read_control()['enabled'])
        self.assertEqual(self.saved, [])
