"""03-input/S喵 output envelope: mocked providers only, no browser state."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from aetherstream.features.output_tool import (
    OutputToolStreamDecoder, StringFieldDecoder, prepare_output_tool,
    record_marker, unwrap_completion, validate_settings,
)
from aetherstream.features.replay import parse_replay_record
from aetherstream.routing.policy_config import enrich_selection, dedupe_identity, global_policy, RegistryError
from aetherstream.api.channel_gateway import prepare_adapter
from aetherstream.upstreams import openai_chat_completions as chat
import test_channel_gateway as gateway_fixture
from test_channel_gateway import ControlledStream, plan, policy, sse, REAL_CLIENT


CONFIG = json.loads((Path(__file__).parent / 'fixtures/output-tool-from-03.json').read_text())
TEXT = '<dm_set>位置核验</dm_set>\n<act>原有行动：“先等消息。”</act>\n<recall>AM01</recall> 🙂 \\ "'
MODEL = 'gemini-3.8-flash-high'


def result(text=TEXT, *, other=False):
    calls = [{'id': 'output', 'type': 'function', 'function': {
        'name': 'actual_content', 'arguments': json.dumps({'content': text}, ensure_ascii=False)}}]
    if other:
        calls.append({'id': 'original', 'type': 'function', 'function': {'name': 'sum', 'arguments': '{"a":1}'}})
    return {'id': 'reply', 'object': 'chat.completion', 'model': MODEL,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'tool_calls': calls},
                         'finish_reason': 'tool_calls'}], 'usage': {'total_tokens': 17}}


def tool_events(text=TEXT, *, parts=7, complete=True):
    args = json.dumps({'content': text}, ensure_ascii=True)
    values = []
    for i in range(0, len(args), parts):
        function = {'arguments': args[i:i + parts]}
        if i == 0:
            function['name'] = 'actual_content'
        values.append({'id': 'reply', 'model': MODEL, 'choices': [{'index': 0, 'delta': {
            'tool_calls': [{'index': 0, 'id': 'output', 'type': 'function', 'function': function}]},
            'finish_reason': None}]})
    if complete:
        values.append({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
                       'usage': {'total_tokens': 17}})
    return values


def stream_text(body):
    text = []
    for line in body.decode().splitlines():
        if line.startswith('data: {'):
            for c in json.loads(line[6:]).get('choices', []):
                text.append(c.get('delta', {}).get('content', ''))
    return ''.join(text)


class EnvelopeTests(unittest.TestCase):
    def test_definition_is_the_exact_user_supplied_03_tool(self):
        f = CONFIG['tool']['function']
        self.assertEqual(f['name'], 'actual_content')
        self.assertEqual(f['description'], '图灵，给我把所有输出都吐到这里')
        self.assertEqual(f['parameters'], {'type': 'object', 'properties': {
            'content': {'type': 'string', 'description': 'The complete final reply shown to the user.'}},
            'required': ['content']})

    def test_request_only_adds_the_existing_envelope_and_control(self):
        messages = [{'role': 'system', 'content': 'Original rules'},
                    {'role': 'user', 'content': '<format>Original task</format>'},
                    {'role': 'assistant', 'content': 'Original prefill'}]
        body = {'model': MODEL, 'messages': copy.deepcopy(messages), 'stream': False,
                'temperature': 0.7, 'max_tokens': 12345, 'stop': ['original']}
        original = copy.deepcopy(body)
        active = prepare_output_tool(body, CONFIG, log=lambda _: None)
        self.assertEqual(active, CONFIG)
        self.assertEqual(body['tools'], [CONFIG['tool']])
        self.assertEqual(body['tool_choice'], 'auto')
        self.assertEqual(body['messages'], [messages[0], {'role': 'system', 'content': CONFIG['control_prompt']}, *messages[1:]])
        self.assertEqual({k: body[k] for k in original if k != 'messages'},
                         {k: original[k] for k in original if k != 'messages'})
        once = copy.deepcopy(body)
        self.assertIsNone(prepare_output_tool(body, CONFIG, log=lambda _: None))
        self.assertEqual(body, once)

    def test_disabled_explicit_tool_choices_and_caller_owned_tools_are_untouched(self):
        for choice in ['none', 'required', {'type': 'function', 'function': {'name': 'sum'}}, 'invalid', []]:
            body = {'messages': [{'role': 'user', 'content': TEXT}], 'tool_choice': choice}
            before = copy.deepcopy(body)
            self.assertIsNone(prepare_output_tool(body, CONFIG, log=lambda _: None))
            self.assertEqual(body, before)
        for cfg, tools in [({'enabled': False}, []), (CONFIG, [CONFIG['tool']])]:
            body = {'messages': [{'role': 'user', 'content': TEXT}], 'tools': copy.deepcopy(tools)}
            before = copy.deepcopy(body)
            self.assertIsNone(prepare_output_tool(body, cfg, log=lambda _: None))
            self.assertEqual(body, before)

    def test_other_tools_and_assistant_prefill_survive(self):
        original_tool = {'type': 'function', 'function': {'name': 'sum'}}
        body = {'messages': [{'role': 'assistant', 'content': 'prefill'}], 'tools': [original_tool]}
        prepare_output_tool(body, CONFIG, log=lambda _: None)
        self.assertEqual(body['tools'], [original_tool, CONFIG['tool']])
        self.assertEqual(body['messages'][-1], {'role': 'user', 'content': CONFIG['control_prompt']})

    def test_invalid_settings_are_rejected(self):
        for changes in [{'tool': None}, {'field': 'absent'}, {'enabled': 'true'},
                        {'anchors': ['']}, {'control_prompt': ''}, {'unexpected': True}]:
            c = copy.deepcopy(CONFIG)
            c.update(changes)
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                validate_settings(c)

    def test_only_configured_38_flash_models_match(self):
        config = {**CONFIG, 'match': {'regex': r'gemini-3[.]8-flash(?:$|[-_/])'}, 'transports': ['chat']}
        f = {'channel_policies': {'schema': 1, 'channels': {}, 'defaults': {}, 'profiles': {}, 'model_rules': []},
             'output_tool': config}
        previous = copy.deepcopy(f)
        previous.pop('output_tool')
        self.assertEqual(global_policy(lambda key: f.get(key)), global_policy(lambda key: previous.get(key)))
        for name, expected in [('gemini-3.8-flash', True), ('free/gemini-3.8-flash-high', True),
                               ('[YS-1]抗截断/gemini-3.8-flash', True), ('Gemini-3.8-Flash-low', True),
                               ('假流式-gemini-3.8-flash', True), ('gemini-3.80-flash', False),
                               ('gemini-3.8-flashier', False), ('gemini-4-argon-high', False), ('glm-5.2', False)]:
            selected = {'model': name, 'upstream_model': name, 'channel': 'any', 'route': 'any', 'transport': 'chat'}
            p = enrich_selection(lambda key: f.get(key), selected)['policy']
            self.assertEqual('output_tool' in p['runtime'], expected, name)

    def test_every_json_escape_and_every_two_part_boundary_round_trips(self):
        for ascii_only in (False, True):
            raw = json.dumps({'content': TEXT + '\b\f\r\t\n'}, ensure_ascii=ascii_only)
            expected = TEXT + '\b\f\r\t\n'
            for boundary in range(len(raw) + 1):
                decoder = StringFieldDecoder('content')
                self.assertEqual(decoder.feed(raw[:boundary]) + decoder.feed(raw[boundary:]), expected)
                decoder.finish()
            decoder = StringFieldDecoder('content')
            self.assertEqual(''.join(decoder.feed(ch) for ch in raw), expected)
            decoder.finish()

    def test_field_can_follow_another_json_value(self):
        raw = json.dumps({'unused': {'list': ['content', 'not our text']}, 'number': 1234.56, 'content': TEXT})
        decoder = StringFieldDecoder('content')
        self.assertEqual(''.join(decoder.feed(ch) for ch in raw), TEXT)
        decoder.finish()

    def test_malformed_argument_does_not_become_successful_text(self):
        decoder = StringFieldDecoder('content')
        decoder.feed('{"content":"partial')
        with self.assertRaises(ValueError) as error:
            decoder.finish()
        self.assertNotIsInstance(error.exception, json.JSONDecodeError)
        with self.assertRaises(ValueError):
            unwrap_completion(result(), {'tool': CONFIG['tool'], 'field': 'wrong'})

    def test_stream_and_json_outputs_match_exactly_and_only_our_call_is_removed(self):
        for parts in [1, 2, 7, 9999]:
            decoder = OutputToolStreamDecoder(CONFIG)
            chunks = [decoder.feed(e) for e in tool_events(parts=parts)]
            self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in chunks), TEXT)
            self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')
            self.assertFalse(any('tool_calls' in c['choices'][0]['delta'] for c in chunks))
        raw = result(other=True)
        before = copy.deepcopy(raw)
        out = unwrap_completion(raw, CONFIG)
        self.assertEqual(raw, before)
        self.assertEqual(out['choices'][0]['message']['content'], TEXT)
        self.assertEqual(out['choices'][0]['message']['tool_calls'], [raw['choices'][0]['message']['tool_calls'][1]])
        self.assertEqual(out['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(out['usage'], raw['usage'])

    def test_plain_text_and_unrelated_function_payloads_are_not_decoded(self):
        e = {'choices': [{'index': 0, 'delta': {'content': TEXT}, 'finish_reason': 'stop'}]}
        self.assertEqual(OutputToolStreamDecoder(CONFIG).feed(e), e)

    def test_token_limit_returns_partial_text_without_fabricating_a_stop(self):
        data = result()
        data['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"content":"partial'
        data['choices'][0]['finish_reason'] = 'length'
        out = unwrap_completion(data, CONFIG)
        self.assertEqual(out['choices'][0]['message']['content'], 'partial')
        self.assertEqual(out['choices'][0]['finish_reason'], 'length')
        e = {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 1, 'function': {
            'name': 'sum', 'arguments': '{"content":"not display text"}'}}]}, 'finish_reason': 'tool_calls'}]}
        self.assertEqual(OutputToolStreamDecoder(CONFIG).feed(e), e)

    def test_call_and_choice_states_never_mix(self):
        decoder = OutputToolStreamDecoder(CONFIG)
        merged = []
        for text, index in [('first', 0), ('second', 1)]:
            e = tool_events(text, parts=9999)[0]
            e['choices'][0]['index'] = index
            merged.append(decoder.feed(e)['choices'][0]['delta']['content'])
        self.assertEqual(merged, ['first', 'second'])

    def test_stream_preserves_real_tools_alongside_the_output_envelope(self):
        decoder = OutputToolStreamDecoder(CONFIG)
        own = tool_events('visible', parts=9999)[0]
        real = {'index': 1, 'id': 'original', 'type': 'function', 'function': {'name': 'sum', 'arguments': '{"a":1}'}}
        own['choices'][0]['delta']['tool_calls'].append(real)
        out = decoder.feed(own)
        self.assertEqual(out['choices'][0]['delta']['content'], 'visible')
        self.assertEqual(out['choices'][0]['delta']['tool_calls'], [real])
        finish = decoder.feed(tool_events('visible')[-1])
        self.assertEqual(finish['choices'][0]['finish_reason'], 'tool_calls')

    def test_late_and_split_function_names_are_buffered_without_argument_loss(self):
        decoder = OutputToolStreamDecoder(CONFIG)
        def event(f):
            return {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': f}]}, 'finish_reason': None}]}
        first = decoder.feed(event({'arguments': '{"content":"same'}))
        second = decoder.feed(event({'name': 'actual_'}))
        last = decoder.feed(event({'name': 'content', 'arguments': ' text"}'}))
        self.assertNotIn('tool_calls', first['choices'][0]['delta'])
        self.assertNotIn('tool_calls', second['choices'][0]['delta'])
        self.assertEqual(last['choices'][0]['delta']['content'], 'same text')

    def test_saved_raw_tool_stream_and_json_replay_as_the_same_visible_reply(self):
        raw = record_marker(CONFIG) + ''.join('data: ' + json.dumps(e) + '\n\n' for e in tool_events()) + 'data: [DONE]\n\n'
        r = parse_replay_record(raw, fallback_model=MODEL, fallback_output_text=TEXT)
        self.assertEqual(r.text, TEXT)
        self.assertEqual(r.finish_reason, 'stop')
        r = parse_replay_record(record_marker(CONFIG) + json.dumps(result()), fallback_model=MODEL, fallback_output_text=TEXT)
        self.assertEqual((r.text, r.finish_reason), (TEXT, 'stop'))

    def test_replay_uses_delivered_snapshot_when_proxy_stopped_before_argument_closed(self):
        raw = record_marker(CONFIG) + 'data: ' + json.dumps(tool_events('HEAD<CUT>TAIL', parts=9999, complete=False)[0]) + '\n[EARLY_STOP by proxy]\n'
        r = parse_replay_record(raw, fallback_model=MODEL, fallback_output_text='HEAD')
        self.assertEqual(r.text, 'HEAD')

    def test_envelope_does_not_change_exact_request_cache_identity(self):
        p = plan(settings=policy(early_stop={'enabled': False}))
        p['upstream_model'] = MODEL
        wrapped = copy.deepcopy(p)
        wrapped['policy']['runtime']['output_tool'] = CONFIG
        data = {'model': MODEL, 'messages': [{'role': 'user', 'content': TEXT}], 'stream': False}
        deps = SimpleNamespace(log=lambda _: None, build_chat_completions_upstream_deps=lambda: None)
        old = prepare_adapter(data, p, deps, trace_id='offline', upstream_stream=False)
        new = prepare_adapter(data, wrapped, deps, trace_id='offline', upstream_stream=False)
        self.assertEqual(dedupe_identity(p, old.payload, old.headers)['key'],
                         dedupe_identity(new.identity_plan, new.identity_payload, new.headers)['key'])


class GatewayEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_client_protocols_keep_their_existing_response_shape(self):
        h = gateway_fixture.ChannelGatewayTests()
        for protocol in ['anthropic', 'responses']:
            for stream in [False, True]:
                data = {'model': MODEL, 'stream': stream}
                if protocol == 'anthropic':
                    data.update(max_tokens=1000, messages=[{'role': 'user', 'content': 'continue'}])
                else:
                    data['input'] = 'continue'
                resolved = plan(model=MODEL, settings=policy(early_stop={'enabled': False}, runtime={'output_tool': CONFIG}))
                def provider(req):
                    if json.loads(req.content)['stream']:
                        return httpx.Response(200, stream=ControlledStream([sse(e) for e in tool_events()] + [b'data: [DONE]\n\n']))
                    return httpx.Response(200, json=result())
                response, body, sent, _ = await h.invoke(resolved, data, provider, downstream=protocol)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(sent), 1)
                self.assertNotIn(b'actual_content', body)
                if not stream:
                    native = json.loads(body)
                    text = (native['content'][0]['text'] if protocol == 'anthropic'
                            else native['output'][0]['content'][0]['text'])
                    self.assertEqual(text, TEXT)
    async def test_all_four_client_and_upstream_modes_return_text_with_one_request(self):
        h = gateway_fixture.ChannelGatewayTests()
        for mode in ['auto', 'stream', 'json']:
            for stream in [False, True]:
                resolved = plan(model=MODEL, settings=policy(streaming={'force_upstream': mode},
                    early_stop={'enabled': False}, runtime={'output_tool': CONFIG}))
                def provider(req):
                    body = json.loads(req.content)
                    if body['stream']:
                        chunks = [sse(e) for e in tool_events()] + [b'data: [DONE]\n\n']
                        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=ControlledStream(chunks))
                    return httpx.Response(200, json=result())
                response, body, sent, _ = await h.invoke(resolved, {
                    'model': MODEL, 'messages': [{'role': 'user', 'content': 'continue'}], 'stream': stream}, provider)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(sent), 1)
                outgoing = json.loads(sent[0].content)
                self.assertEqual(outgoing['tools'], [CONFIG['tool']])
                self.assertEqual(outgoing['model'], MODEL)
                self.assertEqual(outgoing['stream'], stream if mode == 'auto' else mode == 'stream')
                text = stream_text(body) if stream else json.loads(body)['choices'][0]['message']['content']
                self.assertEqual(text, TEXT)
                self.assertNotIn(b'"name": "actual_content"', body)

    async def test_decoded_text_still_triggers_existing_actual_upstream_disconnect(self):
        h = gateway_fixture.ChannelGatewayTests()
        upstream = ControlledStream([sse(tool_events('HEAD<CUT>TAIL', parts=9999, complete=False)[0])], wait_after=True)
        resolved = plan(model=MODEL, settings=policy(runtime={'output_tool': CONFIG}))
        response, body, sent, _ = await h.invoke(resolved, {
            'model': MODEL, 'messages': [{'role': 'user', 'content': 'continue'}], 'stream': True},
            lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=upstream), asgi=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(stream_text(body), 'HEAD')
        self.assertEqual(len(sent), 1)
        self.assertTrue(upstream.closed)

    async def test_plain_fallback_and_upstream_errors_never_trigger_another_request(self):
        h = gateway_fixture.ChannelGatewayTests()
        resolved = plan(model=MODEL, settings=policy(runtime={'output_tool': CONFIG}))
        for status, payload in [(200, {'choices': [{'message': {'content': TEXT}, 'finish_reason': 'stop'}]}),
                                (429, {'error': {'message': 'offline quota failure'}})]:
            response, body, sent, _ = await h.invoke(resolved, {
                'model': MODEL, 'messages': [{'role': 'user', 'content': 'continue'}], 'stream': False},
                lambda _: httpx.Response(status, json=payload))
            self.assertEqual(len(sent), 1)
            if status == 200:
                self.assertEqual(json.loads(body)['choices'][0]['message']['content'], TEXT)
            else:
                self.assertNotEqual(response.status_code, 200)

    async def test_downstream_cancellation_closes_the_real_mock_connection(self):
        ready, closed = asyncio.Event(), asyncio.Event()
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield sse(tool_events('visible', parts=9999, complete=False)[0])
                ready.set()
                await asyncio.Event().wait()
            async def aclose(self):
                closed.set()
        client = REAL_CLIENT(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Body())))
        deps = chat.ChatCompletionsUpstreamDeps(log=lambda _: None, save_request_log=lambda *a, **k: None,
            build_openai_sse_error=lambda *a, **k: b'ERROR', has_stop_tag=lambda _: False,
            find_stop_tag=lambda _: -1, fmt_ms=lambda *_: '0ms', release_caller=lambda *_: None)
        async def consume():
            async for _ in chat.forward_chat_completions_stream(url='http://offline.invalid', request_data={},
                    headers={}, timeout=httpx.Timeout(10), max_raw_sse_bytes=100000, deps=deps,
                    model='test', output_tool=CONFIG):
                pass
        with patch.object(chat.httpx, 'AsyncClient', return_value=client):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(ready.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(closed.is_set())
        self.assertTrue(client.is_closed)
