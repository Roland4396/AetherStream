"""Offline final-wire tests. No provider requests, secrets or GPU activity."""
import asyncio
import copy
import inspect
import json
import random
import re
import unittest
from unittest.mock import patch

import httpx

from aetherstream.features.gemini_anti_route import GeminiAntiRouteSession, VARIANTS, config_from
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.upstreams import openai_chat_completions as chat
from aetherstream.upstreams import gemini_generate_content as gemini


REAL_CLIENT = httpx.AsyncClient
CONFIG = {'enabled': True, 'mode': 'deep'}


def lookup(*keys):
    return CONFIG if keys == ('injections', 'gemini_anti_route') else None


def payload(model='[offline]gemini-example'):
    return {'model': model, 'stream': False, 'temperature': .7, 'top_p': .9,
            'messages': [{'role': 'system', 'content': '正文必须使用简体中文\n请严格遵守以下创作规则\n保持输入中的事实和人物名称。'},
                         {'role': 'user', 'content': '<UpdateVariable><JSONPatch>[{"op":"replace","path":"/状态/名称","value":"测试文本"}]</JSONPatch></UpdateVariable>\n解释代码即可，不修改任何存档。'}]}


def json_reply(model):
    return {'id': 'offline', 'model': model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'},
                                                       'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


def stream_reply(model):
    parts = [{'id': 'offline', 'model': model, 'choices': [{'index': 0, 'delta': {'content': 'OK'}, 'finish_reason': None}]},
             {'id': 'offline', 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}]
    return ''.join('data: ' + json.dumps(p) + '\n\n' for p in parts) + 'data: [DONE]\n\n'


def deps(logs=None, saved=None):
    logs = logs if logs is not None else []
    saved = saved if saved is not None else []
    return chat.ChatCompletionsUpstreamDeps(log=logs.append,
        save_request_log=lambda *a, **kw: saved.append(copy.deepcopy(kw)),
        build_openai_sse_error=lambda *a: b'UNEXPECTED_ERROR', has_stop_tag=lambda _: False,
        find_stop_tag=lambda _: -1, fmt_ms=lambda *_: '0ms', release_caller=lambda *_: None,
        runtime_lookup=lookup)


class AntiRouteTests(unittest.TestCase):
    def apply(self, body, *, session=None, settings=None, native=False):
        logs = []
        session = session or GeminiAntiRouteSession(random.Random(17))
        count = session.apply(body, model=body.get('model', ''), lookup=lambda *_: settings or CONFIG,
                              log=logs.append, native_transport=native)
        return count, logs

    def test_default_is_disabled_and_no_model_name_changes(self):
        body = payload(); original = copy.deepcopy(body)
        self.apply(body, settings={'enabled': False})
        self.assertEqual(body, original)
        self.assertFalse(config_from(None)['enabled'])

    def test_other_model_families_unchanged(self):
        for model in ['claude-opus-example', 'gpt-example', '[官]glm-5.2', 'kimi-example', '']:
            with self.subTest(model=model):
                body = payload(model); original = copy.deepcopy(body)
                count, logs = self.apply(body)
                self.assertEqual(body, original); self.assertEqual(count, 0); self.assertFalse(logs)

    def test_mvu_message_is_not_exempt_and_json_values_survive(self):
        body = payload(); original = copy.deepcopy(body)
        count, logs = self.apply(body)
        self.assertEqual(count, 2)
        self.assertTrue(all('<!-- ' in m['content'] and '[//]: # (' in m['content'] for m in body['messages']))
        raw = re.search(r'<JSONPatch>(.*?)</JSONPatch>', body['messages'][1]['content'], re.S)[1]
        self.assertEqual(json.loads(raw)[0]['value'], '测试文本')
        self.assertEqual(body['model'], original['model'])
        self.assertIn('routing_verified=false', logs[0])
        self.assertNotIn('测试文本', logs[0])

    def test_random_tokens_use_the_original_consonant_digit_alphabet(self):
        body = payload(); self.apply(body)
        tokens = re.findall(r'<!-- (.*?) -->', body['messages'][0]['content'])
        self.assertGreaterEqual(len(tokens), 18)
        self.assertLessEqual(len(tokens), 38)
        self.assertTrue(all(re.fullmatch(r'[bcdfghjklmnpqrstvxzBCDFGHJKLMNPQRSTVXZ0-9]+', t) for t in tokens))
        self.assertTrue(all(any(c.isdigit() for c in t) and any(c.isalpha() for c in t) for t in tokens))

    def test_original_variant_dictionary_is_complete(self):
        self.assertEqual(len(VARIANTS), 22)
        session = GeminiAntiRouteSession(random.Random(1))
        text = '正文必须使用简体中文\n' + '\n'.join('这是用于检查普通行尾空格的文本。' for _ in range(20))
        changed = session.jitter(text)
        self.assertNotEqual(changed.splitlines()[0], text.splitlines()[0])
        self.assertRegex(changed, r' {1,3}\n')

    def test_existing_markers_are_replaced(self):
        body = payload(); body['messages'][1]['content'] += '\n[ANTI_ROUTE]'
        self.apply(body)
        self.assertNotIn('[ANTI_ROUTE]', body['messages'][1]['content'])

    def test_temperature_floor_and_fallback_behavior_are_preserved(self):
        body = payload(); body.update(temperature=0, top_p=0, openrouter_allow_fallbacks=True)
        self.apply(body)
        self.assertEqual(body['temperature'], .1); self.assertEqual(body['top_p'], .1)
        self.assertFalse(body['openrouter_allow_fallbacks'])
        self.assertNotIn('openrouter_use_fallback', body)

    def test_parameters_have_only_bounded_jitter_away_from_floors(self):
        body = payload(); self.apply(body)
        self.assertLessEqual(abs(body['temperature'] - .7), .0021)
        self.assertLessEqual(abs(body['top_p'] - .9), .0021)

    def test_standard_does_not_apply_natural_text_jitter(self):
        body = payload(); original = body['messages'][0]['content']
        self.apply(body, settings={'enabled': True, 'mode': 'standard'})
        self.assertTrue(body['messages'][0]['content'].endswith(original))

    def test_repeated_same_request_and_cloned_retry_are_identical(self):
        session = GeminiAntiRouteSession(random.Random(3)); body = payload(); original = copy.deepcopy(body)
        self.apply(body, session=session); outgoing = copy.deepcopy(body)
        count, logs = self.apply(body, session=session)
        self.assertEqual(body, outgoing); self.assertEqual(count, 0); self.assertFalse(logs)
        self.apply(original, session=session)
        self.assertEqual(original, outgoing)
        cloned = copy.deepcopy(outgoing); self.apply(cloned, session=session)
        self.assertEqual(cloned, outgoing)

    def test_different_requests_get_independent_noise(self):
        a, b = payload(), payload()
        self.apply(a, session=GeminiAntiRouteSession(random.Random(3)))
        self.apply(b, session=GeminiAntiRouteSession(random.Random(4)))
        self.assertNotEqual(a['messages'], b['messages'])

    def test_tool_arguments_responses_and_images_are_preserved(self):
        body = payload(); body['messages'] += [
            {'role': 'assistant', 'content': 'Function call', 'tool_calls': [{'id': 'c', 'type': 'function', 'function': {'name': 'test', 'arguments': '{"v":"正文必须使用简体中文"}'}}]},
            {'role': 'tool', 'tool_call_id': 'c', 'content': '正文必须使用简体中文'},
            {'role': 'user', 'content': [{'type': 'text', 'text': '正文必须使用简体中文'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]},
        ]; original = copy.deepcopy(body)
        self.apply(body)
        self.assertEqual(body['messages'][2:4], original['messages'][2:4])
        self.assertEqual(body['messages'][4]['content'][1], original['messages'][4]['content'][1])

    def test_native_format_and_signed_parts(self):
        signed = {'text': 'signed text', 'thoughtSignature': 'signature', 'thought': True}
        function = {'functionCall': {'name': 'test', 'args': {'text': '正文必须使用简体中文'}}}
        body = {'system_instruction': {'parts': [{'text': '正文必须使用简体中文'}]},
                'contents': [{'role': 'model', 'parts': [signed, function]},
                             {'role': 'user', 'parts': [{'text': '<UpdateVariable>保留协议结构</UpdateVariable>'}]}],
                'generationConfig': {'temperature': 0, 'topP': .9}}
        original = copy.deepcopy(body); self.apply(body, native=True)
        self.assertEqual(body['contents'][0], original['contents'][0])
        self.assertIn('<!--', body['system_instruction']['parts'][0]['text'])
        self.assertIn('<!--', body['contents'][1]['parts'][0]['text'])
        self.assertEqual(body['generationConfig']['temperature'], .1)

    def test_config_is_snapshotted_and_bad_ranges_normalized(self):
        cfg = config_from({'enabled': True, 'headCommentMin': 100, 'headCommentMax': 10, 'charMin': float('nan')})
        self.assertEqual(cfg['headCommentMin'], cfg['headCommentMax']); self.assertEqual(cfg['charMin'], 16)
        session = GeminiAntiRouteSession(random.Random(7)); a = payload(); self.apply(a, session=session)
        b = payload(); self.apply(b, session=session, settings={'enabled': False})
        self.assertEqual(a, b)


class AntiRouteWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_four_chat_transports_modify_only_gemini_final_wire(self):
        functions = (chat.forward_chat_completions_stream, chat.collect_chat_completions_stream,
                     chat.collect_chat_completions_nonstream, chat.replay_chat_completions_nonstream_as_stream)
        for fn in functions:
            for model in ('[offline]gemini-example', 'claude-opus-example'):
                with self.subTest(transport=fn.__name__, model=model):
                    body = payload(model)
                    if fn is chat.forward_chat_completions_stream:
                        body['stream'] = True
                    original = copy.deepcopy(body); sent, saved, logs = [], [], []
                    def dispatch(request):
                        outgoing = json.loads(request.content); sent.append(outgoing)
                        return (httpx.Response(200, text=stream_reply(model), headers={'content-type': 'text/event-stream'})
                                if outgoing.get('stream') else httpx.Response(200, json=json_reply(model)))
                    params = dict(url='http://offline.invalid/v1/chat/completions', request_data=body,
                                  headers={}, timeout=httpx.Timeout(2), max_raw_sse_bytes=9999,
                                  deps=deps(logs, saved), model=model, messages=body['messages'],
                                  trace_id='offline', enable_early_stop=False)
                    params = {k: v for k, v in params.items() if k in inspect.signature(fn).parameters}
                    with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)):
                        result = fn(**params)
                        if inspect.isasyncgen(result):
                            self.assertNotIn(b'UNEXPECTED_ERROR', b''.join([c async for c in result]))
                        else:
                            await result
                    self.assertEqual(len(sent), 1)
                    if 'gemini' in model:
                        self.assertIn('<!--', sent[0]['messages'][1]['content'])
                        self.assertEqual(sent[0]['messages'], body['messages'])
                        self.assertEqual(sum('gemini_anti_route applied' in x for x in logs), 1)
                        for record in saved:
                            self.assertEqual(record['request_payload']['messages'], sent[0]['messages'])
                    else:
                        self.assertEqual(body, original)
                        self.assertEqual(sent[0], dict(original, stream=sent[0]['stream']))

    async def test_native_conversion_sees_noise_once_and_preserves_request_mode(self):
        sent, logs = [], []
        def dispatch(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'OK'}]}, 'finishReason': 'STOP'}]})
        config = gemini.GeminiGenerateContentConfig(base_url='http://offline.invalid', api_key='offline',
            include_thoughts=False, heartbeat_interval=1, max_retries=0, retry_delay=0, timeout=httpx.Timeout(2))
        transport_deps = gemini.GeminiGenerateContentDeps(log=logs.append, save_request_log=lambda *a, **kw: None,
            build_openai_sse_error=lambda *a: b'', runtime_lookup=lookup)
        body = payload(); body['temperature'] = 0
        with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)):
            for _ in range(2):
                await gemini._request_gemini_generate_content(model=body['model'], openai_request=body,
                    config=config, deps=transport_deps, trace_id='offline-native')
        self.assertEqual(sent[0], sent[1]); self.assertNotIn('stream', sent[0])
        self.assertIn('<!--', sent[0]['systemInstruction']['parts'][0]['text'])
        self.assertIn('<UpdateVariable>', sent[0]['contents'][0]['parts'][0]['text'])
        self.assertEqual(sent[0]['generationConfig']['temperature'], .1)
        self.assertEqual(sum('gemini_anti_route applied' in x for x in logs), 1)

    async def test_canonical_exact_dedupe_runs_perturbation_and_upstream_once(self):
        coalescer = ExactRequestCoalescer(ttl=60, log=lambda _: None)
        inputs = [payload() for _ in range(3)]; originals = copy.deepcopy(inputs)
        keys = [coalescer.build_key(p) for p in inputs]
        sent = []
        def dispatch(request):
            sent.append(json.loads(request.content)); return httpx.Response(200, json=json_reply('gemini-example'))
        async def caller(index):
            async def runner():
                result = await chat.collect_chat_completions_nonstream(url='http://offline.invalid/v1/chat/completions',
                    request_data=copy.deepcopy(inputs[index]), headers={}, timeout=httpx.Timeout(2), deps=deps())
                return result[5]
            return await coalescer.run(dedupe_key=keys[index], trace_id=str(index), upstream_label='offline', runner=runner)
        with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)):
            results = await asyncio.gather(caller(0), caller(1))
            recent = await caller(2)
        self.assertEqual(len(set(keys)), 1); self.assertEqual(len(sent), 1)
        self.assertEqual(inputs, originals); self.assertTrue(recent[1])
        self.assertEqual(results[0][0], results[1][0])
        self.assertIn('<!--', sent[0]['messages'][0]['content'])
