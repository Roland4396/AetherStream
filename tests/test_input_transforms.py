"""Pure conversions and mocked final-wire integration, never paid inference."""
import asyncio
import copy
import inspect
import json
import random
import re
import unittest
from unittest.mock import patch

import httpx

from aetherstream.features.gemini_anti_route import GeminiAntiRouteSession
from aetherstream.features.input_transforms import (
    LEXICONS, convert_prose, settings_from, transform_request_input, transform_text,
)
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.upstreams import openai_chat_completions as chat
from aetherstream.upstreams import gemini_generate_content as gemini

REAL_CLIENT = httpx.AsyncClient
CONFIG = {'enabled': True, 'mode': 'deep', 'input_transform': {'enabled': True, 'mode': 'pinyin'}}
ASSEMBLER_FOOTER = ('\n\n以上是用户的本轮输入，以下act是各角色们的行动和台词'
                    '（在写正文时结合用户的输入丝滑穿插交织）：\n'
                    '<act>保持角色行动原文</act>\n<scene>保持场景原文</scene>')


def settings(**extra):
    return settings_from({'enabled': True, 'wrapper': False, **extra})


def body(model='gemini-offline'):
    return {'model': model, 'stream': False, 'messages': [
        {'role': 'system', 'content': '系统设定必须保持原有中文，不做拼音转换。'},
        {'role': 'user', 'content': '历史输入不要转换。'},
        {'role': 'assistant', 'content': '历史回复也不要转换。'},
        {'role': 'user', 'content': '你好，世界！'}], 'temperature': .7}


def reply(model):
    return {'id': 'offline', 'model': model, 'choices': [{'index': 0,
        'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}]}


def sse(model):
    events = [{'id': 'offline', 'model': model, 'choices': [{'index': 0, 'delta': {'content': 'OK'}, 'finish_reason': None}]},
              {'id': 'offline', 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}]
    return ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n'


def deps(logs, saved):
    return chat.ChatCompletionsUpstreamDeps(log=logs.append,
        save_request_log=lambda *a, **kw: saved.append(copy.deepcopy(kw)),
        build_openai_sse_error=lambda *a: b'UNEXPECTED_ERROR', has_stop_tag=lambda _: False,
        find_stop_tag=lambda _: -1, fmt_ms=lambda *_: '0ms', release_caller=lambda *_: None,
        runtime_lookup=lambda *_: CONFIG)


class ConversionTests(unittest.TestCase):
    def convert(self, text, **cfg):
        return transform_text(text, settings(**cfg), random.Random(17))

    def test_disabled_default_and_invalid_mode(self):
        self.assertFalse(settings_from(None)['enabled'])
        with self.assertRaises(ValueError):
            settings(mode='unknown')
        data = body(); original = copy.deepcopy(data)
        self.assertIsNone(transform_request_input(data, settings_from(None), random.Random(1)))
        self.assertEqual(data, original)

    def test_pinyin_tones_and_contextual_polyphones(self):
        self.assertEqual(self.convert('你好，世界！'), 'nǐ hǎo，shì jiè！')
        self.assertEqual(self.convert('重庆银行'), 'chóng qìng yín háng')
        self.assertEqual(self.convert('你好', tone='number'), 'ni3 hao3')
        self.assertEqual(self.convert('你好', tone='none'), 'ni hao')

    def test_non_chinese_symbols_emoji_and_latin_edges(self):
        self.assertEqual(self.convert('Hello你好world🙂'), 'Hello nǐ hǎo world🙂')
        self.assertEqual(self.convert('123 & ABC🙂'), '123 & ABC🙂')

    def test_homophones_are_replacements_with_preserved_punctuation(self):
        result = self.convert('你好，世界！', mode='harmonic', ratio=100)
        self.assertNotEqual(result, '你好，世界！')
        self.assertEqual(len(result), len('你好，世界！'))
        self.assertEqual(result[2], '，'); self.assertEqual(result[-1], '！')
        self.assertEqual(self.convert('你好，世界！', mode='harmonic', ratio=0), '你好，世界！')

    def test_fusion_ratio_extremes(self):
        self.assertEqual(self.convert('你好', mode='fusion', ratio=100), 'nǐ hǎo')
        result = self.convert('你好', mode='fusion', ratio=0)
        self.assertNotEqual(result, '你好')
        self.assertNotRegex(result, '[a-z]')

    def test_english_spanish_mix_and_zero_ratio(self):
        for language in ['en', 'es']:
            result = self.convert('我们可以继续故事。', mode='mix', ratio=100, languages=[language])
            self.assertRegex(result, '[A-Za-z]')
            self.assertTrue(result.endswith('。'))
        self.assertEqual(self.convert('我们可以继续故事。', mode='mix', ratio=0), '我们可以继续故事。')
        self.assertEqual(len(LEXICONS['homophones']), 392)

    def test_technical_regions_and_variables_stay_literal_without_exempting_message(self):
        regions = ['`中文代码`', '```json\n{"中文":"原文"}\n```', '{{中文宏}}',
                   'https://example.invalid/中文路径', '/状态/名称', '<标签 属性="中文">',
                   '<UpdateVariable><JSONPatch>[{"path":"/状态/名称","value":"中文"}]</JSONPatch></UpdateVariable>',
                   '{"中文键":"中文值"}']
        text = '你好\n' + '\n'.join(regions) + '\n世界'
        result = self.convert(text)
        self.assertIn('nǐ hǎo', result); self.assertIn('shì jiè', result)
        for region in regions:
            self.assertIn(region, result)

    def test_unclosed_code_fence_is_not_corrupted(self):
        result = self.convert('你好\n```\n保持未闭合代码')
        self.assertEqual(result, 'nǐ hǎo\n```\n保持未闭合代码')

    def test_preserve_terms_for_character_identity(self):
        self.assertEqual(self.convert('你好苏拉', preserve_terms=['苏拉']), 'nǐ hǎo苏拉')

    def test_neutral_wrapper_is_small_and_does_not_claim_permissions(self):
        result = transform_text('你好', settings(wrapper=True, padding_chars=32), random.Random(1))
        self.assertIn('<stream_input_transform form="pinyin">', result)
        self.assertIn('nǐ hǎo', result)
        self.assertEqual(len(re.findall('<!--', result)), 2)
        self.assertLess(len(result), 300)
        self.assertNotRegex(result, 'AUDIT|STATUS PASS|policy gate|permission|ignore')

    def test_latest_human_only_not_system_history_or_response_schema(self):
        data = body(); data['response_format'] = {'type': 'json_schema', 'json_schema': {'name': '中文格式'}}
        original = copy.deepcopy(data)
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['message'], 3)
        self.assertEqual(data['messages'][:3], original['messages'][:3])
        self.assertEqual(data['response_format'], original['response_format'])
        self.assertEqual(data['messages'][3]['content'], 'nǐ hǎo，shì jiè！')

    def test_marked_human_input_preserves_surrounding_instructions(self):
        data = {'messages': [{'role': 'user', 'content': '世界书资料\n<latest_human_message>你好</latest_human_message>\n<act>保持角色行动原文</act>'}]}
        transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(data['messages'][0]['content'], '世界书资料\n<latest_human_message>nǐ hǎo</latest_human_message>\n<act>保持角色行动原文</act>')

    def test_plain_input_preserves_appended_stage_and_fact_notes(self):
        for suffix in ['\n\n以上是用户的本轮输入，以下act是各角色们的行动和台词：\n<act>角色行动</act>',
                       '\n\n!!important!!\n事实核对要求', '\n\n<plot_progress_control>进度要求</plot_progress_control>']:
            data = {'messages': [{'role': 'user', 'content': '你好' + suffix}]}
            transform_request_input(data, settings(), random.Random(1))
            self.assertEqual(data['messages'][0]['content'], 'nǐ hǎo' + suffix)

    def test_synthetic_continuation_does_not_replace_actual_human_selection(self):
        data = body(); synthetic = {'role': 'user', 'content': '开始吧。JSON 必须合法并符合 schema；保留格式。'}
        data['messages'].append(copy.deepcopy(synthetic))
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['message'], 3); self.assertEqual(data['messages'][-1], synthetic)

    def test_compiled_planner_uses_input_block_not_final_preset_control(self):
        compound = '玩家：首先要看的是DM规则\n<dm_core>保持规则原文</dm_core>\n<Information>世界书原文</Information>\n下接玩家最新的input：\n<input>你好</input>\n输出格式保留原文'
        control = '即将开始思考，角色需按既有格式输出。\nCRUCIAL: 保留控制指令。'
        data = {'messages': [{'role': 'user', 'content': '[Start a new chat]'},
                             {'role': 'assistant', 'content': '初始化'},
                             {'role': 'user', 'content': compound},
                             {'role': 'assistant', 'content': '预置回答'},
                             {'role': 'user', 'content': control}]}
        original = copy.deepcopy(data)
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['message'], 2)
        self.assertEqual(data['messages'][2]['content'], compound.replace('<input>你好</input>', '<input>nǐ hǎo</input>'))
        self.assertEqual(data['messages'][4], original['messages'][4])
        self.assertEqual(data['messages'][0], original['messages'][0])

    def test_assembled_stage_input_before_post_history_user_rules(self):
        for tag in ['Legal_Disclaimer_and_Rights_Declaration', 'FM2.1-UNIFIED']:
            with self.subTest(tag=tag):
                control = '<' + tag + '>预设规则' + '保持原文' * 3000 + '</' + tag + '>'
                data = {'messages': [{'role': 'assistant', 'content': '历史回复'},
                                     {'role': 'user', 'content': '你好' + ASSEMBLER_FOOTER},
                                     {'role': 'user', 'content': control},
                                     {'role': 'system', 'content': '后置格式规则'}]}
                original = copy.deepcopy(data)
                result = transform_request_input(data, settings(), random.Random(1))
                self.assertEqual(result['status'], 'applied')
                self.assertEqual(result['message'], 1)
                self.assertEqual(result['source_chars'], 2)
                self.assertEqual(data['messages'][1]['content'], 'nǐ hǎo' + ASSEMBLER_FOOTER)
                self.assertEqual(data['messages'][0], original['messages'][0])
                self.assertEqual(data['messages'][2:], original['messages'][2:])

    def test_stage_footer_literal_in_code_is_not_an_input_anchor(self):
        literal = '```text\n原样保留' + ASSEMBLER_FOOTER + '\n```'
        data = {'messages': [{'role': 'user', 'content': literal},
                             {'role': 'user', 'content': '<FM2.1-UNIFIED>预设规则</FM2.1-UNIFIED>'}]}
        original = copy.deepcopy(data)
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['reason'], 'no_human_prose')
        self.assertEqual(data, original)
        plain = {'messages': [{'role': 'user', 'content': '你好\n' + literal + '\n世界'}]}
        transform_request_input(plain, settings(), random.Random(1))
        self.assertEqual(plain['messages'][0]['content'], 'nǐ hǎo\n' + literal + '\nshì jiè')

    def test_plain_current_input_wins_over_historical_stage_footer(self):
        historical = {'role': 'user', 'content': '历史输入' + ASSEMBLER_FOOTER}
        data = {'messages': [copy.deepcopy(historical), {'role': 'assistant', 'content': '历史回复'},
                             {'role': 'user', 'content': '你好'}]}
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['message'], 2)
        self.assertEqual(data['messages'][0], historical)
        self.assertEqual(data['messages'][2]['content'], 'nǐ hǎo')

    def test_unmarked_long_user_input_before_presets_ignores_legacy_cap(self):
        text = '中文' * 10000
        data = {'messages': [{'role': 'assistant', 'content': '历史回复'},
                             {'role': 'user', 'content': text},
                             {'role': 'user', 'content': '<FM2.1-UNIFIED>预设规则</FM2.1-UNIFIED>'}]}
        original = copy.deepcopy(data)
        result = transform_request_input(data, settings(max_source_chars=32), random.Random(1))
        self.assertEqual(result['status'], 'applied')
        self.assertEqual(result['message'], 1)
        self.assertEqual(result['selection'], 'latest_user')
        self.assertEqual(result['source_chars'], len(text))
        self.assertEqual(len(data['messages'][1]['content'].split()), len(text))
        self.assertEqual(data['messages'][0], original['messages'][0])
        self.assertEqual(data['messages'][2:], original['messages'][2:])

    def test_native_and_multimodal_stage_boundaries_preserve_other_parts(self):
        for native in [False, True]:
            with self.subTest(native=native):
                if native:
                    parts = [{'text': '你好' + ASSEMBLER_FOOTER}, {'text': '其他规则'},
                             {'functionResponse': {'name': '工具', 'response': {'中文': '原文'}}}]
                    data = {'contents': [{'role': 'user', 'parts': parts},
                                         {'role': 'user', 'parts': [{'text': '<FM2.1-UNIFIED>格式规则</FM2.1-UNIFIED>'}]}]}
                else:
                    parts = [{'type': 'text', 'text': '你好' + ASSEMBLER_FOOTER},
                             {'type': 'text', 'text': '其他规则'},
                             {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
                    data = {'messages': [{'role': 'user', 'content': parts},
                                         {'role': 'user', 'content': '<FM2.1-UNIFIED>格式规则</FM2.1-UNIFIED>'}]}
                original = copy.deepcopy(data)
                result = transform_request_input(data, settings(), random.Random(1))
                key = 'contents' if native else 'messages'
                part_key = 'parts' if native else 'content'
                self.assertEqual(result['message'], 0)
                self.assertEqual(data[key][0][part_key][0]['text'], 'nǐ hǎo' + ASSEMBLER_FOOTER)
                self.assertEqual(data[key][0][part_key][1:], original[key][0][part_key][1:])
                self.assertEqual(data[key][1], original[key][1])

    def test_plain_current_input_wins_over_old_marked_history(self):
        data = {'messages': [{'role': 'user', 'content': '<user_input>历史</user_input>'},
                             {'role': 'assistant', 'content': '历史回复'}, {'role': 'user', 'content': '你好'}]}
        transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(data['messages'][0]['content'], '<user_input>历史</user_input>')
        self.assertEqual(data['messages'][2]['content'], 'nǐ hǎo')

    def test_regeneration_selects_last_stored_user_even_before_assistant(self):
        data = {'messages': [{'role': 'user', 'content': '旧的玩家输入'},
                             {'role': 'assistant', 'content': '旧回复'},
                             {'role': 'user', 'content': '即将开始思考，保留预设指令。'}]}
        original = copy.deepcopy(data)
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['status'], 'applied')
        self.assertEqual(result['message'], 0)
        self.assertEqual(result['selection'], 'latest_user')
        self.assertNotEqual(data['messages'][0], original['messages'][0])
        self.assertEqual(data['messages'][1:], original['messages'][1:])

    def test_ordinary_unmarked_input_before_multiple_post_history_presets(self):
        data = {'messages': [{'role': 'assistant', 'content': '历史回复'},
                             {'role': 'user', 'content': '你好'},
                             {'role': 'user', 'content': '<FM2.1-UNIFIED>图像规则</FM2.1-UNIFIED>'},
                             {'role': 'user', 'content': '即将开始思考，格式规则。'},
                             {'role': 'system', 'content': '后置规则'}]}
        original = copy.deepcopy(data)
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['status'], 'applied')
        self.assertEqual(result['message'], 1)
        self.assertEqual(result['selection'], 'latest_user')
        self.assertEqual(data['messages'][1]['content'], 'nǐ hǎo')
        self.assertEqual(data['messages'][2:], original['messages'][2:])

    def test_last_plain_user_after_preset_still_wins_without_markers(self):
        data = {'messages': [{'role': 'user', 'content': '历史输入'},
                             {'role': 'assistant', 'content': '历史回复'},
                             {'role': 'user', 'content': '<FM2.1-UNIFIED>图像规则</FM2.1-UNIFIED>'},
                             {'role': 'user', 'content': '你好'}]}
        original = copy.deepcopy(data)
        result = transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(result['message'], 3)
        self.assertEqual(data['messages'][:3], original['messages'][:3])
        self.assertEqual(data['messages'][3]['content'], 'nǐ hǎo')

    def test_literal_input_tags_in_code_are_not_human_boundary_markers(self):
        data = {'messages': [{'role': 'user', 'content': '你好，查看示例 `<input>保留代码</input>`'}]}
        transform_request_input(data, settings(), random.Random(1))
        self.assertIn('nǐ hǎo', data['messages'][0]['content'])
        self.assertIn('`<input>保留代码</input>`', data['messages'][0]['content'])

    def test_multimodal_keeps_images_and_tools_literal(self):
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}
        data = {'messages': [{'role': 'tool', 'content': '工具中文'},
                             {'role': 'user', 'content': [{'type': 'text', 'text': '你好'}, image]}]}
        transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(data['messages'][0]['content'], '工具中文')
        self.assertEqual(data['messages'][1]['content'][1], image)
        self.assertEqual(data['messages'][1]['content'][0]['text'], 'nǐ hǎo')

    def test_native_google_parts_preserve_function_response_and_signature(self):
        signed = {'text': '签名内容', 'thoughtSignature': 'sig'}
        function = {'functionResponse': {'name': '测试工具', 'response': {'中文': '原文'}}}
        data = {'systemInstruction': {'parts': [{'text': '系统原文'}]}, 'contents': [
            {'role': 'model', 'parts': [signed]}, {'role': 'user', 'parts': [{'text': '你好'}, function]}]}
        original = copy.deepcopy(data); transform_request_input(data, settings(), random.Random(1))
        self.assertEqual(data['systemInstruction'], original['systemInstruction'])
        self.assertEqual(data['contents'][0], original['contents'][0])
        self.assertEqual(data['contents'][1]['parts'][1], function)
        self.assertEqual(data['contents'][1]['parts'][0]['text'], 'nǐ hǎo')

    def test_long_source_in_all_text_parts_is_fully_transformed_without_cap(self):
        data = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': '你好'}, {'type': 'text', 'text': '中文' * 40}]}]}
        result = transform_request_input(data, settings(max_source_chars=32), random.Random(1))
        self.assertEqual(result['status'], 'applied')
        self.assertEqual(result['parts'], 2)
        self.assertEqual(result['source_chars'], 82)
        self.assertEqual(data['messages'][0]['content'][0]['text'], 'nǐ hǎo')
        self.assertEqual(len(data['messages'][0]['content'][1]['text'].split()), 80)
        self.assertNotIn('max_source_chars', settings_from({'max_source_chars':32}))

    def test_json_only_unclosed_human_and_already_wrapped_are_not_reprocessed(self):
        for text in ['{"中文":"值"}', '<user_input>未闭合', '<stream_input_transform form="pinyin">你好</stream_input_transform>']:
            data = {'messages': [{'role': 'user', 'content': text}]}; original = copy.deepcopy(data)
            transform_request_input(data, settings(), random.Random(1)); self.assertEqual(data, original)

    def test_integrated_session_retries_and_non_gemini_gating(self):
        session = GeminiAntiRouteSession(random.Random(1)); data = body(); original = copy.deepcopy(data)
        logs = []; apply = lambda p: session.apply(p, model=p['model'], lookup=lambda *_: CONFIG, log=logs.append)
        apply(data); converted = copy.deepcopy(data); apply(data); apply(original)
        self.assertEqual(data, converted); self.assertEqual(original, converted)
        self.assertEqual(sum('input_transform status=applied' in x for x in logs), 1)
        for model in ['claude-opus-example', 'gpt-example', 'glm-example']:
            other = body(model); before = copy.deepcopy(other)
            session.apply(other, model=model, lookup=lambda *_: CONFIG, log=logs.append)
            self.assertEqual(other, before)


class TransformWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_history_rule_layout_reaches_final_wire_with_only_player_transliterated(self):
        for fn in [chat.forward_chat_completions_stream, chat.collect_chat_completions_stream,
                   chat.collect_chat_completions_nonstream, chat.replay_chat_completions_nonstream_as_stream]:
            with self.subTest(transport=fn.__name__):
                data = body()
                data['stream'] = fn is chat.forward_chat_completions_stream
                data['messages'][-1]['content'] = '你好' + ASSEMBLER_FOOTER
                data['messages'].append({'role': 'user', 'content': '<FM2.1-UNIFIED>后置规则保持中文</FM2.1-UNIFIED>'})
                data['messages'].append({'role': 'system', 'content': '其他格式规则保持中文'})
                inbound = copy.deepcopy(data); sent, logs, saved = [], [], []
                def dispatch(request):
                    out = json.loads(request.content); sent.append(out)
                    return (httpx.Response(200, text=sse(data['model']), headers={'content-type': 'text/event-stream'})
                            if out.get('stream') else httpx.Response(200, json=reply(data['model'])))
                params = dict(url='http://offline.invalid/v1/chat/completions', request_data=data,
                    headers={}, timeout=httpx.Timeout(2), max_raw_sse_bytes=99999, deps=deps(logs, saved),
                    model=data['model'], messages=data['messages'], trace_id='offline-boundary', enable_early_stop=False)
                params = {k:v for k,v in params.items() if k in inspect.signature(fn).parameters}
                with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)):
                    result = fn(**params)
                    if inspect.isasyncgen(result):
                        self.assertNotIn(b'UNEXPECTED_ERROR', b''.join([c async for c in result]))
                    else:
                        await result
                self.assertEqual(len(sent), 1)
                messages = sent[0]['messages']
                self.assertIn('<stream_input_transform form="pinyin">', messages[3]['content'])
                self.assertIn('nǐ hǎo', messages[3]['content'])
                # The pre-existing deep shield may still add trailing spaces;
                # the new transliteration must preserve the footer's Chinese.
                normalized = '\n'.join(line.rstrip() for line in messages[3]['content'].split('\n'))
                self.assertIn(ASSEMBLER_FOOTER, normalized)
                self.assertIn('后置规则保持中文', messages[4]['content'])
                self.assertNotIn('<stream_input_transform', messages[4]['content'])
                self.assertIn('其他格式规则保持中文', messages[5]['content'])
                self.assertEqual(inbound['messages'][3]['content'], '你好' + ASSEMBLER_FOOTER)

    async def test_all_chat_transports_send_transliteration_and_keep_original_input_copy(self):
        for fn in [chat.forward_chat_completions_stream, chat.collect_chat_completions_stream,
                   chat.collect_chat_completions_nonstream, chat.replay_chat_completions_nonstream_as_stream]:
            with self.subTest(transport=fn.__name__):
                data = body(); data['stream'] = fn is chat.forward_chat_completions_stream
                inbound = copy.deepcopy(data); sent, logs, saved = [], [], []
                def dispatch(request):
                    out = json.loads(request.content); sent.append(out)
                    return (httpx.Response(200, text=sse(data['model']), headers={'content-type': 'text/event-stream'})
                            if out.get('stream') else httpx.Response(200, json=reply(data['model'])))
                params = dict(url='http://offline.invalid/v1/chat/completions', request_data=data,
                    headers={}, timeout=httpx.Timeout(2), max_raw_sse_bytes=99999, deps=deps(logs, saved),
                    model=data['model'], messages=data['messages'], trace_id='offline-input', enable_early_stop=False)
                params = {k: v for k, v in params.items() if k in inspect.signature(fn).parameters}
                with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)):
                    result = fn(**params)
                    if inspect.isasyncgen(result):
                        self.assertNotIn(b'UNEXPECTED_ERROR', b''.join([c async for c in result]))
                    else:
                        await result
                self.assertEqual(len(sent), 1)
                self.assertIn('<stream_input_transform', sent[0]['messages'][-1]['content'])
                self.assertIn('nǐ hǎo', sent[0]['messages'][-1]['content'])
                self.assertIn('系统设定必须保持原有中文', sent[0]['messages'][0]['content'])
                self.assertIn('历史输入不要转换', sent[0]['messages'][1]['content'])
                self.assertEqual(inbound['messages'][-1]['content'], '你好，世界！')
                self.assertEqual(data['messages'], sent[0]['messages'])
                for record in saved:
                    self.assertEqual(record['request_payload']['messages'], sent[0]['messages'])

    async def test_native_conversion_and_exact_dedupe_still_work(self):
        data = body(); original = copy.deepcopy(data); sent = []
        def dispatch(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'OK'}]}, 'finishReason': 'STOP'}]})
        cfg = gemini.GeminiGenerateContentConfig(base_url='http://offline.invalid', api_key='offline',
            include_thoughts=False, heartbeat_interval=1, max_retries=0, retry_delay=0, timeout=httpx.Timeout(2))
        d = gemini.GeminiGenerateContentDeps(log=lambda _: None, save_request_log=lambda *a, **kw: None,
            build_openai_sse_error=lambda *a: b'', runtime_lookup=lambda *_: CONFIG)
        coalescer = ExactRequestCoalescer(ttl=60, log=lambda _: None)
        key = coalescer.build_key(data)
        async def run():
            return await gemini._request_gemini_generate_content(model=data['model'], openai_request=data,
                config=cfg, deps=d, trace_id='offline-native-input')
        with patch('httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(dispatch), timeout=2)):
            await asyncio.gather(*[coalescer.run(dedupe_key=key, trace_id=str(i), upstream_label='offline', runner=run) for i in range(2)])
        self.assertEqual(len(sent), 1)
        self.assertIn('<stream_input_transform', sent[0]['contents'][-1]['parts'][0]['text'])
        self.assertIn('nǐ hǎo', sent[0]['contents'][-1]['parts'][0]['text'])
        self.assertIn('系统设定必须保持原有中文', sent[0]['systemInstruction']['parts'][0]['text'])
        self.assertNotEqual(data, original)
