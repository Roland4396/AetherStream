import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from aetherstream.features.scene_filter import SCENE_INTRO, remove_scene_messages, remove_scene_text
from aetherstream.features.stage_warning import STAGE_INTRO, wrap_stage_messages


SCENE = '''<scene>
Day:1, 12:48 | 图书馆 | 天气=晴朗微风 | 在场: 测试角色
【场景】一张办公桌。 | 可交互=桌子、书
【氛围】安静
【类型】日常 | 难度=简单
期限=13:00(闭馆) | 跳跃=无
</scene>'''
INJECTED = SCENE_INTRO + '\n' + SCENE
STAGE = '<stage><act name="测试角色">now: 看书</act></stage>'
STATE = '系统:\n  当前年份: 2024\n  当前日期: 4月9日\n  当前时间: 12:47\n现在的时间是: 12:47'


class SceneFilterTests(unittest.TestCase):
    def test_exact_intro_and_whole_scene_removed(self):
        text = '正文12:48\n' + INJECTED + '\n<recall>AM0001</recall>\n' + STATE
        expected = '正文12:48\n\n<recall>AM0001</recall>\n' + STATE
        self.assertEqual(remove_scene_text(text), (expected, 1, 0))

    def test_stage_and_recall_preserved_verbatim(self):
        prefix = STAGE_INTRO + '\n' + STAGE + '\n\n'
        suffix = '\n\n以下是<记忆回溯>的编码：\n<recall>AM0001</recall>'
        self.assertEqual(remove_scene_text(prefix + INJECTED + suffix), (prefix + suffix, 1, 0))

    def test_all_duplicates_and_latest_human_copy_removed(self):
        text = INJECTED + '\n<latest_human_message>\n输入\n' + INJECTED + '\n</latest_human_message>'
        expected = '\n<latest_human_message>\n输入\n\n</latest_human_message>'
        self.assertEqual(remove_scene_text(text), (expected, 2, 0))
        self.assertEqual(remove_scene_text(expected), (expected, 0, 0))

    def test_values_are_not_hardcoded_or_validated_as_truth(self):
        text = INJECTED.replace('Day:1, 12:48', 'Day:999, 错误时间').replace('期限=13:00(闭馆)', '期限=无')
        self.assertEqual(remove_scene_text(text), ('', 1, 0))

    def test_crlf_and_indent_preserve_surrounding_whitespace(self):
        block = ('  ' + SCENE_INTRO + '\r\n\t' + SCENE).replace('\n', '\r\n')
        self.assertEqual(remove_scene_text('before\n' + block + '\nend'), ('before\n\nend', 1, 0))

    def test_unanchored_scene_and_prose_time_unchanged(self):
        text = SCENE + '\n正文：明天12:48去图书馆。\n' + STATE
        self.assertEqual(remove_scene_text(text), (text, 0, 0))

    def test_inline_quote_intro_is_not_an_injection(self):
        text = '文档示例：' + INJECTED
        self.assertEqual(remove_scene_text(text), (text, 0, 0))

    def test_instruction_mention_and_similar_tags_untouched(self):
        for text in (SCENE_INTRO, '<scene_hint>12:48</scene_hint>', '请输出<scene>标签。',
                     SCENE_INTRO + '\n中间有别的正文\n' + SCENE):
            with self.subTest(text=text):
                self.assertEqual(remove_scene_text(text), (text, 0, 0))

    def test_malformed_and_ambiguous_fail_closed(self):
        for block in (SCENE[:-8], '<scene></scene>', '<scene>12:48</scene>',
                      '<scene>' + SCENE + '</scene>', '<scene/>' + SCENE,
                      SCENE.replace('<scene>', '<scene name="different">'),
                      SCENE.replace('【场景】', '场景:'),
                      SCENE.replace('【氛围】安静', '<recall>outside</recall>\n【氛围】安静')):
            text = SCENE_INTRO + '\n' + block
            with self.subTest(block=block):
                self.assertEqual(remove_scene_text(text), (text, 0, 1))

    def test_valid_after_closed_invalid_block(self):
        bad = SCENE_INTRO + '\n<scene>not the injected format</scene>\n'
        self.assertEqual(remove_scene_text(bad + INJECTED), (bad, 1, 1))

    def test_current_historical_and_multimodal_user_text(self):
        messages = [{'role': r, 'content': INJECTED} for r in ('system', 'developer', 'assistant', 'tool')]
        messages += [{'role': 'user', 'content': INJECTED}, {'role': 'assistant', 'content': '保留'},
                     {'role': 'user', 'content': [
                         {'type': 'text', 'text': INJECTED + '\n' + STATE,
                          'cache_control': {'type': 'ephemeral'}},
                         {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,example'}},
                     ]}]
        old = copy.deepcopy(messages)
        self.assertEqual(remove_scene_messages(messages), (2, 0))
        self.assertEqual(messages[:4], old[:4])
        self.assertEqual(messages[4], {'role': 'user', 'content': ''})
        self.assertEqual(messages[5], old[5])
        self.assertEqual(messages[6]['content'][0]['text'], '\n' + STATE)
        self.assertEqual(messages[6]['content'][0]['cache_control'], {'type': 'ephemeral'})
        self.assertEqual(messages[6]['content'][1], old[6]['content'][1])
        self.assertEqual(remove_scene_messages(messages), (0, 0))

    def test_split_blocks_not_guessed(self):
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': INJECTED[:-8]}, {'type': 'text', 'text': '</scene>'}]}]
        old = copy.deepcopy(messages)
        self.assertEqual(remove_scene_messages(messages), (0, 1))
        self.assertEqual(messages, old)

    def test_empty_nontext_and_invalid_inputs(self):
        self.assertEqual(remove_scene_messages(None), (0, 0))
        messages = [None, {}, {'role': 'user', 'content': None}, {'role': 'user', 'content': [None]}]
        old = copy.deepcopy(messages)
        self.assertEqual(remove_scene_messages(messages), (0, 0))
        self.assertEqual(messages, old)

    def test_stage_warning_order_does_not_change_scope(self):
        original = [{'role': 'user', 'content': STAGE_INTRO + '\n' + STAGE + '\n' + INJECTED}]
        first, second = copy.deepcopy(original), copy.deepcopy(original)
        self.assertEqual(remove_scene_messages(first), (1, 0))
        self.assertEqual(wrap_stage_messages(first), (1, 0))
        self.assertEqual(wrap_stage_messages(second), (1, 0))
        self.assertEqual(remove_scene_messages(second), (1, 0))
        self.assertEqual(first, second)

    def test_counts_visible_in_default_logs(self):
        from aetherstream.observability.logging import ProxyLogger
        logger = object.__new__(ProxyLogger)
        logger.verbose_trace = False
        self.assertTrue(logger._should_emit('[TRACE test] ai_scene_filter removed=5 invalid_skipped=0'))
        self.assertTrue(logger._should_emit('[TRACE test] ai_scene_filter removed=0 invalid_skipped=1'))


class SceneFilterProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_public_protocols_and_stream_modes_clean_before_upstream(self):
        from starlette.requests import Request
        from aetherstream.api import app
        from aetherstream.api.chat_routes import chat_completions
        from aetherstream.api.protocol_gateway import route_protocol_request

        model = 'free/claude-opus-4-6'
        messages = [
            {'role': 'user', 'content': '旧输入\n' + INJECTED + '\n<recall>AM0001</recall>'},
            {'role': 'assistant', 'content': '正文：12:40到达。'},
            {'role': 'user', 'content': STATE + '\n' + STAGE_INTRO + '\n' + STAGE + '\n'
             + INJECTED + '\n<latest_human_message>\n' + INJECTED + '\n</latest_human_message>'},
        ]
        for protocol in ('chat', 'anthropic', 'responses'):
            for stream in (True, False):
                with self.subTest(protocol=protocol, stream=stream):
                    captured, logs = [], []
                    deps = copy.copy(app.chat_route_dependencies)
                    deps.log = logs.append
                    deps.active_stream_registry = type(app.active_stream_registry)(log=logs.append)
                    deps.release_active_stream_caller = deps.active_stream_registry.release
                    deps.save_request_log = lambda *args, **kwargs: None
                    deps.replay_service = SimpleNamespace(prepare=lambda **kwargs: None)
                    deps.get_claude_upstream_for_provider = lambda _: ('offline', 'http://offline.invalid/v1')
                    deps.resolve_openai_compatible_route = AsyncMock(return_value={
                        'name': 'free-antigravity', 'base_url': 'http://offline.invalid/v1',
                        'api_key': 'offline', 'inject_opus_note': True,
                    })

                    async def fake_stream(**kwargs):
                        captured.append(copy.deepcopy(kwargs['request_data']))
                        event = {'id': 'chatcmpl-offline', 'model': model, 'choices': [
                            {'index': 0, 'delta': {'content': 'OK'}, 'finish_reason': 'stop'}]}
                        yield ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()

                    async def fake_collect(**kwargs):
                        captured.append(copy.deepcopy(kwargs['request_data']))
                        return ('OK', model, {}, 'stop', '')

                    deps.forward_chat_completions_stream = fake_stream
                    deps.collect_chat_completions_stream = fake_collect
                    payload = {'model': model, 'stream': stream, 'max_tokens': 64,
                               'input' if protocol == 'responses' else 'messages': copy.deepcopy(messages)}
                    original = copy.deepcopy(payload)
                    request = Request({'type': 'http', 'method': 'POST', 'path': '/offline',
                                       'headers': [], 'client': ('127.0.0.1', 1234)})
                    request._body = json.dumps(payload).encode()
                    with patch('httpx.AsyncClient.send', side_effect=AssertionError('Offline test only')):
                        if protocol == 'chat':
                            response = await chat_completions(request, deps)
                        else:
                            response = await route_protocol_request(request, deps, protocol=protocol)
                        if hasattr(response, 'body_iterator'):
                            async for _ in response.body_iterator:
                                pass
                    self.assertEqual(response.status_code, 200, logs)
                    self.assertEqual(len(captured), 1, logs)
                    text = json.dumps(captured[0]['messages'], ensure_ascii=False)
                    self.assertNotIn('<scene>', text)
                    self.assertNotIn(SCENE_INTRO, text)
                    self.assertIn('<recall>AM0001</recall>', text)
                    self.assertIn('当前年份: 2024', text)
                    self.assertIn('现在的时间是: 12:47', text)
                    self.assertIn('正文：12:40到达。', text)
                    self.assertIn(STAGE, text.replace('\\"', '"'))
                    self.assertEqual(payload, original)
                    self.assertTrue(any('ai_scene_filter removed=3 invalid_skipped=0' in s for s in logs))


if __name__ == '__main__':
    unittest.main()
