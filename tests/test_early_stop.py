import unittest
from types import SimpleNamespace

from aetherstream.features.early_stop import DEFAULT_EARLY_STOP_TAGS, EarlyStopMatcher
from aetherstream.features.early_stop import is_himodels_upstream
from aetherstream.features.early_stop import bind_content_guard, request_uses_content_blocks


class EarlyStopMatcherTests(unittest.TestCase):
    def guarded_finder(self, request=None):
        matcher = EarlyStopMatcher(lookup=lambda *_: None, env_enabled=True,
                                   env_tags=None, env_case_sensitive=True)
        if request is None:
            request = {'messages': [{'role': 'user', 'content': '正文使用 <content>…</content>。'}]}
        return bind_content_guard(matcher.find, request)

    def test_contract_detection_supports_system_and_content_blocks(self):
        self.assertTrue(request_uses_content_blocks({'system': [{'type': 'text', 'text': '<content>正文</content>'}]}))
        self.assertFalse(request_uses_content_blocks({'messages': [{'role': 'user', 'content': 'Reply OK.'}]}))

    def test_early_stage_disclaimer_does_not_stop(self):
        find = self.guarded_finder()
        self.assertEqual(find('承接核对｜NPC核对｜纠错重构\n<disclaimer>说明'), -1)
        self.assertEqual(find('<thinking>计划</thinking>\n纠错\n<disclaimer>说明'), -1)

    def test_body_must_have_a_matching_open_and_close(self):
        find = self.guarded_finder()
        self.assertEqual(find('</content>\n<disclaimer>'), -1)
        self.assertEqual(find('<content>正文中途<disclaimer>'), -1)
        self.assertEqual(find('<content>第一段</content><content>第二段<disclaimer>'), -1)

    def test_examples_inside_thinking_code_and_comments_do_not_arm(self):
        find = self.guarded_finder()
        for wrapper in ('<thinking>{}</thinking>', '<think>{}</think>', '```xml\n{}\n```', '`{}`', '<!-- {} -->'):
            with self.subTest(wrapper=wrapper):
                self.assertEqual(find(wrapper.format('<content>示例</content>')+'\n<disclaimer>'), -1)

    def test_valid_tail_still_stops_at_exact_position(self):
        text = '<content>完整正文</content>\n<aftertalk>后记</aftertalk>\n<disclaimer>尾部'
        self.assertEqual(self.guarded_finder()(text), text.index('<disclaimer>'))

    def test_ignored_early_tag_does_not_hide_or_retrigger_later_tag(self):
        find = self.guarded_finder()
        early = '纠错\n<disclaimer>提前说明</disclaimer>\n'
        body = '<content>正文</content>\n'
        self.assertEqual(find(early + body), -1)
        self.assertEqual(find(early + body + '<disclaimer>结尾'), len(early + body))

    def test_split_tag_and_closing_body_across_stream_chunks(self):
        find = self.guarded_finder()
        text = ''
        for part in ('纠错<discl', 'aimer>提前</disclaimer>', '<cont', 'ent>正文</cont', 'ent><discl', 'aimer>'):
            text += part
            result = find(text)
        self.assertEqual(result, text.rindex('<disclaimer>'))

    def test_unscoped_requests_and_other_stop_tags_are_unchanged(self):
        find = self.guarded_finder({'messages': [{'role': 'user', 'content': '普通请求'}]})
        self.assertEqual(find('答复<disclaimer>尾部'), 2)
        self.assertEqual(self.guarded_finder()('纠错<!--ST0P_PROXY_test'), 2)
        self.assertEqual(self.guarded_finder()('答复<closing_leaf>尾部'), 2)

    def test_legacy_disclaimer_body_marker_cannot_bypass_guard(self):
        marker = '[AI_SYSTEM detected: fixture]'
        matcher = EarlyStopMatcher(lookup=lambda *keys: ['<disclaimer>', marker] if keys == ('early_stop', 'tags') else None,
                                   env_enabled=True, env_tags=None, env_case_sensitive=True)
        find = bind_content_guard(matcher.find, {'system': '<content>正文</content>'})
        self.assertEqual(find('纠错<disclaimer>'+marker), -1)
        text = '<content>正文</content>'+marker
        self.assertEqual(find(text), text.index(marker))

    def test_himodels_detected_behind_pool_or_direct(self):
        for account in ('himodels_1', 'himodels_2', 'himodels', 'himodels-primary'):
            self.assertTrue(is_himodels_upstream(SimpleNamespace(headers={'x-account-pool-id': account}), 'http://account-pool-proxy/v1/chat/completions'))
        for url in ('https://api.himodels.ai/v1/chat/completions', 'https://himodels.ai/v1/messages'):
            self.assertTrue(is_himodels_upstream(SimpleNamespace(headers={}), url))
        for url in ('https://not-himodels.ai/v1', 'https://himodels.ai.example.com/v1'):
            self.assertFalse(is_himodels_upstream(SimpleNamespace(headers={}), url))

    def test_runtime_disabled_still_disables_tag_matching(self):
        matcher = EarlyStopMatcher(lookup=lambda *keys: False if keys == ('early_stop', 'enabled') else None,
                                   env_enabled=True, env_tags=None, env_case_sensitive=True)
        self.assertFalse(matcher.has('answer<disclaimer>tail'))

    def test_closing_leaf_is_a_default_stop_tag(self):
        matcher = EarlyStopMatcher(
            lookup=lambda *_keys: None,
            env_enabled=True,
            env_tags=None,
            env_case_sensitive=True,
        )

        text = "main response\n<closing_leaf>detached tail"

        self.assertIn("<closing_leaf>", DEFAULT_EARLY_STOP_TAGS)
        self.assertEqual(matcher.find(text), len("main response\n"))


if __name__ == "__main__":
    unittest.main()
