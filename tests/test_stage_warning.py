import copy
import unittest

from aetherstream.features.stage_warning import (
    STAGE_INTRO, WARNING_OPEN, WARNING_CLOSE, wrap_stage_messages, wrap_stage_text,
)


STAGE = '<stage>\n<act name="测试角色">\nnow: 示例\nthen: 待核验\n</act>\n</stage>'
ANCHORED = STAGE_INTRO + '\n' + STAGE


class StageWarningTests(unittest.TestCase):
    def test_wrapper_counts_visible_in_default_logs(self):
        from aetherstream.observability.logging import ProxyLogger
        logger = object.__new__(ProxyLogger)
        logger.verbose_trace = False
        for message in ('ai_stage_warning wrapped=1 invalid_skipped=0',
                        'ai_stage_warning wrapped=0 invalid_skipped=1'):
            self.assertTrue(logger._should_emit('[TRACE test] ' + message))

    def test_only_stage_is_wrapped_verbatim(self):
        suffix = '\n<scene>场景</scene>\n<recall>记忆</recall>\nstage纠错重写'
        text = '用户正文\n' + ANCHORED + suffix
        result, count, invalid = wrap_stage_text(text)
        self.assertEqual((count, invalid), (1, 0))
        self.assertEqual(result, '用户正文\n' + STAGE_INTRO + '\n' + WARNING_OPEN + STAGE + WARNING_CLOSE + suffix)

    def test_all_duplicate_occurrences_preserved_and_idempotent(self):
        text = ANCHORED + '\n历史分隔\n' + ANCHORED
        result, count, invalid = wrap_stage_text(text)
        self.assertEqual((count, invalid), (2, 0))
        self.assertEqual(result.count(STAGE), 2)
        self.assertEqual(wrap_stage_text(result), (result, 0, 0))

    def test_unanchored_tags_and_correction_instructions_are_untouched(self):
        text = STAGE + '\nstage纠错重写：stage 默认全错。'
        self.assertEqual(wrap_stage_text(text), (text, 0, 0))

    def test_malformed_or_ambiguous_blocks_are_not_modified(self):
        for block in (STAGE[:-8], '<stage>plain text</stage>',
                      '<stage><act name="角色">unclosed act</stage>',
                      '<stage>' + STAGE + '</stage>',
                      '<stage/>' + STAGE):
            with self.subTest(block=block):
                text = STAGE_INTRO + '\n' + block
                self.assertEqual(wrap_stage_text(text), (text, 0, 1))

    def test_valid_block_after_closed_invalid_block_still_works(self):
        text = STAGE_INTRO + '<stage>invalid</stage>\n' + ANCHORED
        result, count, invalid = wrap_stage_text(text)
        self.assertEqual((count, invalid), (1, 1))
        self.assertTrue(result.startswith(STAGE_INTRO + '<stage>invalid</stage>'))

    def test_inner_optional_fields_and_details_are_preserved(self):
        block = STAGE.replace('now: 示例', '<details><summary>摘要</summary>interplay: 示例</details>')
        result, count, invalid = wrap_stage_text(STAGE_INTRO + '\r\n \t' + block)
        self.assertEqual((count, invalid), (1, 0))
        self.assertEqual(result, STAGE_INTRO + '\r\n \t' + WARNING_OPEN + block + WARNING_CLOSE)

    def test_role_order_and_multimodal_content_preserved(self):
        messages = [{'role': role, 'content': ANCHORED} for role in ('system', 'developer', 'assistant')]
        messages += [{'role': 'user', 'content': [
            {'type': 'text', 'text': ANCHORED, 'cache_control': {'type': 'ephemeral'}},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,example'}},
        ]}, {'role': 'user', 'content': ANCHORED}]
        original = copy.deepcopy(messages)
        self.assertEqual(wrap_stage_messages(messages), (2, 0))
        self.assertEqual(messages[:3], original[:3])
        self.assertEqual(messages[3]['content'][1], original[3]['content'][1])
        self.assertEqual(messages[3]['content'][0]['cache_control'], {'type': 'ephemeral'})
        self.assertEqual(wrap_stage_messages(messages), (0, 0))

    def test_split_or_empty_content_is_not_guessed(self):
        messages = [{'role': 'user', 'content': None}, {'role': 'user', 'content': [
            {'type': 'text', 'text': STAGE_INTRO + '<stage><act name="角色">'},
            {'type': 'text', 'text': '</act></stage>'},
        ]}]
        original = copy.deepcopy(messages)
        self.assertEqual(wrap_stage_messages(messages), (0, 1))
        self.assertEqual(messages, original)
        self.assertEqual(wrap_stage_messages(None), (0, 0))
