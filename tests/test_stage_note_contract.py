import copy
import unittest

from aetherstream.features.opus_notes import (
    PRO_OPUS_LAST_USER_APPEND_MARKER,
    PRO_OPUS_LAST_USER_APPEND_TEXT,
    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
    PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT,
    PRO_OPUS46_LAST_USER_APPEND_MARKER,
    PRO_OPUS46_LAST_USER_APPEND_TEXT,
)
from aetherstream.features.request_injections import apply_forced_opus_note


APPROVED_TEXT = """!!important!!
stage 是 AI 生成的剧情草案。以上一轮正文和已确认状态为依据，结合角色设定，检查场景承接、人物知情与行动因果。

核对所有在场及本轮相关 NPC 的位置、状态、行动和反应，补齐遗漏。

指出具体错误并修正对应部分，其余安排按规划推进。

简述必要修正，继续正文。
!!important!!"""


class StageNoteContractTests(unittest.TestCase):
    def test_approved_text_verbatim(self):
        self.assertEqual(PRO_OPUS_LAST_USER_APPEND_TEXT, APPROVED_TEXT)
        self.assertTrue(APPROVED_TEXT.startswith(PRO_OPUS_LAST_USER_APPEND_MARKER))
        self.assertEqual(PRO_OPUS46_LAST_USER_APPEND_TEXT, APPROVED_TEXT)
        self.assertEqual(PRO_OPUS46_LAST_USER_APPEND_MARKER, PRO_OPUS_LAST_USER_APPEND_MARKER)

    def test_last_user_injection_is_exact_and_idempotent(self):
        for content in (
            '<latest_human_message>test</latest_human_message>',
            'plain test',
            [{'type': 'text', 'text': '<latest_human_message>test</latest_human_message>'}],
        ):
            with self.subTest(content_type=type(content).__name__):
                request = {'messages': [
                    {'role': 'system', 'content': 'unchanged system'},
                    {'role': 'user', 'content': 'unchanged history'},
                    {'role': 'assistant', 'content': 'unchanged reply'},
                    {'role': 'user', 'content': copy.deepcopy(content)},
                    {'role': 'assistant', 'content': 'unchanged prefill'},
                ]}
                before = copy.deepcopy(request)
                kwargs = dict(selected_model='claude-opus-4-6', trace_prefix='test',
                              route_label='test', log=lambda _: None)
                apply_forced_opus_note(request, **kwargs)
                once = copy.deepcopy(request)
                apply_forced_opus_note(request, **kwargs)
                self.assertEqual(request, once)
                self.assertEqual(request['messages'][:3], before['messages'][:3])
                self.assertEqual(request['messages'][-1], before['messages'][-1])
                body = request['messages'][3]['content']
                if isinstance(body, list):
                    body = '\n'.join(x['text'] for x in body if x.get('type') == 'text')
                self.assertEqual(body.count(APPROVED_TEXT), 1)
                self.assertNotIn('<disclaimer>', body)
                self.assertTrue(body.endswith(APPROVED_TEXT))
                self.assertIn(PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT, body)
                self.assertEqual(body.count(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER), 3)


class UserPromptPreservationTests(unittest.TestCase):
    def test_user_supplied_disclaimer_is_not_removed(self):
        text = '<disclaimer>User-supplied text</disclaimer>'
        request = {'messages': [{'role': 'user', 'content': text}]}
        apply_forced_opus_note(request, selected_model='test', trace_prefix='test',
                               route_label='test', log=lambda _: None)
        expected = text + '\n\n' + PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT + '\n\n' + PRO_OPUS_LAST_USER_APPEND_TEXT
        self.assertEqual(request['messages'][0]['content'], expected)
        self.assertEqual(request['messages'][0]['content'].count('<disclaimer>'), 1)


if __name__ == '__main__':
    unittest.main()
