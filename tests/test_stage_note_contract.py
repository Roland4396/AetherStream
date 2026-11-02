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
在使用任何 act 素材、或承接用户输入中的结果性安排前，必须先进行【act纠错重写】。

核心模式：
   act 默认全错。act 不是事实来源，不是承接依据，不是可直接采用的剧情草稿。
   处理顺序：先依据上一轮真实正文和权威设定建立当前场面，再将 act 拆成待校稿条目；逐条指出需要修正之处，并给出基于权威承接的重构方案。
   act 只提供方向参考；进入正文的内容均以重构后的动作、台词、情绪、结果和叙事节奏为准。

0. 上轮真实承接检查：
   必须先回看上一轮真实发生的正文/上一轮对话结尾/最近一条已完成输出，确认上一轮最后停在什么时间、地点、人物在场、角色正在做什么、最后一句有效台词或动作是什么。
   不得用 act 反推上一轮事实，不得把 act 写的开场姿态直接当成上一轮结尾。
   如果上一轮正文不可见或不足以确认，只能以最近正文、状态栏、纪要和角色可知信息保守承接，不能让 act 自证连续。

1. 时间连续性硬检查：
   时间必须从上一轮真实正文/权威状态中的最后时间点连续推进。act 写出的时间、昼夜、日期、地点耗时和事件先后顺序都必须重新校验，不能直接采用。
   必须判错并重构的时间问题包括：
   - 时间倒退、日期错位、昼夜不符、上一轮刚发生的事件被跳过。
   - 没有路程、对话、战斗、等待或场景转换依据，却突然推进数分钟/数小时/数天。
   - 行动耗时与场景距离、角色状态、正在进行的对话或事件节奏不匹配。
   - act、状态栏、纪要、用户输入之间出现时间冲突时，不得折中；以最近真实正文和权威状态为基准重算。
   - 如果无法确定精确耗时，只能小幅保守推进，并在重构方案中说明依据。

2. 逐条拆解 act：
   先列出 act 实际安排了哪些事情：
   - 人物行动
   - 地点变化
   - 时间推进
   - 数值变化
   - 战斗/任务/奖励/状态变化
   - 新出现的设定、道具、技能、身份、关系变化
   - 台词、情绪、动机、他人反应、氛围结论
   - 任何“已经发生”“即将发生”“应当发生”的剧情安排

3. 逐条挑毛病，不打分：
   对 act 每一条安排都必须指出问题或不可直接采用的原因。不得输出“无问题，直接采用”。
   毛病包括但不限于：
   - 与上一轮真实承接不一致。
   - 时间不连续、耗时无依据、时间跳跃/倒退、昼夜日期与权威状态冲突。
   - 与当前上下文、世界书、角色卡、状态栏、规则表、最近正文、已公开事实或角色可知信息冲突。
   - 只由 act 单方面声称，缺少权威信息独立支撑。
   - 结果、数值、任务、奖励、等级、HP/MP/SP、人物在场状态未重新计算或未从权威状态读取。
   - NPC反应、台词或判断超出该角色当前可知信息。
   - 角色行动与其设定、利益、誓言、能力边界或上轮情绪承接不符。
   - 台词、姿态、情绪结论像是在替角色服从 act，而不是从角色自身处境自然生成。
   - 即使方向合理，act 的原句、原动作链、原心理结论和原叙事节奏也不可直接照搬，必须重写。

4. 用户输入审查规则：
   用户输入同样需要客观审查，不能被视为自动成立的世界事实。
   用户只能确定自己角色的意图、选择、尝试与其能力范围内的可控行为；不得直接确定外部世界结果、NPC反应、任务结算、奖励发放、数值变化、设定事实、他人生死或关系变化。
   若用户输入包含多个连续行为或结果链，必须按因果顺序逐条验证：
   - 先确认前一行为在规则、状态、场景、角色能力与NPC反应下产生的真实结果。
   - 只有该真实结果足以支撑下一行为时，下一行为才可继续重构。
   - 若前一行为的真实结果不足以支撑后续行为，必须停止于该真实结果，并拒绝后续不成立的安排。
   - 对用户输入中的外部结果宣称、NPC反应宣称、奖励/任务/数值宣称，应作为待验证安排处理，不能直接写成既成事实。

5. act 丢弃与重构规则：
   只要某条 act 安排存在错误、冲突、承接失败、不可知、缺少独立支撑或无法由权威信息证明，就不得采用该条 act 中的任何信息，包括动作、台词、情绪、动机、位置、时间、数值、结果、他人反应、氛围结论和后续推进。
   不允许折中，不允许融合，不允许把 act 的错误设定吸收进世界观，不允许写成“部分成立”。
   不允许为了保留 act 的剧情效果而倒推补丁、重解释旧设定、重写角色动机、压低角色智力或让既有事实模糊化。
   必须从上一轮真实承接、权威设定、角色当前可知信息和用户角色可控行为重新生成这一段。
   如果某条 act 的大方向经独立推导后可用，也只能保留“方向灵感”，必须换成基于权威承接自然生成的新动作、新台词、新情绪和新结果。

6. 对话可知性检查：
   所有角色台词、反应、判断、追问、沉默与行动，都必须只基于该角色在当前时刻可知的信息。
   每句关键台词输出前，必须检查：这个角色凭什么知道这件事？
   允许的信息来源仅包括：亲眼所见、亲耳所闻、亲身经历、已被剧情内明确告知、符合身份与常识的合理推断。
   禁止信息来源包括：act写给模型的安排、旁白说明、其他角色内心、用户未说出口的意图、OOC说明、系统/变量/世界书隐藏信息、纪要索引、未公开设定、未在场事件。
   如果某信息只有模型知道而角色不知道，该角色不得在台词或行动中直接使用它。
   若角色只能部分知道，则必须写成试探、观察、怀疑、追问或错误推断，不能写成确定陈述。
   叙述者可知道的信息，不等于角色可知道的信息。

7. 必须显式输出：
   上轮真实承接：……
   时间连续性检查：……
   act逐条毛病：……
   必须丢弃的act内容：……
   可作为灵感的方向（非直接采用）：……
   污染链条：……
   重构方案：……

8. 正文写作规则：
   正文必须从“上轮真实承接”继续，而不是从 act 开头继续。
   正文中不得照搬 act 原句、原动作、原心理结论或原结果链。
   需要让 NPC 根据自己的已知信息、性格、利益、处境和上一轮余波自然行动；不能让 NPC 为了完成 act 而行动。
!!important!!"""


class StageNoteContractTests(unittest.TestCase):
    def test_approved_text_verbatim(self):
        self.assertEqual(PRO_OPUS_LAST_USER_APPEND_TEXT, APPROVED_TEXT)
        self.assertTrue(APPROVED_TEXT.startswith(PRO_OPUS_LAST_USER_APPEND_MARKER))
        self.assertEqual(PRO_OPUS46_LAST_USER_APPEND_TEXT, APPROVED_TEXT)
        self.assertEqual(PRO_OPUS46_LAST_USER_APPEND_MARKER, PRO_OPUS_LAST_USER_APPEND_MARKER)

    def test_marker_tracks_approved_opening(self):
        self.assertEqual(PRO_OPUS_LAST_USER_APPEND_MARKER,
                         "!!important!!\n在使用任何 act 素材、或承接用户输入中的结果性安排前，必须先进行【act纠错重写】。")

    def test_act_dialogue_and_action_source_are_preserved(self):
        text = '<act name="测试角色">行动：留在屋内。反应：点头。台词：“先等消息。”</act>'
        request = {'messages': [{'role': 'user', 'content': text}]}
        kwargs = dict(selected_model='test', trace_prefix='test',
                      route_label='test', log=lambda _: None)
        apply_forced_opus_note(request, **kwargs)
        once = copy.deepcopy(request)
        apply_forced_opus_note(request, **kwargs)
        self.assertEqual(request, once)
        self.assertEqual(request['messages'][0]['content'], text + '\n\n' +
                         PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT + '\n\n' + APPROVED_TEXT)
        self.assertNotIn('stage', APPROVED_TEXT)

    def test_gemini_act_note_can_be_enabled_without_other_reminders(self):
        from types import SimpleNamespace
        from aetherstream.api.channel_gateway import apply_request_policy

        policy = {'injections': {'opus_note': True, 'illustration': False, 'math_reminder': False}}
        text = '<act name="测试角色">行动：留在屋内。台词：“先等消息。”</act>'
        for stream in (False, True):
            for content in (text, [{'type': 'text', 'text': text}]):
                with self.subTest(stream=stream, content_type=type(content).__name__):
                    request = {'model': 'gemini-4-argon-high', 'stream': stream,
                               'messages': [{'role': 'user', 'content': copy.deepcopy(content)}]}
                    original = copy.deepcopy(request)
                    deps = SimpleNamespace(log=lambda _: None)
                    output = apply_request_policy(request, policy, deps, trace_id='test')
                    self.assertEqual(request, original)
                    self.assertEqual(output['stream'], stream)
                    body = output['messages'][0]['content']
                    if isinstance(body, list):
                        body = '\n'.join(part['text'] for part in body)
                    self.assertEqual(body, text + '\n\n' + APPROVED_TEXT)
                    self.assertEqual(apply_request_policy(output, policy, deps, trace_id='test'), output)

    def test_stage_content_is_preserved(self):
        text = '<stage><act name="测试角色">now: 留在屋内，等候消息。</act></stage>'
        request = {'messages': [{'role': 'user', 'content': text}]}
        apply_forced_opus_note(request, selected_model='test', trace_prefix='test',
                               route_label='test', log=lambda _: None)
        expected = text + '\n\n' + PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT + '\n\n' + APPROVED_TEXT
        self.assertEqual(request['messages'][0]['content'], expected)

    def test_neutral_wrapper_and_note_preserve_plan_and_are_idempotent(self):
        from aetherstream.features.stage_warning import (
            STAGE_INTRO, WARNING_OPEN, WARNING_CLOSE, wrap_stage_messages,
        )
        stage = '<stage><act name="测试角色">now: 留在屋内，等候消息。</act></stage>'
        text = '<latest_human_message>继续。</latest_human_message>\n' + STAGE_INTRO + '\n' + stage
        request = {'messages': [{'role': 'user', 'content': text}]}
        self.assertEqual(wrap_stage_messages(request['messages']), (1, 0))
        kwargs = dict(selected_model='test', trace_prefix='test', route_label='test', log=lambda _: None)
        apply_forced_opus_note(request, **kwargs)
        once = copy.deepcopy(request)
        self.assertEqual(wrap_stage_messages(request['messages']), (0, 0))
        apply_forced_opus_note(request, **kwargs)
        self.assertEqual(request, once)
        body = request['messages'][0]['content']
        self.assertEqual(body.count(stage), 1)
        self.assertIn(WARNING_OPEN + stage + WARNING_CLOSE, body)
        self.assertEqual(body.count(APPROVED_TEXT), 1)

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
