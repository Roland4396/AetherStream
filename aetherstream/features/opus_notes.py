"""Optional Opus roleplay note injection constants."""

PRO_OPUS_LAST_USER_APPEND_TEXT = """!!important!!
stage 是 AI 生成的剧情草案。以上一轮正文和已确认状态为依据，结合角色设定，检查场景承接、人物知情与行动因果。

核对所有在场及本轮相关 NPC 的位置、状态、行动和反应，补齐遗漏。

指出具体错误并修正对应部分，其余安排按规划推进。

简述必要修正，继续正文。
!!important!!"""
PRO_OPUS_LAST_USER_APPEND_MARKER = "!!important!!\nstage 是 AI 生成的剧情草案。"
PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER = "（记得插图。）"
PRO_OPUS_LAST_USER_ILLUSTRATION_TEXT = (
    "!!important!!\n" + PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER * 3 + "\n!!important!!"
)

# Backward-compatible names for older call sites/log wording.
PRO_OPUS46_LAST_USER_APPEND_TEXT = PRO_OPUS_LAST_USER_APPEND_TEXT
PRO_OPUS46_LAST_USER_APPEND_MARKER = PRO_OPUS_LAST_USER_APPEND_MARKER
