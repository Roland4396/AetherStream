"""Cross-protocol terminal tool definitions and streaming detectors."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any


TERMINAL_TOOL_NAME = "submit_response_self_evaluation"
TERMINAL_TOOL_HISTORY_EXAMPLE_ID = "call_terminal_history_example"
TERMINAL_TOOL_SYSTEM_PROMPT = (
    "请在本次回复中调用 submit_response_self_evaluation 工具，"
    "使用它完成对本次回复的自我评价。"
)

TERMINAL_TOOL_DESCRIPTION = (
    "Closing self-evaluation tool for this reply. Call it exactly once as the final action after a "
    "non-empty user-visible response in the same assistant message. Never call it as the first or "
    "only content block. After receiving results from ordinary tools, first provide the user-visible "
    "answer or result summary, then call this tool. If other tools are needed, emit those tool calls "
    "first and this call last in the same response, without waiting for tool results. Evaluate this "
    "specific reply and cite concrete statements, examples, or structural choices from it. Generic "
    "filler is invalid."
)

_FIELD_SPECS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "answer_summary",
        {
            "type": "string",
            "minLength": 0,
            "maxLength": 20000,
            "description": (
                "Summarize the user's goal, the approach taken, the core conclusion, and what this "
                "reply actually delivered. 150-300 Chinese characters. BAD: '回答了用户的问题' "
                "(generic). GOOD: '解释了非流式请求为何绕过终止工具，并给出入口与聚合器两处修复'. "
                "Write in Simplified Chinese."
            ),
        },
    ),
    (
        "diagnosis",
        {
            "type": "string",
            "minLength": 0,
            "maxLength": 20000,
            "description": (
                "Evaluate THIS reply under exactly these headings: 优点、准确性与逻辑、完整性与相关性、"
                "清晰度与风格、假设与不确定性、问题与遗漏. Cite concrete claims, examples, or sections "
                "from the reply and explain their impact. 500-900 Chinese characters. BAD: '整体清晰完整，"
                "仍有改进空间' (generic). GOOD: '“非空 tools 会被保留”的结论有日志中的 tool_count=60 "
                "支撑，但没有说明禁用开关仍可移除工具'. Write in Simplified Chinese."
            ),
        },
    ),
    (
        "improvement_plan",
        {
            "type": "string",
            "minLength": 0,
            "maxLength": 20000,
            "description": (
                "Give concrete, executable edits to this reply: what to retain, remove, add, rewrite, or "
                "reorder. 200-400 Chinese characters. BAD: '可以进一步优化表达' (not actionable). "
                "GOOD: '删除第二段重复结论，在首段补充 trace_id，并把未验证判断标为推断'. "
                "Write in Simplified Chinese."
            ),
        },
    ),
    (
        "overall_score",
        {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Integer score from 0 to 100 for this specific reply.",
        },
    ),
    (
        "score_rationale",
        {
            "type": "string",
            "minLength": 0,
            "maxLength": 20000,
            "description": (
                "Explain the main score gains, deductions, and the changes required for a higher score. "
                "150-300 Chinese characters. BAD: '综合评分85分' (no evidence). GOOD: '因准确定位两条失败"
                "路径得分，但缺少部署后验证扣8分；补充健康检查与原始事件证据可提高评分'. "
                "Write in Simplified Chinese."
            ),
        },
    ),
)


def terminal_tool_enabled_for_model(model: Any) -> bool:
    return False


def terminal_tool_parameters() -> dict[str, Any]:
    properties = {
        name: copy.deepcopy(spec)
        for name, spec in _FIELD_SPECS
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name, _spec in _FIELD_SPECS],
        "additionalProperties": False,
    }


def openai_chat_terminal_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TERMINAL_TOOL_NAME,
            "description": TERMINAL_TOOL_DESCRIPTION,
            "parameters": terminal_tool_parameters(),
        },
    }


def openai_responses_terminal_tool() -> dict[str, Any]:
    function = openai_chat_terminal_tool()["function"]
    return {
        "type": "function",
        "name": function["name"],
        "description": function["description"],
        "parameters": function["parameters"],
    }


def anthropic_terminal_tool() -> dict[str, Any]:
    function = openai_chat_terminal_tool()["function"]
    return {
        "name": function["name"],
        "description": function["description"],
        "input_schema": function["parameters"],
    }


def gemini_terminal_declaration() -> dict[str, Any]:
    function = openai_chat_terminal_tool()["function"]
    return {
        "name": function["name"],
        "description": function["description"],
        "parameters": function["parameters"],
    }


def _has_openai_tool(tools: list[Any], name: str) -> bool:
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return True
        if tool.get("type") == "function" and tool.get("name") == name:
            return True
    return False


def terminal_tool_minimal_arguments() -> dict[str, Any]:
    return {
        "answer_summary": "",
        "diagnosis": "",
        "improvement_plan": "",
        "overall_score": 0,
        "score_rationale": "",
    }


def append_openai_terminal_history_example(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict) and messages[index].get("role") == "assistant"
        ),
        None,
    )
    if assistant_index is None:
        return False

    assistant = messages[assistant_index]
    tool_calls = assistant.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
        assistant["tool_calls"] = tool_calls
    if any(
        isinstance(call, dict)
        and isinstance(call.get("function"), dict)
        and call["function"].get("name") == TERMINAL_TOOL_NAME
        for call in tool_calls
    ):
        return False

    tool_calls.append({
        "id": TERMINAL_TOOL_HISTORY_EXAMPLE_ID,
        "type": "function",
        "function": {
            "name": TERMINAL_TOOL_NAME,
            "arguments": json.dumps(
                terminal_tool_minimal_arguments(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    })
    messages.insert(assistant_index + 1, {
        "role": "tool",
        "tool_call_id": TERMINAL_TOOL_HISTORY_EXAMPLE_ID,
        "name": TERMINAL_TOOL_NAME,
        "content": "ok",
    })
    return True


def append_anthropic_terminal_history_example(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict) and messages[index].get("role") == "assistant"
        ),
        None,
    )
    if assistant_index is None:
        return False

    assistant = messages[assistant_index]
    content = assistant.get("content")
    if isinstance(content, str):
        content = ([{"type": "text", "text": content}] if content else [])
    elif isinstance(content, list):
        content = list(content)
    else:
        content = []
    if any(
        isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("name") == TERMINAL_TOOL_NAME
        for block in content
    ):
        return False

    content.append({
        "type": "tool_use",
        "id": TERMINAL_TOOL_HISTORY_EXAMPLE_ID,
        "name": TERMINAL_TOOL_NAME,
        "input": terminal_tool_minimal_arguments(),
    })
    assistant["content"] = content
    messages.insert(assistant_index + 1, {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": TERMINAL_TOOL_HISTORY_EXAMPLE_ID,
            "content": "ok",
        }],
    })
    return True


def prepend_openai_chat_terminal_prompt(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    already_present = any(
        isinstance(message, dict)
        and message.get("role") in {"system", "developer"}
        and message.get("content") == TERMINAL_TOOL_SYSTEM_PROMPT
        for message in messages
    )
    messages = [
        message
        for message in messages
        if not (
            isinstance(message, dict)
            and message.get("role") in {"system", "developer"}
            and message.get("content") == TERMINAL_TOOL_SYSTEM_PROMPT
        )
    ]
    messages.insert(0, {"role": "system", "content": TERMINAL_TOOL_SYSTEM_PROMPT})
    payload["messages"] = messages
    return not already_present


def prepend_anthropic_terminal_prompt(payload: dict[str, Any]) -> bool:
    system = payload.get("system")
    if isinstance(system, list):
        blocks = list(system)
    elif isinstance(system, str) and system:
        blocks = [{"type": "text", "text": system}]
    elif isinstance(system, dict):
        blocks = [system]
    else:
        blocks = []

    already_present = any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("text") == TERMINAL_TOOL_SYSTEM_PROMPT
        for block in blocks
    )
    blocks = [
        block
        for block in blocks
        if not (
            isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text") == TERMINAL_TOOL_SYSTEM_PROMPT
        )
    ]
    blocks.insert(0, {"type": "text", "text": TERMINAL_TOOL_SYSTEM_PROMPT})
    payload["system"] = blocks
    return not already_present


def inject_openai_chat_terminal_tool(payload: dict[str, Any]) -> bool:
    if not terminal_tool_enabled_for_model(payload.get("model")):
        return False
    prepend_openai_chat_terminal_prompt(payload)
    append_openai_terminal_history_example(payload)
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []
        payload["tools"] = tools
    already_present = _has_openai_tool(tools, TERMINAL_TOOL_NAME)
    tools[:] = [
        tool
        for tool in tools
        if not (
            isinstance(tool, dict)
            and (
                (
                    isinstance(tool.get("function"), dict)
                    and tool["function"].get("name") == TERMINAL_TOOL_NAME
                )
                or (tool.get("type") == "function" and tool.get("name") == TERMINAL_TOOL_NAME)
            )
        )
    ]
    tools.append(openai_chat_terminal_tool())
    payload["tool_choice"] = "auto"
    return not already_present


def inject_anthropic_terminal_tool(payload: dict[str, Any]) -> bool:
    if not terminal_tool_enabled_for_model(payload.get("model")):
        return False
    prepend_anthropic_terminal_prompt(payload)
    append_anthropic_terminal_history_example(payload)
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []
        payload["tools"] = tools
    already_present = any(
        isinstance(tool, dict) and tool.get("name") == TERMINAL_TOOL_NAME
        for tool in tools
    )
    tools[:] = [
        tool
        for tool in tools
        if not (isinstance(tool, dict) and tool.get("name") == TERMINAL_TOOL_NAME)
    ]
    tools.append(anthropic_terminal_tool())
    payload["tool_choice"] = {
        "type": "any",
        "disable_parallel_tool_use": False,
    }
    return not already_present


def convert_openai_chat_tools_to_responses(tools: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            converted.append(copy.deepcopy(tool))
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            if tool.get("name"):
                converted.append(copy.deepcopy(tool))
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        item: dict[str, Any] = {"type": "function", "name": name}
        for source, target in (
            ("description", "description"),
            ("parameters", "parameters"),
            ("strict", "strict"),
        ):
            if source in function:
                item[target] = copy.deepcopy(function[source])
        converted.append(item)
    return converted


def convert_openai_chat_tools_to_anthropic(tools: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        item: dict[str, Any] = {
            "name": name,
            "input_schema": copy.deepcopy(function.get("parameters") or {"type": "object"}),
        }
        if function.get("description"):
            item["description"] = str(function["description"])
        converted.append(item)
    return converted


def convert_openai_chat_tools_to_gemini(tools: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        item: dict[str, Any] = {
            "name": name,
            "parameters": copy.deepcopy(function.get("parameters") or {"type": "object"}),
        }
        if function.get("description"):
            item["description"] = str(function["description"])
        converted.append(item)
    return converted


@dataclass(frozen=True)
class OpenAIChatToolDecision:
    has_tool_data: bool = False
    defer: bool = False
    terminal: bool = False
    invalid_terminal: bool = False
    ordinary_seen: bool = False


class OpenAIChatTerminalDetector:
    """Classify fragmented Chat Completions Tool Calls without leaking a terminal call."""

    def __init__(self, target_name: str = TERMINAL_TOOL_NAME):
        self.target_name = target_name
        self._names: dict[tuple[int, int], str] = {}
        self._seen_keys: set[tuple[int, int]] = set()
        self._ordinary_keys: set[tuple[int, int]] = set()
        self._terminal_keys: set[tuple[int, int]] = set()

    @property
    def ordinary_seen(self) -> bool:
        return bool(self._ordinary_keys)

    def feed(self, data: Any) -> OpenAIChatToolDecision:
        if not isinstance(data, dict):
            return OpenAIChatToolDecision(ordinary_seen=self.ordinary_seen)
        choices = data.get("choices")
        if not isinstance(choices, list):
            return OpenAIChatToolDecision(ordinary_seen=self.ordinary_seen)
        has_tool_data = False
        for choice_pos, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            choice_index = int(choice.get("index") or choice_pos)
            container = choice.get("delta")
            if not isinstance(container, dict):
                container = choice.get("message")
            if not isinstance(container, dict):
                continue
            calls = container.get("tool_calls")
            if not isinstance(calls, list):
                legacy = container.get("function_call")
                calls = [{"index": 0, "function": legacy}] if isinstance(legacy, dict) else []
            for call_pos, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                has_tool_data = True
                call_index = int(call.get("index") or call_pos)
                key = (choice_index, call_index)
                self._seen_keys.add(key)
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                fragment = function.get("name")
                if not isinstance(fragment, str) or not fragment:
                    continue
                previous = self._names.get(key, "")
                if fragment == self.target_name or (previous and fragment.startswith(previous)):
                    current = fragment
                else:
                    current = previous + fragment
                self._names[key] = current
                if current == self.target_name:
                    self._terminal_keys.add(key)
                    self._ordinary_keys.discard(key)
                elif not self.target_name.startswith(current):
                    self._ordinary_keys.add(key)
                    self._terminal_keys.discard(key)

        unresolved = self._seen_keys - self._ordinary_keys - self._terminal_keys
        invalid_terminal = bool(self._terminal_keys and self._ordinary_keys)
        terminal = bool(self._terminal_keys) and not self._ordinary_keys and not unresolved
        defer = has_tool_data and bool(unresolved) and not self._ordinary_keys
        return OpenAIChatToolDecision(
            has_tool_data=has_tool_data,
            defer=defer,
            terminal=terminal,
            invalid_terminal=invalid_terminal,
            ordinary_seen=self.ordinary_seen,
        )


def is_terminal_responses_event(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    item = data.get("item")
    if isinstance(item, dict) and item.get("type") == "function_call":
        return item.get("name") == TERMINAL_TOOL_NAME
    response = data.get("response")
    if isinstance(response, dict):
        for output in response.get("output", []) or []:
            if (
                isinstance(output, dict)
                and output.get("type") == "function_call"
                and output.get("name") == TERMINAL_TOOL_NAME
            ):
                return True
    return data.get("type") == "function_call" and data.get("name") == TERMINAL_TOOL_NAME


def is_terminal_anthropic_event(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    block = data.get("content_block")
    if isinstance(block, dict) and block.get("type") == "tool_use":
        return block.get("name") == TERMINAL_TOOL_NAME
    return data.get("type") == "tool_use" and data.get("name") == TERMINAL_TOOL_NAME


def is_terminal_gemini_chunk(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    for candidate in data.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            call = part.get("functionCall") or part.get("function_call")
            if isinstance(call, dict) and call.get("name") == TERMINAL_TOOL_NAME:
                return True
    return False
