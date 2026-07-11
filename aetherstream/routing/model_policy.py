from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPolicy:
    codex_models: frozenset[str]
    allowed_gpt_models: frozenset[str]
    allowed_gemini_models: frozenset[str]
    allowed_claude_models: frozenset[str]
    codex_unsupported_fields: frozenset[str]

    def is_gemini_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        model_lower = model_name.lower()
        # Explicit provider prefixes must win over model-family heuristics.
        # free/xxx is routed by the stream-proxy free-channel branch before the
        # built-in Gemini branch, so do not let it fall into direct Gemini HTTP.
        if '/' in model_lower:
            prefix, _, _ = model_lower.partition('/')
            if prefix in ('free', 'codecli'):
                return False
        return model_lower.startswith("gemini-") or "gemini" in model_lower

    def is_codex_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        return model_name in self.codex_models

    def is_claude_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        name = model_name.lower()
        # Provider prefixes are explicit routing hints.
        # free/ only uses Anthropic when the real model is Claude.  Non-Claude
        # free models are OpenAI-compatible through the Pioneer/account pool.
        # codecli remains restricted to Claude-looking model ids.
        if '/' in name:
            prefix, _, rest = name.partition('/')
            if prefix == 'free':
                return self.is_claude_family(rest)
            if prefix == 'codecli':
                return self.is_claude_family(rest)
            return False
        return self.is_claude_family(name)

    def is_claude_family(self, model_name: str) -> bool:
        if not model_name:
            return False
        name = model_name.lower()
        return name.startswith("claude-") or "/claude-" in name or "anthropic/claude-" in name

    def is_gpt_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        return model_name.lower().startswith("gpt-")

    def is_model_allowed(self, model_name: str) -> bool:
        # Do not hard-reject unknown upstream model ids. The allow-lists are used
        # for /v1/models exposure, while routing should pass through model ids so
        # the actual upstream can decide support.
        return True

    def apply_codex_reasoning(self, request_data: dict) -> list[str]:
        removed_fields = []
        for field in self.codex_unsupported_fields:
            if field in request_data:
                request_data.pop(field, None)
                removed_fields.append(field)
        return removed_fields

    def apply_claude_sampling_compat(self, request_data: dict) -> list[str]:
        removed_fields = []
        if not self.is_claude_model(str(request_data.get("model", ""))):
            return removed_fields
        if "temperature" in request_data and "top_p" in request_data:
            request_data.pop("top_p", None)
            removed_fields.append("top_p")
        return removed_fields
