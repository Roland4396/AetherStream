"""Legacy wrapper; use responses_upstream for new imports."""

from aetherstream.upstreams.openai_responses import *  # noqa: F401,F403

CodexUpstreamDeps = ResponsesUpstreamDeps
forward_codex_chat_stream = forward_responses_as_chat_stream
collect_codex_chat_completion = collect_responses_as_chat_completion
collect_codex_response_nonstream = collect_responses_nonstream
replay_codex_chat_stream = replay_responses_as_chat_stream
