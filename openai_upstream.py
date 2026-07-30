"""Legacy wrapper; use chat_completions_upstream for new imports."""

from aetherstream.upstreams.openai_chat_completions import *  # noqa: F401,F403

OpenAIUpstreamDeps = ChatCompletionsUpstreamDeps
forward_stream = forward_chat_completions_stream
collect_stream = collect_chat_completions_stream
collect_non_stream = collect_chat_completions_nonstream
forward_non_stream_as_openai_stream = replay_chat_completions_nonstream_as_stream
replay_openai_chat_stream = replay_chat_completions_stream
