"""Legacy wrapper; use anthropic_messages_upstream for new imports."""

from aetherstream.upstreams.anthropic_messages import *  # noqa: F401,F403

AnthropicUpstreamDeps = AnthropicMessagesDeps
forward_anthropic_stream = forward_anthropic_messages_stream
collect_anthropic_message_response = collect_anthropic_messages_response
forward_anthropic_chat_stream = forward_anthropic_messages_as_chat_stream
collect_anthropic_chat_completion = collect_anthropic_messages_as_chat_completion
