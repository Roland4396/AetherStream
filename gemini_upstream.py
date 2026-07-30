"""Legacy wrapper; use gemini_generate_content_upstream for new imports."""

from aetherstream.upstreams.gemini_generate_content import *  # noqa: F401,F403

GeminiUpstreamConfig = GeminiGenerateContentConfig
GeminiUpstreamDeps = GeminiGenerateContentDeps
forward_gemini_stream = forward_gemini_generate_content_stream
collect_gemini_non_stream = collect_gemini_generate_content
