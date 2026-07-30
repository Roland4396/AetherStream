# Current Feature Map

This document describes the current behavior boundaries of AetherStream. It is
an implementation map, not a list of planned features.

## Public APIs

| Endpoint | Contract |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat JSON or SSE |
| `POST /v1/messages` | Anthropic Messages passthrough |
| `POST /v1/audio/speech` | Private OpenAI-compatible GPT-SoVITS audio stream |
| `GET /v1/audio/voices` | Available private GPT-SoVITS voice catalog |
| `GET /v1/models` | Local and dynamic upstream model directory |
| `GET /health` | Process liveness |
| `GET/POST /admin/replay` | Saved-response replay control |
| `GET /admin/replay/log-output` | Saved normalized output |

The legacy `/admin/claude-replay` paths remain aliases for existing clients.

## Routing

The chat endpoint currently supports these route families:

1. Local diagnostic slow streams.
2. Explicit `free/` account-pool models over OpenAI Chat.
3. Models discovered from configured OpenAI-compatible `/v1/models` endpoints.
4. Native Gemini HTTP.
5. Claude and `codecli/` models over Anthropic Messages.
6. GPT models over Responses or OpenAI Chat.
7. Explicit OpenAI Responses models when configured.
8. Model-family fallback to a configured OpenAI-compatible upstream.

Saved-response replay is an entry interceptor. It runs before this routing list,
so adding a provider route does not require replay integration.

## Protocol Adapters

| Upstream protocol | Request direction | Response direction |
| --- | --- | --- |
| OpenAI Chat | passthrough | SSE forward or JSON collection |
| Anthropic Messages | Chat conversion or passthrough | OpenAI SSE/JSON normalization |
| Gemini native | Chat text conversion | OpenAI SSE/JSON normalization |
| OpenAI Responses | Chat conversion | OpenAI SSE/JSON normalization |

The execution helpers also support true non-stream upstream calls replayed as
SSE with keepalives.

## Request Policies

- GPT usage-policy system injection and prompt-cache key selection.
- GPT reasoning effort and Responses request conversion.
- Claude Code headers, system prefixes, beta flags, sessions, and model compatibility.
- Claude prompt-cache breakpoints and optional cache keepalive requests.
- Claude output effort, thinking, tools, context-management, and keyword filtering.
- Pioneer Opus note injection controlled by runtime flags.
- Direct Opus note injection.
- Shared assistant-prefill continuation with CG asset constraints.
- Drawing-context filtering for selected Gemini, Haiku, DeepSeek, and pro routes.
- Pro-route reasoning suppression.
- Provider-specific sampling compatibility.

Project prompt injections live in `features/request_injections.py` and
`features/opus_notes.py`; API routes should not contain copies of those texts.

## Reliability

- Downstream keepalives while waiting for headers or SSE lines.
- Upstream connection priming for selected Claude/Gemini-compatible hosts.
- Gemini retry and timeout settings.
- Exact non-stream request coalescing with a short result cache.
- Downstream disconnect detection at the ASGI boundary.
- Idempotent caller lifecycle release independent of provider implementation.
- Partial-output persistence before slow upstream shutdown.
- Optional early-stop tags and upstream cancellation.
- TTS downstream-disconnect cancellation while waiting for upstream headers.
- Optional companion SillyTavern restart after Gemini failures.

## Logs

Each saved request can contain normalized input metadata, outbound request,
assistant output, raw upstream SSE or JSON, trace data, and replay metadata.

The local store uses ten rotating slots. Slot writes are serialized and atomic.
When a reused slot has no raw response, any raw file from the previous occupant
is removed so input and raw response cannot be paired incorrectly.

## Saved-Response Replay

The unified replay parser recognizes OpenAI Chat SSE and JSON, Anthropic
Messages SSE, OpenAI Responses SSE, Gemini native SSE, and normalized output
text as a final fallback.

Both stream and non-stream clients use the same normalized replay record.
Interrupted logs with usable output are replayed as snapshots and receive a
synthetic normal finish. The admin list reports source format, source completion,
snapshot status, output length, and whether the artifact is usable.

`once` is atomically claimed after the selected artifact parses successfully and
before response delivery starts. Concurrent requests cannot claim the same
selection twice. Claiming also verifies that the selected file did not change.
Malformed enabled control state fails closed instead of falling through to a
paid upstream.

## Runtime Configuration

Environment variables provide process-level defaults and credentials.
`runtime-flags.json` is polled for provider directory entries and hot policy
switches. Runtime configuration must not be copied into public diagnostics
because it can contain API keys and private endpoints.
