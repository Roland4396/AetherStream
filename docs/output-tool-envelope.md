# Transparent output-tool delivery

The user-selected definition is `actual_content` from request trace `c56858c3`
(`03_input.json`, 2026-11-02). Its source is the enabled TavernHelper script
`🛡️ 防截断（S喵）`, not chatu8's unrelated `agw_emit_*` image-prompt tool.
Do not replace its name, description or schema with a newly authored tool.

Configuration belongs to runtime-flags root `output_tool`. Its model-feature
matcher uses `gemini-3[.]8-flash(?:$|[-_/])`, case-insensitively, for chat upstreams.
It changes no channel selection, request mode, sampling, plot instruction or memory.
Keeping the root config outside legacy channel-policy hashes preserves JSON
locks/cache keys with old slots even during rollback. The selected config is
attached only to the new handler's request-owned `policy.runtime.output_tool`.
The tool declaration, output field, existing control prompt and anchor strings
come from configuration. Explicit caller tool choices and caller-owned tools
remain untouched. No tool execution or additional model round is introduced.

SSE arguments are incrementally decoded before existing content/early-stop
handling; JSON responses are unwrapped before collection/replay. Real tools,
provider errors, token-limit finish reasons, usage and disconnect ownership are
preserved. Existing plain responses still pass through. Delivery-only fields
are excluded from exact-request identities so rollout/toggles do not bypass
existing in-flight locks or cached visible replies.

Raw upstream logs retain their actual tool packets, prefixed with a private
`OUTPUT_TOOL_ENVELOPE` marker. Saved-response replay uses that marker to decode
this proxy-owned envelope; an early-stopped output snapshot remains the authority
for what was delivered. Unmarked OpenAI chat logs also recognize Tavern's exact
`actual_content` / `content` final-reply contract, including split function names,
split arguments and non-streaming completions. This replay-only compatibility
does not take ownership of caller tools on live inference requests.

Unmarked arbitrary or mixed ordinary tool calls fail replay preparation instead
of being silently dropped or executed. Invalid envelope arguments likewise fail
before consuming a once selection, with no fallback inference. Diagnostic
`[empty response]` snapshots are not reply text, and an unmarked envelope's
successfully decoded raw content cannot be overwritten by a stale output file.
Streaming and JSON replay, as well as the replay list preview, use the same parser.

Acceptance uses `tests/test_output_tool.py`, `tests/test_replay_client_tools.py`,
the full offline suite, and isolated
real Nginx SSE/WebSocket promotion/rollback. Never send inference to test this
feature. Deploy an immutable image via the release controller; do not restart an
active service or edit Tavern settings/plugins to activate it.
