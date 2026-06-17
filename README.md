# Stream Proxy

A pragmatic OpenAI-compatible streaming proxy for multi-provider LLM routing.

It was built for long-running chat / roleplay / agent workflows where upstream
providers differ in protocol, streaming behavior, timeout behavior, and error
format.

## Features

- OpenAI-compatible `/v1/chat/completions` entrypoint.
- Routes requests to OpenAI-compatible, Gemini, Codex/Responses, and Anthropic
  Messages-style upstreams.
- Converts non-stream upstream responses into OpenAI-style SSE streams to avoid
  UI / gateway idle timeouts.
- Replays collected non-stream responses as streams.
- Anthropic Messages shim for Claude-like providers.
- Runtime-configurable upstream directory via `runtime-flags.json`.
- Optional prompt-cache breakpoint / keepalive helpers for Claude-compatible
  providers.
- Optional early-stop tags that close upstream streams after a configured marker.
- Keyword/context filtering hooks for provider-specific compatibility.
- Detailed request / SSE logging for debugging stream truncation, provider
  errors, and downstream disconnects.

## Quick start

```bash
cp .env.example .env
cp runtime-flags.example.json runtime-flags.json
# edit .env and runtime-flags.json with your own upstreams / keys

docker compose -f docker-compose.example.yml up -d --build
```

The proxy listens on:

```text
http://127.0.0.1:3002/v1/chat/completions
```

If `FIXED_API_KEY` is set, clients must use:

```text
Authorization: Bearer <FIXED_API_KEY>
```

## Configuration

### Environment variables

See `.env.example` for the main environment variables.

### Runtime flags

`runtime-flags.json` is intentionally ignored by git because it may contain API
keys and private upstream URLs. Use `runtime-flags.example.json` as a template.

OpenAI-compatible upstreams can be configured like:

```json
{
  "openai_compatible_upstreams": [
    {
      "name": "my-provider",
      "base_url": "https://api.example.com/v1",
      "api_key_env": "MY_PROVIDER_API_KEY"
    }
  ]
}
```

Prefer environment variables for keys. Do not commit real runtime config.

## Development

Run locally:

```bash
pip install fastapi uvicorn 'httpx[http2]' docker
uvicorn proxy:app --host 0.0.0.0 --port 3002 --reload
```

Basic syntax check:

```bash
python -m py_compile *.py
```

## Security note

This project is a proxy. Treat logs and runtime config as sensitive. Request logs
may include prompts, generated text, upstream errors, or provider metadata.

Before publishing forks or bug reports, scrub:

- API keys and bearer tokens
- `runtime-flags.json`
- `.env`
- request logs
- private upstream URLs if needed

## License

Add a license before public release.
