# AetherStream

AetherStream 是一个面向长会话、角色扮演、Agent、SillyTavern 等场景的多协议 LLM 流式代理。

它提供 OpenAI 兼容的 `/v1/chat/completions` 入口，在内部完成多供应商路由、协议转换、非流式转流式、SSE 重放、长请求保活、上游错误兼容与请求日志追踪。

这个项目最初来自真实高强度长会话工作流：上下文极长、上游不稳定、网关容易超时、不同模型协议不一致、前端必须持续收到流式响应。AetherStream 的目标就是把这些复杂性收敛到一个可控的代理层里。

## 核心能力

- **统一 OpenAI 兼容入口**
  - 对外暴露 `/v1/chat/completions`
  - 兼容常见 OpenAI SDK / 前端 / SillyTavern 接入方式

- **多供应商路由**
  - OpenAI-compatible upstream
  - Gemini native HTTP
  - OpenAI Responses API 风格上游
  - Anthropic Messages 风格上游
  - Claude-like provider shim

- **非流式转流式**
  - 将不支持 stream 的上游响应收集后重放为 OpenAI SSE
  - 用于避免前端、反代或网关长时间无响应导致断连

- **SSE 流式转发与重放**
  - 原生转发上游 SSE
  - 支持将已收集响应按 OpenAI chunk 格式回放
  - 支持请求日志中的 raw SSE 复现调试

- **Anthropic Messages shim**
  - 将 OpenAI chat messages 转为 Anthropic `/v1/messages`
  - 兼容 Claude-like 上游
  - 支持 system blocks、metadata、thinking、tool/context 兼容参数

- **长请求保活**
  - 下游 idle keepalive
  - 非流式 replay keepalive
  - 可选 Claude prompt cache keepalive 辅助逻辑

- **早停机制**
  - 可配置 stop tag
  - 命中特定标记后主动关闭上游连接，减少无效输出与费用

- **关键词 / 上下文过滤**
  - 支持对请求做 provider-specific 清洗
  - 可用于规避某些模型对特定上下文结构的不兼容

- **详细诊断日志**
  - request / response / raw SSE 日志
  - trace id
  - downstream disconnect 追踪
  - upstream incomplete / error event / transport error 定位

## 适用场景

AetherStream 特别适合这些场景：

- SillyTavern / 酒馆长会话
- 角色扮演和世界书上下文很长的请求
- 多模型、多供应商混合路由
- 需要把非流式模型包装成流式输出
- 上游经常断流、504、SSE error，但前端需要稳定体验
- 想统一 OpenAI / Gemini / Anthropic / Responses 风格接口
- 想保留完整请求日志用于排障和复现

## 快速开始

```bash
cp .env.example .env
cp runtime-flags.example.json runtime-flags.json
```

编辑 `.env` 和 `runtime-flags.json`，填入你自己的上游地址与 API Key。

然后启动：

```bash
docker compose -f docker-compose.example.yml up -d --build
```

默认监听：

```text
http://127.0.0.1:3002/v1/chat/completions
```

如果设置了：

```env
FIXED_API_KEY=your-proxy-key
```

客户端需要使用：

```text
Authorization: Bearer your-proxy-key
```

## 配置说明

### 环境变量

参考 `.env.example`。

常用配置：

```env
PORT=3002
DEBUG=false
TIMEOUT=600
FIXED_API_KEY=change-me
LOG_DIR=/app/logs
RUNTIME_FLAGS_PATH=/app/runtime-flags.json
```

OpenAI Responses API：

```env
RESPONSES_API_KEY=
RESPONSES_BASE_URL=https://api.openai.com/v1
GPT_USE_RESPONSES=true
```

Gemini：

```env
GEMINI_ENABLED=false
GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
```

Anthropic / Claude-like：

```env
CLAUDE_API_KEY=
CLAUDE_BASE_URL=https://api.anthropic.com
CLAUDE_ANTHROPIC_VERSION=2023-06-01
CLAUDE_THINKING_TYPE=disabled
```

### runtime-flags.json

`runtime-flags.json` 是运行时配置文件，**默认被 git 忽略**，因为它可能包含 API Key、私有上游地址和运行策略。

请从示例复制：

```bash
cp runtime-flags.example.json runtime-flags.json
```

推荐使用 `api_key_env`，避免在 JSON 中写明文 key：

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

然后在 `.env` 中配置：

```env
MY_PROVIDER_API_KEY=your-real-key
```

## 本地开发

```bash
pip install fastapi uvicorn 'httpx[http2]' docker
uvicorn proxy:app --host 0.0.0.0 --port 3002 --reload
```

语法检查：

```bash
python -m py_compile *.py $(find aetherstream -name '*.py')
```

## 安全注意事项

这是一个代理服务，请把以下内容都视为敏感信息：

- `.env`
- `runtime-flags.json`
- 请求日志
- raw SSE 日志
- 上游错误详情
- prompt / response 内容
- 私有 provider URL

开 issue、发日志或二次分发前，请先脱敏。

本仓库默认忽略：

```text
.env
runtime-flags.json
logs/
*.local.json
*.secret.json
```

## 项目定位

AetherStream 不是一个“只转发请求”的简单 proxy，而是一个面向复杂 LLM 应用的稳定性中间层。

它关心的问题包括：

- 上游协议不一致
- 长请求网关超时
- 模型流式输出不完整
- provider 返回半截 SSE 后再 error
- 非流式接口导致前端等待过久
- 多供应商路由和模型命名混乱
- 调试时缺少可复现 raw SSE

如果你的应用经常被这些问题折磨，AetherStream 可能正好适合。

## License

MIT License，详见 [LICENSE](./LICENSE)。
