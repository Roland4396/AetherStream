# AetherStream

> **实现过程参考版（2026-10-09）**：这是从实际运行版本整理的源码快照，不包含部署者的 key、账号、聊天日志或运行状态。适合阅读实现、在隔离环境测试；**不是可直接暴露公网的多用户网关**。请先阅读 [安全边界](SECURITY.md)、[实现链路](docs/implementation-walkthrough.md) 和 [本次密钥审计](docs/security-audit-20261009.md)。

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

## 渠道与模型的请求方式

请求方式只有一个选项：**跟随酒馆 / 固定流式 / 固定非流式**。默认跟随酒馆，
不再使用全局“非流转流”或 Claude 全局转换开关。只有渠道设置与该渠道内的精确模型设置
能够改变请求方式；模型设置优先于渠道设置。全局、profile、route 或跨渠道正则不能偷偷开启转换。

私有管理页面为 `/admin/request-modes`，沿用 Stream 私有管理令牌。页面显示实际渠道、
实际模型名、最终方式与设置来源；检查模型仅解析账号池元数据，不发起推理。
设置仍保存在原 `runtime-flags.json`，没有第二份数据库。并发保存采用配置版本检查，
旧页面不能覆盖新设置。只影响新请求，不改变正在生成的请求。

客户端的响应格式始终保持原选择。“固定流式/固定非流式”只规定发给供应商的请求方式，
不会静默更改酒馆开关。配置位置、迁移与兼容细节见 [渠道功能说明](docs/account-channels.md)。

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

### 独立限时 SSE 入口（不自动绑定渠道）

`POST /v1/chat/completions/timed?duration_seconds=290`

请求体与普通 Chat Completions 相同，必须设置 `"stream": true`；认证、模型和
渠道选择继续使用现有逻辑。时间默认 290 秒，允许大于 0、最多 600 秒，从进入
处理函数开始计算，包含请求体读取、路由和等待首个上游数据的时间。

到期取消转发任务，走现有断连清理逻辑关闭上游 HTTP 流，并结束下游 SSE；不伪造
`finish_reason=stop` 或 `[DONE]`。响应头尚未发送时超时返回 504；正常提前完成和
上游错误保持原行为。上游供应商是否在断连后停止计费/计算，取决于供应商。

普通 `/v1/chat/completions` 不受影响；当前没有任何渠道自动启用这个入口。
仅用于手动限时请求，后续再指定哪些渠道接入。

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

### 访问边界

示例 Compose 只把端口绑定到本机回环地址。不要改成公网监听后直接使用。
本快照的 `FIXED_API_KEY` 只是遗留读取项，**没有实现客户端鉴权**；设置它并不能保护 API。
回放管理接口也需要由可信网络或外层代理保护。需要远程访问时，应先在独立反向代理中配置认证和访问控制。
上游 API key 与网关访问凭据是两种不同的凭据；不要把网关管理密钥当作上游 key。

## 配置说明

### 环境变量

参考 `.env.example`。

常用配置：

```env
PORT=3002
DEBUG=false
TIMEOUT=600
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
pip install -r requirements.txt
uvicorn proxy:app --host 127.0.0.1 --port 3002 --reload
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

沿用仓库已有的 GNU GPL v3 许可证，完整条款以 [LICENSE](./LICENSE) 为准；本次发布没有更换许可证。

## AI stage 风险包裹

Chat Completions 普通入口及限时 SSE 入口转发前，统一为 user 消息内具有固定引导语
`以上是用户的本轮输入，以下是stage包含角色行动：` 的完整 `<stage>…</stage>` 添加：

```text
<ai_generated_stage>
以下为 AI 生成的本轮 stage 安排，请注意核对事实性：
<stage>原内容不变</stage>
</ai_generated_stage>
```

不限模型/渠道，stream 与非 stream 均适用；历史重复块分别保留，重复处理不叠加包裹。
只识别引导语后具有命名 act 的非嵌套 stage；残缺/歧义块保持原样，日志仅记录数量。
不修改 scene、recall、纠错指令或非 user 消息；不推断同文块的历史来源。
多模态消息仅处理完整 text 块，不跨内容块猜测边界。原始 SSE 重放不改动。

## Native Claude quota-cycle keeper

The optional `aetherstream.features.quota_keeper` module runs as an asynchronous
background task in the FastAPI lifespan, not a cron/systemd subprocess. It does
not participate in chat routing. A keeper failure is reported as degraded
without making the proxy or `/health` fail.

Enable it through the existing hot-reloaded runtime flags (disabled by default):

```json
"quota_keeper": {
  "enabled": true,
  "check_interval_sec": 60
}
```

The interval is a schedule check, **not a generation frequency** (minimum 60s).
Only enabled Antigravity credentials are considered; duplicate email imports
share one schedule. Only Claude `3p-5h` and `3p-weekly` windows control decisions.
An already-running cycle is left alone regardless of remaining quota; an
exhausted weekly quota waits for its reset. An unused quota response's sliding
`now + 5h` deadline is not treated as an active clock. Before a wakeup, a fresh
successful quota probe must still show an unused available Claude window.

A wakeup uses the exact selected credential and fixed
`claude-opus-4-6-thinking` model, asks only for `OK`, and caps output at 64 tokens
with thinking disabled. There are no generation retries, model/account fallback,
or Gemini generations. HTTP generation errors pause that account; quota lookup
errors have bounded backoff. Viewing status or checking health never probes or
generates upstream.

Deployment settings:

| Environment variable | Purpose |
| --- | --- |
| `QUOTA_KEEPER_STATE_DIR` | Persistent state directory; default `/app/data/quota-keeper` |
| `QUOTA_KEEPER_GPROXY_BASE_URL` | Private gproxy origin; default `http://gproxy:8787` |
| `QUOTA_KEEPER_GPROXY_CREDENTIALS_FILE` | Read-only file containing `GPROXY_ADMIN_USER` and `GPROXY_ADMIN_PASSWORD` |
| `QUOTA_KEEPER_ADMIN_TOKEN_FILE` | Separate private bearer token file (at least 32 characters); absent/invalid means all keeper admin calls are denied |

Mount state read-write and secrets read-only; never bake them into the image,
put credentials in runtime flags, or expose an additional public port. The
keeper accesses gproxy by service DNS and does not need the Docker socket.
Give the container a stop grace period of at least 120 seconds. On shutdown the
keeper stops scheduling new accounts and allows the current operation to finish;
if it must be cancelled, its already-persisted request guard prevents a blind
resubmission on restart.

Private admin endpoints (bearer token required):

- `GET /admin/quota-keeper`: disk-only state, schedule, quota windows, counters
  and failure reasons; no credential tokens or email labels are returned.
- `POST /admin/quota-keeper/credentials/{id}/resume`: clears a keeper pause and
  schedules a re-check, **not** an immediate generation. Existing request guards
  remain intact. Returns 409 during a concurrent state mutation.

State is atomic, mode 0600, protected by a cross-process `run.lock`, and compatible
with the earlier standalone keeper. Unreadable or corrupt state stops keeper work
instead of starting a new empty schedule. For a systemd-to-native cutover, disable
the old timer, let any active oneshot finish, preserve the entire state directory
(including `next_at`, `guard_until`, and paused rows), and mount that same directory
before enabling the native feature. Do not reset paused accounts during migration.
For rollback, first stop the native scheduler/container before re-enabling the
old timer. Logs use the `quota_keeper` marker and never include credentials or
upstream response bodies.

## 本次同步的实现

- `/v1/chat/completions`、`/v1/messages`、`/v1/responses` 的协议入口与转换。
- 请求生命周期、共享状态、单实例后台任务，以及 blue/green 发布与回滚测试。
- GLM 5.2 模型族末端请求参数归一化。
- Claude 拒绝日志中的上游分类、解释与追踪号；未知原因明确标注，不根据提示词猜测。
- 保留相关离线回归测试；测试无需真实 key 或付费模型调用。

## 无损发布与新功能约束

本机生产部署使用固定 Nginx 入口和 blue/green 应用槽位，详见
[发布、回滚与恢复](docs/rolling-releases.md)。所有新增功能须遵守
[生命周期与共享状态约束](AGENTS.md)，不得通过重启活动容器交付更新。

生产发布脚本包含原单机部署的目录、网络和服务命名约定，属于实现参考；单独克隆本仓库并不等于复制完整部署栈。请先阅读脚本并适配自己的环境，不要直接对已有服务执行发布命令。

## Gemini 出站文本扰动（可选）

`injections.gemini_anti_route` 是用户提供的 v3.2 前端脚本的服务端移植。
默认关闭；`enabled: true, mode: "deep"` 保留随机 HTML/Markdown 注释、原脚本
22 组同义句、行尾空格，以及采样参数的微扰。`standard` 保留首部注释和参数处理。
没有 MVU/变量标签豁免：带变量块的消息也参与文本处理；工具、图片和原生签名块
仍作为协议数据保留。它只处理 Gemini 家族或原生 Gemini transport，其他模型不变。

只在最终上游 transport、精确请求去重取得执行权之后修改出站副本；每请求独立状态，
同次请求重试复用同一结果，不像前端事件和 fetch 拦截那样重复叠加。
不改调用者流/非流选择、路由、模型 ID、聊天存档或世界书，不新增后台任务。
采样规则忠实保留原脚本的 `0.1` 下限，温度 `0` 也会升至 `0.1`；已有的
OpenRouter fallback 字段会设为 false，不凭空加入路由字段。

这是文本扰动功能，不是模型身份检测或“防降级”保证。它会增加输入及上下文开销，
也可能降低缓存复用。`gemini_anti_route` 日志只记录修改数量、字符差和模式，
不记录随机串或提示词；其出现仅代表执行了扰动，不能证明真实路由改变。

### 用户输入转写与中性包装

同一开关下的 `input_transform` 可单独开启；其默认关闭，示例配置如下：

```json
{"enabled":true,"mode":"pinyin","tone":"symbol","ratio":40,"languages":["en"],
 "wrapper":true,"padding_chars":32,"preserve_terms":[]}
```

`mode` 支持 `pinyin`、`harmonic`、`fusion`、`mix`（英语/西语词表替换，非语义翻译）。
拼音由固定版本 pypinyin 在本地计算；同音字和混语词表取自用户提供的 v5.1.35。
无运行时 CDN、无额外模型请求、无桌宠、无候选面板、无续写、无剪裁或保存聊天。
只选择最新玩家输入；有 `<latest_human_message>`、`<user_input>` 或 `<input>` 时只处理其内部，
保留后附 stage、格式指令、元数据、工具、图片、JSON、变量协议和代码片段。
排除组装提示末尾已识别的预设控制回合后，按原脚本逆序选择最后一条用户消息；
不要求输入带 XML 标签。重新生成时也选择最后一条已有用户消息。
手机/stage 组装器的“以上是用户的本轮输入”行动页脚也作为显式输入边界，
避免后置 user 角色的图像/格式规则被误当成玩家输入；页脚、行动和场景保持原文。
这里的保留指新增转写层；后续深度扰动仍按其既有策略执行。
没有按变量标签跳过整条消息；同条消息的普通中文仍可转换。不设额外的输入字符上限，
旧 `max_source_chars` 配置不再生效，不因文字长而跳过或截断。
`preserve_terms` 可配置不转写的人名或专名。

包装只用明确标注的转写块与随机注释，不搬运前端的伪审计结论、权限暗示和多层谜题。
本功能仅改变输入表示，不能保证模型恢复所有同音语义，更不保证影响服务端路由。
新增转换仍放在去重取得执行权之后，与原有扰动一起保持每请求一次及重试稳定性。

## 配置驱动的输出工具封装

为指定模型复用已有工具定义，将工具参数透明还原为正文；保持原有请求方式、早停与回放。详见 [输出工具封装](docs/output-tool-envelope.md)。

### Optional Argon weekly cycle

Argon has a separately opt-in weekly schedule under `quota_keeper.argon_weekly`.
It reads only the independent Argon bucket, preserves the Claude policy above,
and uses one tiny exact-account GProxy request only for an unused cycle.
See [configuration, durable guards and recovery](docs/argon-weekly-keeper.md).
