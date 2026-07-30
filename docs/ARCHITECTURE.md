# 架构说明

AetherStream 采用“兼容入口 + 包内模块化”的结构：根目录保留历史模块名作为 wrapper，真实实现迁移到 `aetherstream/` 包内。这样可以继续兼容 `uvicorn proxy:app`、旧 import 路径和现有 Docker 入口，同时逐步拆掉 god module。

## 当前目录结构

```text
proxy.py                              兼容入口，只导出 aetherstream.api.app:app
anthropic_messages_upstream.py        Anthropic Messages 兼容层
chat_completions_upstream.py          OpenAI Chat Completions 兼容层
gemini_generate_content_upstream.py   Gemini GenerateContent 兼容层
responses_upstream.py                 OpenAI Responses 兼容层
anthropic_upstream.py                 旧名称兼容 wrapper
openai_upstream.py                    旧名称兼容 wrapper
gemini_upstream.py                    旧名称兼容 wrapper
codex_upstream.py                     旧名称兼容 wrapper
request_transforms.py                 兼容 wrapper
request_logging.py                    兼容 wrapper
stream_common.py                      兼容 wrapper
stream_state.py                       兼容 wrapper
model_policy.py                       兼容 wrapper

aetherstream/
  api/app.py                          FastAPI composition root、配置与共享服务装配
  api/dependencies.py                 路由显式依赖容器与启动时校验
  api/chat_routes.py                  OpenAI Chat 入口与 provider 编排
  api/messages_routes.py              Anthropic Messages 入口
  api/audio_routes.py                 GPT-SoVITS speech/voice 私有转发入口
  api/admin_routes.py                 replay 管理接口
  api/system_routes.py                health 与模型目录接口
  config/urls.py                      上游 URL 归一化
  routing/model_policy.py             模型族识别、采样/参数兼容策略
  transforms/requests.py              OpenAI Chat -> Anthropic / Responses 转换
  upstreams/openai_chat_completions.py OpenAI Chat Completions stream / collect / replay
  upstreams/anthropic_messages/       Anthropic Messages protocol package
    types.py                          Shared dependency contract
    transport.py                      Connection priming and stream shutdown
    protocol.py                       SSE parsing and OpenAI chunk helpers
    cache.py                          Claude prompt-cache keepalive lifecycle
    messages.py                       Native Anthropic Messages passthrough
    chat_stream.py                    Anthropic -> OpenAI streaming adapter
    collectors.py                     Non-stream response collectors
    legacy_replay.py                  Legacy provider replay compatibility only
  upstreams/gemini_generate_content.py Gemini GenerateContent stream / collect
  upstreams/openai_responses.py       OpenAI Responses stream / collect
  streaming/sse.py                    OpenAI SSE 错误帧等公共工具
  streaming/state.py                  活跃流注册与释放
  streaming/dedupe.py                 非流精确请求合并与短期结果缓存
  observability/logging.py            请求日志、raw SSE、caller fingerprint
  observability/wiretap.py            ASGI 级 downstream disconnect/发送链路追踪
  observability/summaries.py          请求结构摘要、Claude cache breakpoint 摘要
  features/claude_replay.py           跨渠道 replay 控制与日志列表
  features/replay.py                  跨协议日志解析与统一重放服务
  features/request_injections.py      跨渠道共享的项目提示注入策略
  features/drawing_filter.py          绘图上下文清洗
  features/gpt_policy.py              GPT/Responses 策略注入与 prompt cache key
  features/opus_notes.py              Opus 专用追加提示文本
  runtime/docker_control.py           可选 Docker 容器重启封装
  utils/coerce.py                     bool/list/float 配置解析工具
```

## 主要数据流

```text
client
  -> /v1/chat/completions、/v1/messages 或 /v1/audio/*
  -> 统一 trace / caller 生命周期 / replay 拦截
  -> aetherstream.api.chat_routes 路由判断
  -> routing/model_policy 做模型族与兼容参数处理
  -> transforms/requests 做协议转换（按需）
  -> upstreams/* 请求具体 provider
  -> streaming/observability 处理 SSE、日志、断连与 replay
  -> OpenAI-style SSE / JSON 返回给 client
```

音频路径不进入模型路由：Stream 在等待远端首包时同步监听客户端断开，
通过共享网络命名空间内的 SSH 隧道转发到 GPU 主机的私有 Unix socket，
并以二进制流返回，不保存合成音频。

## 设计原则

1. **入口兼容**：根目录 wrapper 不放业务逻辑，保证旧启动方式和旧 import 不崩。
2. **协议边界清晰**：Chat Completions、Responses、Anthropic Messages、Gemini GenerateContent 的上游细节放在 `upstreams/`。
3. **横切能力独立**：日志、wiretap、replay、去重、过滤、策略注入不混在上游实现里。
4. **依赖显式**：路由通过经过校验的依赖容器访问运行时服务，不修改模块 `globals()`。
5. **生产行为优先**：重构以搬迁和边界整理为主，不顺手改业务逻辑。
6. **排障可复现**：继续保留 input/output/raw_sse 日志与 replay 能力。

## 后续重构方向

`aetherstream/api/app.py` 仍承担较多编排职责。后续可以继续拆：

```text
config/settings.py                    环境变量解析与静态配置
routing/openai_compatible.py          /v1/models 动态目录与 pass-through 选择
features/claude_compat.py             Claude client/model compatibility
features/claude_sessions.py           Claude session/user-id 生命周期
```

拆分顺序建议：先迁移纯函数和无状态 service，再缩小 composition root；每一步都要保留 `py_compile`、import smoke、日志复现能力。
