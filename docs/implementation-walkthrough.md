# 从请求到响应：实现阅读路线

这份源码快照用于理解真实代理的处理过程，不附带原部署的 key、账号池、聊天内容和运行状态。下列内容描述代码结构，不代表读者已经具备完整的生产部署。先看 [安全边界](../SECURITY.md)。

## 1. 启动与依赖组装

从 [`proxy.py`](../proxy.py) 进入 [`aetherstream/api/app.py`](../aetherstream/api/app.py)。应用在这里读取环境变量、创建运行时配置、日志器、模型策略、协议依赖和生命周期管理器。

[`runtime/flags.py`](../aetherstream/runtime/flags.py) 读取可热更新的 JSON；公开的 `runtime-flags.example.json` 只有示例，实际部署文件不入库。上游 key 优先通过 `api_key_env` 指定环境变量名，不能从公开示例取得任何可用账号。

## 2. 公共协议入口

- [`api/chat_routes.py`](../aetherstream/api/chat_routes.py)：Chat Completions 主处理链。
- [`api/messages_routes.py`](../aetherstream/api/messages_routes.py)：Anthropic Messages 入口。
- [`api/responses_routes.py`](../aetherstream/api/responses_routes.py)：Responses 入口。
- [`api/protocol_gateway.py`](../aetherstream/api/protocol_gateway.py) 和 [`transforms/downstream_protocols.py`](../aetherstream/transforms/downstream_protocols.py)：公共协议转换与共享调用链。
- [`api/timed_routes.py`](../aetherstream/api/timed_routes.py)：显式限时 SSE 入口，不自动开启模型调用。

公共入口与上游协议不是一回事：客户端使用的协议可以与实际供应商协议不同。

## 3. 选择上游与组装请求

[`routing/model_policy.py`](../aetherstream/routing/model_policy.py) 与 `app.py` 的模型目录、路由选择共同决定目标。不同模型族的兼容逻辑集中在 `features/` 和 `transforms/`。

例如 [`features/glm_thinking.py`](../aetherstream/features/glm_thinking.py) 在实际发送前统一 GLM 5.2 模型族的思考参数，避免仅靠某个渠道前缀；[`features/stage_warning.py`](../aetherstream/features/stage_warning.py) 对符合明确结构的 stage 文本添加提示。它们的作用范围与测试可以直接对照 `tests/` 阅读。

凭据在上游请求头中组装。需要特别区分：供应商 API key、客户端带来的 Authorization、独立管理 token，不应混用；没有配置渠道 key 时的透传行为见安全说明。

## 4. 上游传输与下游交付

`upstreams/` 按协议组织：

- `openai_chat_completions.py`
- `openai_responses.py`
- `gemini_generate_content.py`
- `anthropic_messages/` 的传输、流式转换、收集器、缓存与重放模块

流式转发、非流式收集、保活和重放是不同步骤。已收到下游请求并不代表一定立即调用上游：去重、配置校验、回放等路径可能先返回。关闭连接也不等于供应商必然立即停止计算或计费。

## 5. 日志与拒绝原因

[`observability/logging.py`](../aetherstream/observability/logging.py) 维护请求、文本输出和原始响应日志；[`observability/refusals.py`](../aetherstream/observability/refusals.py) 从上游结构化字段读取拒绝分类和解释。

新日志会显示上游分类、解释及追踪号；上游缺失原因时明确标注，不从提示词或正文猜测。原始诊断日志仍然属于私密数据，不能因为拒绝摘要做了脱敏就把整套日志公开。

## 6. 并发、后台任务和发布

- [`runtime/lifecycle.py`](../aetherstream/runtime/lifecycle.py)：请求、连接、后台任务与排空状态。
- [`runtime/shared.py`](../aetherstream/runtime/shared.py)：跨进程共享状态。
- [`runtime/tasks.py`](../aetherstream/runtime/tasks.py)：请求衍生任务所有权。
- [`features/quota_keeper/`](../aetherstream/features/quota_keeper/)：可选的私有后台功能，示例默认关闭，开启前须理解其上游调用行为。
- [`tools/release.py`](../tools/release.py)：单机 blue/green 发布参考，依赖原部署的配套容器、私网、目录和密钥文件；不是单独克隆即可直接执行的安装器。

固定入口切换新请求，旧实例保留已有连接直到排空。详见 [滚动发布](rolling-releases.md) 与 [开发约束](../AGENTS.md)。

## 7. 用测试理解过程

先构建一个本地开发镜像，再执行完整离线测试：

```bash
docker build -t aetherstream:local .
python3 tools/test_offline.py --image aetherstream:local
```

测试容器禁用网络，不挂载真实凭据。`tools/test_rolling.py` 另建隔离 Docker 网络，验证 SSE/WebSocket 发布、回滚及排空；它不是生产发布命令。

这次公开同步保持应用 Python 文件与运行快照一致；发布整理仅涉及文档、示例、许可证元数据和防止误提交的规则，没有改运行中的服务。
