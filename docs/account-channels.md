# 渠道选择与请求功能的职责边界

## Pool 选择渠道，Stream 执行功能

- `account-pool-proxy/pools.json.routing`（schema 2）只声明渠道 ID、每渠道自己的前缀/匹配规则、公开模型到上游模型的映射、协议、动态目录源和账号组。**渠道选择始终调用 Pool**；Stream 不按前缀另建一套选择器。
- `stream-proxy/runtime-flags.json` 保存 Stream 功能。全局触发词在 `early_stop.tags`；非模式的去重/保活设置在 `streaming`；原 Claude 设置在 `claude`；详细注入设置在 `injections`。
- `channel_policies` 是同一 runtime-flags 文件里的按渠道/路由 ID 索引的功能覆盖。它不允许 `match`、`models`、`strip_prefix`、`accounts` 或 `transport` 等选择字段。
- Pool 返回已选中的 `channel`、`route`、`transport` 和 `upstream_model`。Stream 根据这些结果合并自己的功能配置，不让 Pool 返回的 feature policy 或早停开关取得控制权。

## Stream 功能合并

以下多层合并只适用于早停、注入等其他功能。**请求方式不参与这些层次的继承**，单独采用下节的两层规则。

全局设置与 `channel_policies.defaults` → 渠道 profiles → 渠道 policy → 路由 profiles → 路由 policy → model_rules。对象递归合并，标量/数组替换；false 与空数组有效。

`model_rules` 只决定所选渠道的功能差异，不参与渠道选择。`account_overrides` 根据 Pool 最终返回的实际账号 ID 在 Stream 内覆盖早停设置，保持原有账号差异；Pool 不存储/发送早停策略。

`channel_policies` 的简化结构：

```json
{"schema":1,"defaults":{"injections":{"opus_note":false}},"profiles":{},"model_rules":[],"channels":{"example":{"policy":{"streaming":{"force_upstream":"auto"}},"routes":{"example-chat":{"policy":{}}}}},"account_overrides":{}}
```

每次请求固定一份配置快照。编辑触发词或功能配置影响新请求，不改动进行中的请求。Pool 新增渠道/路由无需同步登记第二份准入名单；没有专属覆盖时默认跟随调用方，不改变 Pool 的渠道选择。

## 请求方式：只有渠道与精确模型两层

默认行为固定为**跟随酒馆**，不是可被暗中开启的全局转换规则。唯一可设置的字段是
`streaming.force_upstream`，只允许位于渠道 policy 或该渠道内的精确模型规则。

| 值 | 界面名称 | 酒馆非流请求发给供应商 | 酒馆流式请求发给供应商 |
|---|---|---|---|
| `auto` | 跟随酒馆 | 非流 | 流式 |
| `stream` | 固定流式 | 流式 | 流式 |
| `json` | 固定非流式 | 非流 | 非流 |

渠道设置位于 `channel_policies.channels.<channel>.policy.streaming.force_upstream`。
模型覆盖使用 `model_rules`，必须指定**一个渠道、精确实际模型名**：

```json
{"match":{"exact":["actual-model"]},"channels":["example"],"policy":{"streaming":{"force_upstream":"json"}}}
```

优先级仅为：未设置时跟随酒馆 → 渠道 → 此渠道内的精确模型。
模型明确选 `auto` 会覆盖渠道固定模式；界面选“使用渠道设置”则删除该模型覆盖。
同一渠道的多个公开别名如果指向同一个实际模型，共用该模型设置；检查页明确显示实际模型名。
冲突的模型规则直接拒绝，不靠规则顺序偷偷决定。流模式不能藏在 profile、route、
全局默认或跨渠道正则里；其他功能的已有继承与账号覆盖不变。

### 管理与可观测性

- 私有入口 `/admin/request-modes`，不新增公共端口。数据与写入接口都要求独立的 Stream 管理令牌，不能使用普通推理 API key。
- 浏览器只在当前页面内存保存令牌，不放进 URL、cookie、localStorage 或 sessionStorage。配置响应不包含供应商凭据。
- 页面可选择渠道模式、检查公开模型的真实路由、设置/移除精确模型覆盖；检查仅调用元数据，不触发推理或 GPU。
- 所有写入回到同一个 `runtime-flags.json`，使用完整配置版本和跨进程内核文件锁，保留单文件 bind mount 的 inode。读取/就绪检查遵守同一个读锁。
- 保存前验证且备份原文件，失败时回滚；仅保留十份私有备份。过期页面收到冲突提示，不覆盖其他设置。
- 每次真实请求打印 `request_mode client_stream=... upstream_stream=... mode=... source=... channel=... upstream_model=...`。来源只可能是默认、渠道、模型，不再凭 `nonstream` 日志标签猜测上游方式。
- 保存只改变新请求的快照；SSE、JSON、音频、WebSocket 与后台任务的排空协议不变。没有自动重放推理。

### 迁移和回滚

`tools/migrate_request_modes.py` 只生成候选文件，不直接覆盖运行配置。它将已有 profile/route 的
单一一致模式展开到明确渠道；同渠道路由冲突或仍有全局强制转换时拒绝猜测。
原订阅 Responses 的明确流式要求迁为该渠道的 `stream`，不是全局例外。

先使用 `--keep-inert-legacy` 生成 expand 配置，再滚动更新。旧实例排空前，兼容配置中的
`auto/false` 只能保持不转换，不能重新开启旧的布尔开关。两槽均采用新读取器后，删除这些
遗留字段。JSON 共享请求的哈希使用固定的旧字段投影保持锁键兼容；该投影**只用于计算锁键**，
不参与请求方式决策，不是另一个设置入口。

旧的 `claude.stream.nonstream_to_stream` 不再控制后备协议适配器；未绑定明确请求策略的
非流请求也保持非流。历史日志不删除，本次没有修改重复写日志的行为。

## 真实断开与共享请求

下游断开会取消并等待上游工作，关闭实际 HTTP 连接；上游 SSE EOF/连接错误结束下游并清理任务，不自动重试。早停仍使用统一的真实断开实现，触发词不由本次迁移修改。

Anthropic SSE 的可见文本统一从起始消息内容、起始文本块和后续文本增量提取；首段即命中或跨事件拼成触发词时，沿用同一检测与真实关闭路径。账号级开关和内容保护规则不变。普通运行日志保留早停关闭摘要（响应/客户端是否已关闭、耗时及计数），不在该摘要中记录正文或触发词。

JSON 精确去重保护仍在等待的其他客户端；**最后一个真实等待者离开后取消上游，不继续无人等待的后台生成**。跨进程等待者使用内核文件锁，不用过期 TTL 授权重复工作。取消结果不缓存。SSE 不进入此合并屏障。

## 滚动迁移

先给 Pool 的兼容描述符增加上游模型标识（不改选择），再给 Stream 增加本地功能配置并滚动发布。旧 Stream 排空后，将 Pool 配置收缩为 selection-only schema 2，移除 defaults/profiles/model_rules/dedupe 和账号 feature policy。schema 1 读取代码只用于旧实例排空/回滚兼容；新配置不使用它。

Stream 的精确请求哈希兼容数据迁入 `channel_policies`，保持现有共享锁键/缓存包装。不会从迁移快照覆盖 Pool 账号状态、统计、凭据或人工禁用。Pool 更新只在连接和请求空闲时进行；Stream 只用不可变镜像及 `tools/release.py deploy`，不重启稳定 Nginx 入口。
