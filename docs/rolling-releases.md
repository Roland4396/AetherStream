# 单机长连接无损发布

## 结构

```text
现有客户端 → stream-proxy（固定 Nginx，原域名/IP/127.0.0.1:3002 不变）
                         └─ stream_backplane 私网 → blue 或 green
                                                   ├─ model_core 上游
                                                   └─ gpu_control / gpu-orchestrator
```

Nginx 使用官方固定 digest；发布时只验证配置并 HUP reload，不重启入口。
新 worker 接收新连接，旧 worker 保留正在处理的 SSE、JSON 长响应、语音响应和 WebSocket。
两个应用槽位共用持久状态。后端只监听 backplane IP，没有新增主机端口。

成熟组件各司其职：Nginx 负责连接交接；Uvicorn/ASGI 负责请求生命周期；FileLock
提供同机进程间锁；DiskCache/SQLite 提供会话、精确响应及取消标记的事务存储。
没有另外引入 Kubernetes，也没有自制网络代理。

## 发布、回滚和恢复

在父仓库运行：

```bash
docker build -t migration-stream-proxy:<唯一版本> stream-proxy
python3 stream-proxy/tools/release.py status
python3 stream-proxy/tools/release.py deploy --image migration-stream-proxy:<唯一版本>
python3 stream-proxy/tools/release.py rollback
python3 stream-proxy/tools/release.py drain --wait 1800
python3 stream-proxy/tools/release.py recover --wait 1800
```

`deploy` 解析镜像 ID，更新**空闲槽位**，等待 `/ready` 和双向状态/API 兼容，
验证 Nginx 配置，切换新请求，再交接后台任务。部署锁拒绝两个操作者同时发布。
`rollback` 复用保留的旧容器/镜像，不重新构建；排空尚未完成时也可以回滚。
`recover` 根据实际入口版本和 fsync 后的 journal 恢复被中断的控制命令。

只有以下条件全部成立才停止旧容器：

1. 所有 HTTP、WebSocket 已完成最终清理；
2. 非流式合并请求、请求衍生后台工作和取消监视器均完成；
3. 旧后台任务完全停止，唯一租约已交给新版本；
4. 旧 Nginx worker 已退出，旧应用没有上游连接；
5. 再次检查仍为空闲。

超时退出码 `2` 表示**排空未完成，旧容器仍然保留运行**，不是让操作者强制清理。
失败退出码 `1` 要先检查 journal、日志和状态。不要用 `docker compose down`、
`restart`、`--force-recreate`、`rm -f` 绕过发布过程。

控制文件位于 `deploy/control/`，journal、后台租约及共享缓存位于 `deploy/state/`；
这些是本机状态，不提交 Git。原 quota-keeper state/锁、运行时开关、日志和凭据仍
使用原路径。跨进程日志锁保证 input/output/raw 三件套不被两个版本交叉覆盖。
旧状态不会因为发版而清空；日志里的常规“检查完成”不再每分钟重复输出，状态变化和错误保留。

## 新功能硬门槛

见 [AGENTS.md](../AGENTS.md)。所有路由自动计数，所有功能有执行范围，所有原生异步
任务有所有权审查。请求衍生任务与全局周期任务分开：前者留在原实例自然完成，后者通过
生命周期租约单实例交接。缓存预热的待执行任务和 latest-wins 取消也可跨版本同步。
新共享状态必须声明版本兼容；不兼容候选拒绝上线，不以删除数据或打断旧请求换取成功。

## 验收和边界

完整离线回归只提取早停配置，不挂载生产凭据/日志/状态，并使用 `--network none`。
两项可重复执行的验收入口为：

```bash
python3 stream-proxy/tools/test_offline.py --image migration-stream-proxy:<唯一版本>
python3 stream-proxy/tools/test_rolling.py --image migration-stream-proxy:<唯一版本>
```

该工具创建独立 internal Docker 网络和临时容器，记录测试结果后清理测试服务；
不连接生产上游，也不触碰生产入口。Docker 和用于网络命名空间检查的本机 sudo 权限须可用。

`/ready` 不调用上游；`/admin/runtime` 以及启用时的 `/admin/runtime/probe/stream`、
`/admin/runtime/probe/ws` 需要独立的管理令牌。探针不进行模型推理，用于在真实代理路径上
验证旧连接持续收到完整、有序的数据，同时新连接已进入新版本。

首装从单进程入口切换到固定 Nginx，需要一次确认空闲后的短暂入口交接。
安装完成后的应用发版/回滚不再替换入口。此设计不承诺机器断电、内核故障、上游自身故障或
稳定入口二进制升级时仍保持连接；跨主机部署也不能直接复用本地文件锁。

参考：[Nginx reload 行为](https://nginx.org/en/docs/control.html)、
[代理模块](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)、
[WebSocket 代理](https://nginx.org/en/docs/http/websocket.html)、
[FileLock](https://py-filelock.readthedocs.io/en/latest/)、
[DiskCache](https://grantjenks.com/docs/diskcache/tutorial.html)。
