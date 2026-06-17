# 运维与排障

## 启动

```bash
cp .env.example .env
cp runtime-flags.example.json runtime-flags.json
docker compose -f docker-compose.example.yml up -d --build
```

## 修改与部署约定

本项目在生产环境中遵循 **改代码不等于重建** 的约定：

- 可以提交代码、推送 GitHub、运行静态检查。
- 不要在未获得明确授权时执行 `docker build`、`docker compose build`、`docker compose up -d`、`docker restart`。
- 只有用户明确说“重建”、“现在重建”、“执行重建”时，才允许重建或重启运行中的容器。
- 如果用户说“先别重建 / 不重建 / 不重启”，即使改完也只能汇报等待，不能执行任何部署命令。

推荐流程：

```bash
python -m py_compile *.py $(find aetherstream -name '*.py')
git status
git commit
git push
# 等待明确授权后再重建
```

## 查看日志

```bash
docker logs -f stream-proxy
```

## 常见问题

### 上游返回 error event

部分 provider 会在 SSE 中返回：

```text
event: error
data: {...}
```

AetherStream 会记录 raw SSE，并尽量保留已经收到的正文。

### 长请求前端超时

可以启用非流式转流式或 keepalive，让下游持续收到 chunk。

### 需要复现一次输出

查看 `logs/` 中保存的 input/output/raw_sse 文件，必要时使用 replay 工具重放。
