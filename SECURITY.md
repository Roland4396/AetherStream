# 安全说明

AetherStream 是一个 LLM 代理层，运行时会接触 API Key、请求正文、模型回复和上游错误信息。

## 不要提交的内容

请不要提交或公开：

- `.env`
- `runtime-flags.json`
- `logs/`
- raw SSE 日志
- 请求输入 / 输出日志
- 真实 API Key、Bearer Token、x-api-key
- 私有上游 URL

## 推荐配置方式

优先使用环境变量保存密钥：

```json
{
  "name": "my-provider",
  "base_url": "https://api.example.com/v1",
  "api_key_env": "MY_PROVIDER_API_KEY"
}
```

然后在 `.env` 中配置：

```env
MY_PROVIDER_API_KEY=your-real-key
```

## 发布前检查

发布前建议执行：

```bash
python -m py_compile *.py
rg -n "sk-|ant-api|pio_sk_|AIza|Bearer |api_key|secret|token" --glob "!logs/**" --glob "!runtime-flags.json" .
```

并人工确认所有命中项都是占位符或代码字段名。
