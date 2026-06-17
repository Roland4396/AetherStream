import json


def build_openai_sse_error(
    status_code: int,
    message: str,
    error_type: str = "upstream_error",
) -> bytes:
    """构造 OpenAI 风格 SSE error 数据块。"""
    payload = {
        "error": {
            "type": error_type,
            "status": status_code,
            "message": message,
        }
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
