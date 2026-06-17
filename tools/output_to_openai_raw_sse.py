#!/usr/bin/env python3
"""Convert saved *_output.txt response text to OpenAI-compatible *_raw_sse.txt."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import uuid
from pathlib import Path


def _read_output(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    sep_index = None
    for idx, line in enumerate(lines[:20]):
        stripped = line.strip()
        if stripped and set(stripped) == {"="}:
            sep_index = idx
            break
    if sep_index is None:
        return {}, text

    header: dict[str, str] = {}
    for line in lines[:sep_index]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip().lower()] = value.strip()
    return header, "".join(lines[sep_index + 1:])


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return [text] if text else []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _model_for_sse(model: str) -> str:
    if not model:
        return "replay-model"
    match = re.search(r"claude-[A-Za-z0-9._-]+", model)
    if match:
        return match.group(0)
    return model.strip("[]") or model


def _chunk_obj(stream_id: str, created: int, model: str, *, content: str | None = None, role: str | None = None, finish_reason: str | None = None) -> dict:
    delta: dict[str, str] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [{
            "delta": delta,
            "logprobs": None,
            "finish_reason": finish_reason,
            "index": 0,
        }],
        "usage": None,
    }


def convert(output_path: Path, raw_sse_path: Path | None, chunk_size: int, backup: bool) -> Path:
    header, content = _read_output(output_path)
    if raw_sse_path is None:
        raw_sse_path = output_path.with_name(output_path.name.replace("_output.txt", "_raw_sse.txt"))
    if raw_sse_path == output_path:
        raise ValueError("output path and raw_sse path are the same")

    if backup and raw_sse_path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = raw_sse_path.with_name(f"{raw_sse_path.name}.bak-{stamp}")
        shutil.copy2(raw_sse_path, backup_path)
        print(f"backup={backup_path}")

    model = _model_for_sse(header.get("model", ""))
    output_time = header.get("time") or time.strftime("%Y%m%d_%H%M%S")
    created = int(time.time())
    stream_id = f"req_replay_{uuid.uuid4().hex[:24]}"

    lines: list[str] = [
        f"Time: {output_time}",
        f"Model: {header.get('model') or model}",
        "=" * 50,
    ]
    role_chunk = _chunk_obj(stream_id, created, model, role="assistant")
    lines.append(f"data: {json.dumps(role_chunk, ensure_ascii=False)}")
    lines.append("")

    for part in _chunk_text(content, chunk_size):
        chunk = _chunk_obj(stream_id, created, model, content=part)
        lines.append(f"data: {json.dumps(chunk, ensure_ascii=False)}")
        lines.append("")

    stop_chunk = _chunk_obj(stream_id, created, model, finish_reason="stop")
    lines.append(f"data: {json.dumps(stop_chunk, ensure_ascii=False)}")
    lines.append("[FINISH_REASON: stop]")
    raw_sse_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_sse_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    parser.add_argument("raw_sse_path", nargs="?", type=Path)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    out = convert(args.output_path, args.raw_sse_path, args.chunk_size, not args.no_backup)
    print(f"wrote={out}")
    print(f"size={out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
