"""Simulated NVIDIA upstream for safe local load testing.

Run the proxy with NVIDIA_BASE_URL pointing at this mock:

    NVIDIA_BASE_URL=http://localhost:9001/v1 python main.py

This implements the same /v1/chat/completions surface (JSON + SSE) the real
NVIDIA API exposes, with a small artificial latency so requests stay in flight
long enough to exercise the rate limiter's queue and window logic.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Mock NVIDIA Upstream")

LATENCY_MS = float(100)


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": "meta/llama-3.3-70b-instruct", "object": "model", "owned_by": "nvidia"},
            {"id": "deepseek-ai/deepseek-r1", "object": "model", "owned_by": "nvidia"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    stream = body.get("stream", False)
    model = body.get("model", "mock-model")
    req_id = f"mock-{uuid.uuid4().hex[:12]}"
    prompt_tokens = sum(len(m.get("content", "").split()) for m in body.get("messages", []))
    max_tokens = int(body.get("max_tokens", 16) or 16)

    await asyncio.sleep(LATENCY_MS / 1000.0)

    if stream:
        async def gen():
            yield f'data: {_chunk(req_id, model, {"role": "assistant", "content": "ok"}, None)}\n\n'.encode()
            yield f'data: {_chunk(req_id, model, {}, "stop")}\n\n'.encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    content = {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": min(max_tokens, 1),
            "total_tokens": prompt_tokens + min(max_tokens, 1),
        },
    }
    return StreamingResponse(
        iter([__import__("json").dumps(content).encode()]),
        media_type="application/json",
    )


def _chunk(req_id: str, model: str, delta: dict, finish: str | None) -> str:
    import json

    return json.dumps(
        {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9001, log_level="warning")