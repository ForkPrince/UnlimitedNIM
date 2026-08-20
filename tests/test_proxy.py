from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.proxy import _parse_priority, _stream_events
from tests.conftest import cancel_all, mock_upstream_handler, settle, sse_chunks


@pytest.mark.asyncio
async def test_tc018_upstream_429_retries_with_backoff(make_app, standard_body):
    calls: list = []
    handler = mock_upstream_handler(
        json_body={"ok": True}, count_429=3, calls=calls
    )
    fx = make_app(handler=handler)
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=standard_body)
    assert resp.status_code == 200
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_tc018_upstream_429_exhausted_returns_429(make_app, standard_body):
    calls: list = []
    handler = mock_upstream_handler(json_body={"err": 1}, count_429=99, calls=calls)
    fx = make_app(handler=handler)
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=standard_body)
    assert resp.status_code == 429
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_tc019_upstream_500_no_retry(make_app, standard_body):
    calls: list = []
    handler = mock_upstream_handler(json_body={"err": "boom"}, status_code=500, calls=calls)
    fx = make_app(handler=handler)
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=standard_body)
    assert resp.status_code == 500
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_tc020_stream_abort_releases_resources(make_app, standard_body):
    fx = make_app(handler=mock_upstream_handler(body=sse_chunks({"a": 1}, {"a": 2})))
    body = {**standard_body, "stream": True}
    slot = await fx.limiter.acquire(3)
    assert slot.request_id in fx.limiter.in_flight

    class FakeRequest:
        async def is_disconnected(self):
            return False

    gen = _stream_events(fx.limiter, fx.settings, slot, body, FakeRequest())
    first = await gen.asend(None)
    assert first
    await gen.aclose()
    assert slot.request_id not in fx.limiter.in_flight
    assert fx.limiter.window_count == 1


@pytest.mark.asyncio
async def test_streaming_success_releases_slot(make_app, standard_body):
    fx = make_app(handler=mock_upstream_handler(body=sse_chunks({"a": 1}, {"a": 2})))
    body = {**standard_body, "stream": True}
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert b"data:" in resp.content
    assert fx.limiter.in_flight == {}
    assert fx.limiter.window_count == 1


@pytest.mark.asyncio
async def test_tc031_nonstream_completion_keeps_window_and_drains(make_app, standard_body):
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}))
    slots = await asyncio.gather(*[fx.limiter.acquire(3) for _ in range(40)])
    queued = [asyncio.create_task(fx.limiter.acquire(3)) for _ in range(3)]
    await settle()
    await fx.limiter.release(slots[0])
    await settle()
    assert fx.limiter.window_count == 40
    assert fx.limiter.queue_depth == 3
    assert all(not t.done() for t in queued)
    await cancel_all(*queued)


@pytest.mark.asyncio
async def test_tc025_large_body_priority_and_single_slot(make_app, standard_body):
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}))
    big = {**standard_body, "messages": [{"role": "user", "content": "x" * 300000}]}
    async with fx.client() as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=big,
            headers={"Content-Length": str(len(json.dumps(big).encode()))},
        )
    assert resp.status_code == 200
    assert fx.limiter.window_count == 1
    assert fx.limiter.queue_depth == 0


@pytest.mark.asyncio
async def test_tc026_empty_messages_rejected(make_app, standard_body):
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}))
    bad = {**standard_body, "messages": []}
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=bad)
    assert resp.status_code == 400
    assert fx.limiter.window_count == 0
    assert fx.limiter.total_fired == 0


@pytest.mark.asyncio
async def test_tc026_missing_messages_rejected(make_app, standard_body):
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}))
    bad = {k: v for k, v in standard_body.items() if k != "messages"}
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=bad)
    assert resp.status_code == 400
    assert fx.limiter.window_count == 0


@pytest.mark.asyncio
async def test_tc026_invalid_json_rejected(make_app):
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}))
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", content=b"{not json")
    assert resp.status_code == 400
    assert fx.limiter.window_count == 0


@pytest.mark.asyncio
async def test_tc027_unknown_model_mapped_to_default(make_app, standard_body):
    calls: list = []
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}, calls=calls))
    body = {**standard_body, "model": "gpt-banana-turbo"}
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert calls[0]["model"] == fx.settings.default_model


@pytest.mark.asyncio
async def test_tc027_known_model_passthrough(make_app, standard_body):
    calls: list = []
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}, calls=calls))
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=standard_body)
    assert resp.status_code == 200
    assert calls[0]["model"] == "meta/llama-3.3-70b-instruct"


@pytest.mark.asyncio
async def test_tc029_queue_full_returns_503_retry_after(make_app, standard_body):
    fx = make_app(max_rpm=1, max_queue=1)
    await fx.limiter.acquire(3)
    queued = asyncio.create_task(fx.limiter.acquire(3))
    await settle()
    async with fx.client() as client:
        resp = await client.post("/v1/chat/completions", json=standard_body)
    assert resp.status_code == 503
    assert "retry-after" in resp.headers
    await cancel_all(queued)


@pytest.mark.asyncio
async def test_tc032_models_static_not_counted(make_app):
    fx = make_app()
    async with fx.client() as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "meta/llama-3.3-70b-instruct" in ids
    assert fx.limiter.total_fired == 0
    assert fx.limiter.window_count == 0


@pytest.mark.asyncio
async def test_tc033_models_100_rapid_calls(make_app):
    fx = make_app()
    async with fx.client() as client:
        for _ in range(100):
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
    assert fx.limiter.window_count == 0
    assert fx.limiter.total_fired == 0


@pytest.mark.asyncio
async def test_tc034_status_during_load(make_app):
    fx = make_app(handler=mock_upstream_handler(json_body={"ok": True}))
    slots = await asyncio.gather(*[fx.limiter.acquire(3) for _ in range(40)])
    queued = [asyncio.create_task(fx.limiter.acquire(3)) for _ in range(200)]
    await settle()
    async with fx.client() as client:
        resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["window_count"] == 40
    assert data["queue_depth"] == 200
    assert data["window_resets_in"] >= 0
    assert data["max_rpm"] == 40
    await cancel_all(*queued)


def test_parse_priority_mapping():
    class Req:
        def __init__(self, header=None):
            self.header = header

        @property
        def headers(self):
            return {"X-Priority": self.header} if self.header else {}

    class S:
        large_body_bytes = 262144

    assert _parse_priority(Req(), b"x", 10, S()) == 3
    assert _parse_priority(Req(), b"x" * 300000, 300000, S()) == 5
    assert _parse_priority(Req("2"), b"x", 10, S()) == 2
    assert _parse_priority(Req("banana"), b"x", 10, S()) == 3
    assert _parse_priority(Req("0"), b"x", 10, S()) == 1
    assert _parse_priority(Req("9"), b"x", 10, S()) == 5