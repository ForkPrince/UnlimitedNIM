from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Callable, Optional

import httpx
import pytest

from app import Settings, create_app, RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def set(self, seconds: float) -> None:
        self.t = seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


async def settle(times: int = 3) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def cancel_all(*tasks) -> None:
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def sse_chunks(*items) -> bytes:
    return "".join(f"data: {json.dumps(item)}\n\n" for item in items).encode()


def mock_upstream_handler(
    status_code: int = 200,
    body: Optional[bytes] = None,
    json_body: Optional[dict] = None,
    headers: Optional[dict] = None,
    count_429: int = 0,
    calls: Optional[list] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if calls is not None:
            calls.append(json.loads(request.content.decode("utf-8", "replace")))
        if state["n"] <= count_429:
            return httpx.Response(429, json={"error": "rate limited"}, request=request)
        resp_headers = {"content-type": "text/event-stream", **(headers or {})}
        if json_body is not None:
            return httpx.Response(status_code, json=json_body, headers=resp_headers, request=request)
        stream = httpx.ByteStream(body or b"")
        return httpx.Response(status_code, headers=resp_headers, stream=stream, request=request)

    return handler


class AppFixture:
    def __init__(self, app, limiter: RateLimiter, settings: Settings):
        self.app = app
        self.limiter = limiter
        self.settings = settings
        self.transport = httpx.ASGITransport(app=app)

    def client(self, base_url: str = "http://test") -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, base_url=base_url)


@pytest.fixture
def make_app(fake_clock: FakeClock) -> Callable[..., AppFixture]:
    def _make(handler: Optional[Callable[[httpx.Request], httpx.Response]] = None, **kwargs) -> AppFixture:
        settings_kwargs = {
            "backoff_base": 0.001,
            "sweep_interval": 0.001,
            "max_rpm": kwargs.pop("max_rpm", 40),
        }
        settings_kwargs.update(kwargs)
        settings = Settings(**settings_kwargs)
        if handler is not None:
            settings.transport = httpx.MockTransport(handler)
        limiter = RateLimiter(
            max_rpm=settings.max_rpm,
            window_size=settings.window_size,
            max_queue=settings.max_queue,
            retry_ttl=settings.retry_ttl,
            clock=fake_clock,
        )
        app = create_app(settings, limiter=limiter)
        return AppFixture(app, limiter, settings)

    return _make


@pytest.fixture
def standard_body() -> dict:
    return {
        "model": "meta/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "max_tokens": 16,
    }


@pytest.fixture(autouse=True)
def _disable_warnings() -> None:
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*task.*")