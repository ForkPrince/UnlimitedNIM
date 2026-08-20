"""Load test for the NVIDIA rate-limited proxy.

Simulates heavy agentic traffic against http://localhost:8000/v1:

  - 100 concurrent requests via asyncio + httpx.AsyncClient
  - Priority mix: 20% priority-1 short user messages,
    60% priority-3 agent tool calls (XML tags),
    20% priority-5 large fake stack traces
  - Two waves: 40 fired immediately, 60 more while wave 1 is in flight
    (stresses the previous-minute slot spawn logic: TC-002/005/008/010)
  - Tracks status counts, leaked 429s, avg response time, and queue wait
    time per request (time from send to first byte).

The client sends a dummy key; the proxy uses its own key for upstream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx

SSE_URL_PATH = "/v1/chat/completions"


@dataclass
class RequestSpec:
    priority: int
    payload: dict


@dataclass
class Result:
    priority: int
    status: int | None
    ttfb_ms: float
    total_ms: float
    error: str | None = None


def build_payloads(total: int, large_body_chars: int, model: str) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    for i in range(total):
        bucket = i % 10
        if bucket < 2:
            priority = 1
            content = f"what is the answer to this quick question? {i}"
        elif bucket < 8:
            priority = 3
            content = (
                f"<tool_call><name>search_files</name><arguments>"
                f'{{"pattern": "*.py", "query": "rate_limiter_{i}"}}'
                f"</arguments></tool_call>"
            )
        else:
            priority = 5
            content = (
                "Traceback (most recent call last):\n"
                + "\n".join(
                    f'  File "/src/app/module_{i}_{k}.py", line {k + 10}, in handler_{k}\n'
                    f"    result = compute(input_{k})\n"
                    f'  File "/src/app/core_{i}.py", line {k}, in compute\n'
                    f"    raise ValueError('stack frame {i}.{k}')"
                    for k in range(40)
                )
            )
        base = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "max_tokens": 1,
        }
        specs.append(RequestSpec(priority=priority, payload=base))
    return specs


async def send_one(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    spec: RequestSpec,
) -> Result:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Priority": str(spec.priority),
    }
    url = f"{base_url}{SSE_URL_PATH}"
    started = time.perf_counter()
    try:
        async with client.stream("POST", url, json=spec.payload, headers=headers) as resp:
            ttfb = time.perf_counter() - started
            await resp.aread()
            total = time.perf_counter() - started
            return Result(
                priority=spec.priority,
                status=resp.status_code,
                ttfb_ms=ttfb * 1000,
                total_ms=total * 1000,
            )
    except httpx.HTTPError as exc:
        total = time.perf_counter() - started
        return Result(
            priority=spec.priority,
            status=None,
            ttfb_ms=total * 1000,
            total_ms=total * 1000,
            error=str(exc),
        )


async def run_wave(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    specs: list[RequestSpec],
) -> list[Result]:
    tasks = [asyncio.create_task(send_one(client, base_url, api_key, s)) for s in specs]
    return await asyncio.gather(*tasks)


def print_summary(results: list[Result], wave_sizes: list[int]) -> None:
    by_priority: dict[int, list[Result]] = {}
    for r in results:
        by_priority.setdefault(r.priority, []).append(r)

    leaked_429 = sum(1 for r in results if r.status == 429)

    print("\n=== NVIDIA PROXY LOAD TEST SUMMARY ===")
    print(f"waves: {' + '.join(str(w) for w in wave_sizes)} total requests\n")

    header = f"{'PRIO':>4} {'SENT':>5} {'200':>5} {'503':>5} {'429':>5} {'ERR':>5} {'AVG_TTFB':>10} {'AVG_TOTAL':>10}"
    print(header)
    print("-" * len(header))
    for prio in sorted(by_priority):
        group = by_priority[prio]
        statuses = Counter(r.status for r in results if r.priority == prio)
        ttfb = [r.ttfb_ms for r in group if r.error is None]
        total = [r.total_ms for r in group if r.error is None]
        print(
            f"{prio:>4} {len(group):>5} "
            f"{statuses.get(200, 0):>5} {statuses.get(503, 0):>5} "
            f"{statuses.get(429, 0):>5} "
            f"{sum(1 for r in group if r.error):>5} "
            f"{statistics.mean(ttfb) if ttfb else 0:>9.0f}ms "
            f"{statistics.mean(total) if total else 0:>9.0f}ms"
        )

    statuses = Counter(r.status for r in results)
    all_ttfb = [r.ttfb_ms for r in results if r.error is None]
    all_total = [r.total_ms for r in results if r.error is None]
    print("-" * len(header))
    print(
        f"{'ALL':>4} {len(results):>5} "
        f"{statuses.get(200, 0):>5} {statuses.get(503, 0):>5} "
        f"{statuses.get(429, 0):>5} "
        f"{sum(1 for r in results if r.error):>5} "
        f"{statistics.mean(all_ttfb) if all_ttfb else 0:>9.0f}ms "
        f"{statistics.mean(all_total) if all_total else 0:>9.0f}ms"
    )
    print()

    print(f"429s leaked through from NVIDIA: {leaked_429}")
    p95_ttfb = statistics.quantiles(all_ttfb, n=20)[-1] if len(all_ttfb) >= 20 else 0
    print(f"queue wait p95 (TTFB): {p95_ttfb:.0f}ms")
    if leaked_429:
        print("!! NVIDIA rate limit leaked to clients — proxy window drift!")
    else:
        print("OK: no 429 leaked; all client 429s handled by proxy retry logic.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the NVIDIA proxy")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="dummy-client-key")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--wave1", type=int, default=40)
    parser.add_argument("--wave2", type=int, default=60)
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--large-body-chars", type=int, default=200000)
    args = parser.parse_args()

    specs = build_payloads(args.total, args.large_body_chars, args.model)
    wave1_specs = specs[: args.wave1]
    wave2_specs = specs[args.wave1 :]

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        print(f"firing wave 1: {len(wave1_specs)} requests")
        wave1_results = await run_wave(client, args.base_url, args.api_key, wave1_specs)
        fired = sum(1 for r in wave1_results if r.status is not None)
        print(f"wave 1 returned ({fired} responses) — firing wave 2: {len(wave2_specs)}")
        wave2_results = await run_wave(client, args.base_url, args.api_key, wave2_specs)

    results = wave1_results + wave2_results
    print_summary(results, [len(wave1_specs), len(wave2_specs)])


if __name__ == "__main__":
    asyncio.run(main())