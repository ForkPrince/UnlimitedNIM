<p align="center">
  <img src="assets/logo.svg" alt="UnlimitedNIM" width="600"/>
</p>

<h3 align="center">UnlimitedNIM</h3>

<p align="center">
  Make NVIDIA's 40 RPM free tier feel unlimited. Smart priority queue proxy for coding agents.
</p>

<p align="center">
  <a href="https://github.com/shivnathtathe/UnlimitedNIM/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"/></a>
  <a href="https://github.com/shivnathtathe/UnlimitedNIM"><img src="https://img.shields.io/badge/tests-47%20passing-brightgreen.svg" alt="Tests"/></a>
  <a href="https://github.com/shivnathtathe/UnlimitedNIM"><img src="https://img.shields.io/badge/NVIDIA%20NIM-40%20RPM-76b900.svg" alt="NVIDIA NIM"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-async-009688.svg" alt="FastAPI"/>
</p>

<p align="center">
  <b>Works with Cline · Roo Code · OpenCode · Any OpenAI-compatible agent</b>
</p>

## The Problem

NVIDIA's free NIM tier limits you to **40 requests per minute**. Coding agents like
**Cline, Roo Code, and OpenCode** generate bursts of traffic — an agent loop firing
tool calls, context reads, and model invocations in parallel will slam straight into
that wall. The result: a flood of `429 Rate Limit Exceeded` errors, retried work,
frustrated agents, and stalls in the middle of a task.

You don't get more throughput from NVIDIA — but you *can* stop wasting the window
you have.

## What This Is

UnlimitedNIM is a thin, streaming reverse proxy that sits between your coding agent
and NVIDIA's API. It enforces NVIDIA's 40 RPM limit *for you* — in a way that's
invisible to your agent:

- **Sliding-window rate limiter** — never fires more than 40 requests per minute,
  so NVIDIA never sees a burst and never returns 429.
- **Priority queue** — when the window is full, requests wait in line instead of
  failing:
  - **priority 1** — user messages (you typed it, you're waiting)
  - **priority 3** — agent loop tool calls (the bulk of traffic)
  - **priority 5** — debug dumps and huge stack traces (fire when there's room)
- **Full-duplex streaming** — SSE responses stream straight through, chunk by chunk.
- **Long-thinking resilience** — if the model pauses between chunks (thinking, or
  generating a massive file edit), the proxy emits SSE `: keep-alive` comments and
  keeps the stream open instead of aborting it on a read timeout.
- **Starvation protection** — a low-priority request stuck longer than the TTL is
  promoted so it eventually fires.
- **Invisible 429 recovery** — if NVIDIA still returns `429`, the proxy releases
  the slot, waits for the sliding window (or NVIDIA's `Retry-After`), and retries
  without ever showing the rate-limit error to the client.
- **Backpressure** — a bounded queue rejects with `503 + Retry-After` instead of
  leaking memory or flooding NVIDIA.

### The Screenshot-Worthy Stat

> **260+ real NVIDIA requests driven through the proxy — zero leaked 429s, zero rejections.**

Wave 1 (40 concurrent) fired instantly. Wave 2 (60 more, fired while wave 1 was still
in flight) queued and streamed in across the next window — every single request
completed, and not one rate-limit error reached a client.

## How It Works

```
Coding agent ──POST /v1/chat/completions──▶ UnlimitedNIM ──▶ NVIDIA API (40 RPM cap)
                      ▲                          │
                      └──── 200 + SSE stream ◄────┘
                        (or waits in priority queue)
```

1. A request arrives with an `X-Priority` header (1/3/5).
2. If the current 60s window has room, it fires immediately.
3. Otherwise it enters the min-heap priority queue — highest priority (lowest
   number) leaves first; same priority is FIFO.
4. As the window rolls over (or requests finish), the queue drains up to 40/min,
   never exceeding the limit.
5. Streaming responses relay byte-for-byte; client disconnects abort the upstream
   call and free the slot (the window slot is never refunded — NVIDIA already
   counted it, so we never re-fire).

## Quick Start

```bash
# 1. Set your NVIDIA key
#    (either export it or copy .env.example → .env and fill it in)
export NVIDIA_API_KEY=your-key-here

# 2. Install
pip install -r requirements.txt

# 3. Run
python main.py
# INFO: Uvicorn running on http://0.0.0.0:8000

# 4. Point your agent at http://localhost:8000/v1
```

Verify it's alive:

```bash
curl http://localhost:8000/status
```

## Configuration

All options come from environment variables or a `.env` file (auto-loaded).

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | *(required)* | Your NVIDIA API key (used upstream) |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Upstream NVIDIA base URL |
| `NVIDIA_DEFAULT_MODEL` | `meta/llama-3.3-70b-instruct` | Model used when a client sends an unknown model |
| `NVIDIA_MODELS` | *(built-in list)* | Comma-separated valid models passed through untouched |
| `REWRITE_UNKNOWN_MODELS` | `true` | `true` = rewrite unknown client models to default; `false` = pass through |
| `MAX_RPM` | `40` | Max requests per minute (matches NVIDIA free tier) |
| `WINDOW_SIZE` | `60` | Rate-limit window in seconds |
| `MAX_QUEUE` | `500` | Max queue depth before rejecting with `503 + Retry-After` |
| `RETRY_TTL` | `60` | Seconds before a queued low-priority request is promoted to priority 1 |
| `HTTP_RETRIES` | `-1` | Upstream `429` retries (`-1` = retry forever) |
| `BACKOFF_BASE` | `1.0` | Backoff base seconds; waits at least until the sliding window has room, and honours `Retry-After` |
| `CONNECT_TIMEOUT` | `30` | httpx connect timeout (s) |
| `READ_TIMEOUT` | `300` | httpx read timeout (s); also the stream idle-keep-alive interval |
| `WRITE_TIMEOUT` | `30` | httpx write timeout (s) |
| `POOL_TIMEOUT` | `30` | httpx pool timeout (s) |
| `MAX_STREAM_AGE` | `300` | Max consecutive idle seconds on a stream; also max in-flight age before the sweeper force-releases it |
| `SWEEP_INTERVAL` | `5` | Sweeper interval (s) |
| `LARGE_BODY_BYTES` | `262144` | Payload bytes above which a request defaults to priority 5 |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |

## Point Your Agent At It

### OpenCode

Add to `opencode.json` (or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "nvidia": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "UnlimitedNIM",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "anything-works"
      },
      "models": {
        "nvidia/nemotron-3.5-lightning-30b-a3b": {
          "name": "Nemotron 3.5 Lightning 30B"
        }
      }
    }
  }
}
```

### Cline

1. Settings → API Provider → **OpenAI Compatible**
2. Base URL: `http://localhost:8000/v1`
3. API Key: anything (the proxy uses its own NVIDIA key)
4. Model: `nvidia/nemotron-3.5-lightning-30b-a3b`

### Roo Code

1. Settings → API Configuration → Provider → **OpenAI Compatible**
2. Base URL: `http://localhost:8000/v1`
3. API Key: anything
4. Model: `nvidia/nemotron-3.5-lightning-30b-a3b`

## Load Testing

`load_test.py` simulates heavy agentic traffic (100 concurrent requests, mixed
priorities, two waves) and reports status counts, leaked 429s, and queue wait times:

```bash
python load_test.py                       # full run (40 + 60 waves)
python load_test.py --total 12 --wave1 5 --wave2 7   # quick smoke
```

For safe local runs without touching NVIDIA, start the included mock upstream and
point the proxy at it:

```bash
python mock_upstream.py                    # terminal 1
NVIDIA_BASE_URL=http://localhost:9001/v1 python main.py   # terminal 2
```

## Project Layout

```
app/
  config.py        # env-driven settings
  rate_limiter.py  # sliding-window limiter + priority queue + in-flight tracking
  proxy.py         # FastAPI app, streaming upstream, 429 retry, validation
main.py            # entry point
load_test.py       # agentic traffic simulator
mock_upstream.py   # fake NVIDIA upstream for local testing
tests/             # 47 tests covering all 34 test-case specs
```

## Test Coverage

The suite implements the full 34-case spec — window resets, previous-minute
in-flight tracking, priority ordering, starvation promotion, clock skew, upstream
429/500 handling, client disconnects, and the compliance layer (`/v1/models`,
`/status`).

```bash
python -m pytest -q   # 47 passed
```