"""Env-driven settings for the NVIDIA rate-limiting proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass
class Settings:
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    nvidia_base_url: str = field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("NVIDIA_DEFAULT_MODEL", "meta/llama-3.3-70b-instruct")
    )
    max_rpm: int = field(default_factory=lambda: _env_int("MAX_RPM", 40))
    window_size: float = field(default_factory=lambda: _env_float("WINDOW_SIZE", 60.0))
    max_queue: int = field(default_factory=lambda: _env_int("MAX_QUEUE", 500))
    retry_ttl: float = field(default_factory=lambda: _env_float("RETRY_TTL", 300.0))
    # -1 = retry upstream 429s forever (the proxy stays transparent to clients)
    http_retries: int = field(default_factory=lambda: _env_int("HTTP_RETRIES", -1))
    backoff_base: float = field(default_factory=lambda: _env_float("BACKOFF_BASE", 1.0))
    connect_timeout: float = field(default_factory=lambda: _env_float("CONNECT_TIMEOUT", 30.0))
    # Long read timeout for LLM streaming (models can pause >60s between tokens)
    read_timeout: float = field(default_factory=lambda: _env_float("READ_TIMEOUT", 300.0))
    write_timeout: float = field(default_factory=lambda: _env_float("WRITE_TIMEOUT", 30.0))
    pool_timeout: float = field(default_factory=lambda: _env_float("POOL_TIMEOUT", 30.0))
    max_stream_age: float = field(default_factory=lambda: _env_float("MAX_STREAM_AGE", 300.0))
    sweep_interval: float = field(default_factory=lambda: _env_float("SWEEP_INTERVAL", 5.0))
    large_body_bytes: int = field(default_factory=lambda: _env_int("LARGE_BODY_BYTES", 262144))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    transport: Optional[httpx.AsyncBaseTransport] = None

    known_models: tuple = field(
        default_factory=lambda: tuple(
            m.strip()
            for m in os.getenv(
                "NVIDIA_MODELS",
                "nvidia/nemotron-3.5-lightning-30b-a3b,"
                "meta/llama-3.3-70b-instruct,"
                "meta/llama-3.1-8b-instruct,"
                "meta/llama-3.1-405b-instruct,"
                "mistralai/mistral-nemo-12b-instruct,"
                "nvidia/nemotron-4-340b-instruct,"
                "deepseek-ai/deepseek-r1,"
                "z-ai/glm-5.2",
            ).split(",")
            if m.strip()
        )
    )
    rewrite_unknown_models: bool = field(
        default_factory=lambda: os.getenv("REWRITE_UNKNOWN_MODELS", "true").lower() == "true"
    )

    def upstream_url(self) -> str:
        return f"{self.nvidia_base_url}/chat/completions"

    def httpx_timeout(self):
        import httpx

        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.write_timeout,
            pool=self.pool_timeout,
        )