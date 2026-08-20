"""Entry point: run the proxy with uvicorn."""

from __future__ import annotations

import logging

import uvicorn

from app import Settings, create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()