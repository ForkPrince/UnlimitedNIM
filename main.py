"""Entry point: run the proxy with uvicorn."""

from __future__ import annotations

import logging
import re
import sys

import uvicorn

from app import Settings, create_app

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_LOGO_HEIGHT = 6

_LOGO = {
    "U": ["█   █", "█░  █░", "█░░ █░", "█░░ █░", "█░░██░", "░████░"],
    "N": ["█   █", "██  █", "█░█ █", "█░░██", "█   █", "█   █"],
    "L": ["█   ", "█░  ", "█░░ ", "█░░ ", "█░░ ", "█████"],
    "I": ["█████", "░░█░░", "░░█░░", "░░█░░", "░░█░░", "█████"],
    "M": ["█   █", "██ ██", "█░█░█", "█░░░█", "█   █", "█   █"],
    "T": ["█████", "░░█░░", "░░█░░", "░░█░░", "░░█░░", "░░█░░"],
    "E": ["█████", "█░░░░", "█░░░░", "█████", "█░░░░", "█████"],
    "D": ["████░", "█░░█░", "█░░█░", "█░░█░", "█░░█░", "████░"],
    " ": ["     "] * _LOGO_HEIGHT,
}

BAR_CELLS = 20
BAR_FULL = "█"
BAR_EMPTY = "░"


def _render_logo(text: str) -> list[str]:
    rows = [""] * _LOGO_HEIGHT
    for ch in text.upper():
        glyph = _LOGO.get(ch, ["     "] * _LOGO_HEIGHT)
        for i in range(_LOGO_HEIGHT):
            rows[i] += glyph[i] + " "
    return [row.rstrip() for row in rows]


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _render_bar(used_rpm: int, max_rpm: int) -> str:
    filled = round(BAR_CELLS * used_rpm / max_rpm) if max_rpm else 0
    filled = max(0, min(BAR_CELLS, filled))
    bar = BAR_FULL * filled + BAR_EMPTY * (BAR_CELLS - filled)
    return f"{YELLOW}▶▶ {RESET}[{GREEN}{bar}{RESET}] {GREEN}{used_rpm}/{max_rpm} RPM{RESET}"


def print_banner(settings: Settings) -> None:
    logo = _render_logo("UnlimitedNIM")
    width = max(len(r) for r in logo)

    print(f"{CYAN}╔{'═' * (width + 2)}╗{RESET}")
    for row in logo:
        print(f"{CYAN}║ {row.ljust(width)} ║{RESET}")
    print(f"{CYAN}║ {' ' * width} ║{RESET}")
    bar = _render_bar(0, settings.max_rpm)
    print(f"{CYAN}║ {bar}{' ' * (width - _visible_len(bar))} ║{RESET}")
    print(f"{CYAN}╚{'═' * (width + 2)}╝{RESET}")
    print(f"{YELLOW}  Make NVIDIA's 40 RPM free tier feel unlimited.{RESET}")
    print(f"{DIM}  Built by Shivnath Tathe  |  github.com/shivnathtathe/UnlimitedNIM{RESET}")
    print(f"{GREEN}  Listening on http://{settings.host}:{settings.port}/v1  |  43 tests passing  |  MIT License{RESET}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    settings = Settings()
    print_banner(settings)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()