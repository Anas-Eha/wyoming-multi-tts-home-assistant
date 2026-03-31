"""Application entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

import uvicorn

from app.engines.manager import EngineManager
from app.engines.registry import build_registry
from app.http.server import create_http_app
from app.state.session_state import SessionStateStore
from app.wyoming.server import serve_wyoming


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wyoming-multi-tts")
    parser.add_argument("--uri", default=os.getenv("WYOMING_URI", "tcp://0.0.0.0:10210"))
    parser.add_argument("--http-host", default=os.getenv("HTTP_HOST", "0.0.0.0"))
    parser.add_argument("--http-port", type=int, default=int(os.getenv("HTTP_PORT", "8280")))
    parser.add_argument("--state-path", default=os.getenv("STATE_PATH", "/data/state/session.json"))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> None:
    manager = EngineManager(build_registry(), SessionStateStore(args.state_path))
    app = create_http_app(manager)
    autoload_task = None
    if manager.should_autoload():
        async def startup_autoload() -> None:
            try:
                await manager.autoload_active_engine()
            except Exception:
                logging.exception("Startup autoload failed")

        autoload_task = asyncio.create_task(startup_autoload())
    http_server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.http_host,
            port=args.http_port,
            log_level="debug" if args.debug else "info",
        )
    )
    tasks = [
        serve_wyoming(manager, args.uri),
        http_server.serve(),
    ]
    if autoload_task is not None:
        tasks.append(autoload_task)
    await asyncio.gather(*tasks)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
