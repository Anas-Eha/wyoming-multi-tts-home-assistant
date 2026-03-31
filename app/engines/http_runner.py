"""Generic HTTP runner for engine subprocesses."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from typing import Any

from .base import EngineError, EngineNotLoadedError, TtsEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="engine http runner")
    parser.add_argument("--module", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args(argv)


def load_engine(module_path: str, class_name: str) -> TtsEngine:
    module = import_module(module_path)
    engine_class = getattr(module, class_name)
    return engine_class()


def json_response(
    handler: BaseHTTPRequestHandler,
    payload: dict[str, Any],
    *,
    status: int = HTTPStatus.OK,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    return json.loads(handler.rfile.read(content_length).decode("utf-8"))


def engine_error_response(
    handler: BaseHTTPRequestHandler,
    err: Exception,
    *,
    status: int,
) -> None:
    json_response(
        handler,
        {"ok": False, "error": str(err), "error_type": err.__class__.__name__},
        status=status,
    )


def build_handler(engine: TtsEngine):
    class EngineHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_get(self) -> dict[str, Any] | None:
            if self.path == "/health":
                return {"ok": True, "status": engine.status().asdict()}
            if self.path == "/status":
                return {"ok": True, "status": engine.status().asdict()}
            return None

        def do_GET(self) -> None:  # noqa: N802
            payload = self._handle_get()
            if payload is not None:
                json_response(self, payload)
                return
            json_response(
                self,
                {"ok": False, "error": f"Unknown GET path '{self.path}'"},
                status=HTTPStatus.NOT_FOUND,
            )

        def _handle_post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
            if self.path == "/load":
                status = engine.load(payload.get("device_preference"))
                return {"ok": True, "status": status.asdict()}
            if self.path == "/unload":
                engine.unload()
                return {"ok": True, "status": engine.status().asdict()}
            if self.path == "/synthesize":
                result = engine.synthesize(
                    payload["text"],
                    voice=payload.get("voice"),
                    language=payload.get("language"),
                    options=payload.get("options"),
                )
                return {"ok": True, "result": result.to_transport_dict()}
            return None

        def do_POST(self) -> None:  # noqa: N802
            payload = read_json_body(self)
            try:
                response_payload = self._handle_post(payload)
                if response_payload is not None:
                    json_response(self, response_payload)
                    return
            except EngineNotLoadedError as err:
                engine_error_response(self, err, status=HTTPStatus.CONFLICT)
                return
            except EngineError as err:
                engine_error_response(self, err, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as err:  # pragma: no cover - runtime dependent
                engine_error_response(self, err, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            json_response(
                self,
                {"ok": False, "error": f"Unknown POST path '{self.path}'"},
                status=HTTPStatus.NOT_FOUND,
            )

    return EngineHandler


def main() -> None:
    args = parse_args()
    engine = load_engine(args.module, args.class_name)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(engine))
    server.serve_forever()


if __name__ == "__main__":
    main()
