"""FastAPI application for control panel and debug synthesis."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.audio.text_chunking import split_text_for_tts
from app.engines.base import EngineError, EngineNotLoadedError
from app.engines.manager import EngineManager

from .models import ActivateEngineRequest, EngineOptionsRequest, OpenAiSpeechRequest, SelectEngineRequest, SynthesizeRequest

OPENAI_SPEECH_MODEL = "wyoming-multi-tts"
_SYNTHESIS_INCLUDE_BASE64 = os.getenv("SYNTHESIS_INCLUDE_BASE64", "true").lower() in ("true", "1", "yes")


def _json_response(payload) -> JSONResponse:
    return JSONResponse(payload)


def _bad_request(err: EngineError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(err))


def _server_error(err: Exception) -> HTTPException:
    return HTTPException(status_code=500, detail=str(err))


def _synthesis_payload(result) -> dict:
    payload = result.asdict()
    if _SYNTHESIS_INCLUDE_BASE64:
        payload["wav_base64"] = base64.b64encode(result.wav_audio).decode("ascii")
    else:
        payload["wav_bytes"] = len(result.wav_audio)
    return payload


def _audio_response(result, response_format: str) -> Response:
    normalized_format = response_format.strip().lower()
    if normalized_format == "wav":
        media_type = "audio/wav"
        audio_bytes = result.wav_audio
    elif normalized_format == "pcm":
        media_type = "application/octet-stream"
        audio_bytes = result.pcm_audio
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. Supported formats: wav, pcm",
        )
    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "X-TTS-Model": OPENAI_SPEECH_MODEL,
            "X-TTS-Engine-Id": result.engine_id,
            "X-TTS-Voice": result.voice,
            "X-TTS-Language": result.language,
            "X-TTS-Backend": result.backend,
        },
    )


def create_http_app(manager: EngineManager) -> FastAPI:
    app = FastAPI(title="wyoming-multi-tts")
    root = Path(__file__).resolve().parent.parent
    static_dir = root / "static"
    template_path = root / "templates" / "index.html"

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return template_path.read_text(encoding="utf-8")

    @app.get("/health")
    async def health():
        return _json_response(manager.health_payload())

    @app.get("/api/status")
    async def api_status():
        return _json_response(manager.active_status())

    @app.get("/api/engines")
    async def api_engines():
        return _json_response({"engines": manager.list_engines()})

    @app.post("/api/engines/select")
    async def api_select_engine(request: SelectEngineRequest):
        try:
            return _json_response(await manager.select_engine(request.engine_id))
        except EngineError as err:
            raise _bad_request(err) from err

    @app.post("/api/engines/activate")
    async def api_activate_engine(request: ActivateEngineRequest):
        try:
            return _json_response(await manager.activate_engine(request.engine_id))
        except EngineError as err:
            raise _bad_request(err) from err

    @app.post("/api/engines/options")
    async def api_engine_options(request: EngineOptionsRequest):
        try:
            return _json_response(await manager.set_active_engine_options(request.options))
        except EngineError as err:
            raise _bad_request(err) from err

    @app.post("/api/engines/load")
    async def api_load_engine():
        try:
            return _json_response(await manager.load_active_engine())
        except EngineError as err:
            raise _bad_request(err) from err
        except RuntimeError as err:
            raise _server_error(err) from err

    @app.post("/api/engines/unload")
    async def api_unload_engine():
        return _json_response(await manager.unload_active_engine())

    @app.get("/api/voices")
    async def api_voices():
        return _json_response({"voices": [voice.__dict__ for voice in manager.active_voices()]})

    @app.post("/api/synthesize")
    async def api_synthesize(request: SynthesizeRequest):
        try:
            result = await manager.synthesize(
                text=request.text,
                voice=request.voice,
                language=request.language,
                options=request.options,
            )
        except EngineNotLoadedError as err:
            raise HTTPException(status_code=409, detail=str(err)) from err
        except EngineError as err:
            raise _bad_request(err) from err
        except RuntimeError as err:
            raise _server_error(err) from err

        return _json_response(_synthesis_payload(result))

    @app.post("/v1/audio/speech")
    async def openai_audio_speech(request: OpenAiSpeechRequest):
        if request.stream:
            return await _openai_audio_speech_stream(request, manager)
        try:
            result = await manager.synthesize(
                text=request.input,
                voice=request.voice,
                language=None,
                options=None,
            )
        except EngineNotLoadedError as err:
            raise HTTPException(status_code=409, detail=str(err)) from err
        except EngineError as err:
            raise _bad_request(err) from err
        except HTTPException:
            raise
        except RuntimeError as err:
            raise _server_error(err) from err
        return _audio_response(result, request.response_format)

    return app


async def _openai_audio_speech_stream(
    request: OpenAiSpeechRequest,
    manager: EngineManager,
) -> StreamingResponse:
    """Stream audio chunks as they are synthesized from text fragments."""
    normalized_format = request.response_format.strip().lower()
    if normalized_format not in ("wav", "pcm"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{request.response_format}'. Supported formats: wav, pcm",
        )

    media_type = "audio/wav" if normalized_format == "wav" else "application/octet-stream"
    chunks = split_text_for_tts(request.input)
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty input text")

    async def audio_generator() -> AsyncIterator[bytes]:
        for i, chunk_text in enumerate(chunks):
            try:
                result = await manager.synthesize(
                    text=chunk_text,
                    voice=request.voice,
                    language=None,
                    options=None,
                )
            except EngineNotLoadedError as err:
                raise HTTPException(status_code=409, detail=str(err)) from err
            except EngineError as err:
                raise _bad_request(err) from err

            if normalized_format == "wav":
                # For streaming, each chunk needs its own WAV header
                # so the client can play it incrementally
                yield result.wav_audio
            else:
                yield result.pcm_audio

    return StreamingResponse(
        audio_generator(),
        media_type=media_type,
        headers={
            "X-TTS-Model": OPENAI_SPEECH_MODEL,
            "X-TTS-Streaming": "true",
        },
    )
