from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.requests import Request

from app.engine import FeatureFirstEngine
from app.models import (
    BatchRunRequest,
    EngineStateResponse,
    KeyframeRequest,
    LinkProfileRequest,
    RFSilenceRequest,
    SummaryResponse,
    TrackingConfigRequest,
    VideoSourceRequest,
)

_batch_lock = threading.Lock()
_batch_state: dict[str, Any] = {"running": False, "result": None, "error": None}

app = FastAPI(title="Feature-First ISR Demo", version="0.1.0")
engine = FeatureFirstEngine(width=960, height=540, fps=20)
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def _startup() -> None:
    engine.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    engine.stop()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/state", response_model=EngineStateResponse)
def get_state() -> dict:
    return engine.get_state()


@app.post("/api/link_profile")
def set_link_profile(payload: LinkProfileRequest) -> JSONResponse:
    engine.set_link_profile(
        link_kbps=payload.link_kbps,
        packet_loss=payload.packet_loss,
        semantic_ratio=payload.semantic_ratio,
    )
    return JSONResponse({"ok": True})


@app.post("/api/request_keyframe")
def request_keyframe(payload: KeyframeRequest) -> JSONResponse:
    engine.request_keyframe(track_id=payload.track_id)
    return JSONResponse({"ok": True})


@app.post("/api/rf_silence")
def rf_silence(payload: RFSilenceRequest) -> JSONResponse:
    engine.trigger_rf_silence(duration_s=payload.duration_s)
    return JSONResponse({"ok": True, "duration_s": payload.duration_s})


@app.post("/api/video_source")
def set_video_source(payload: VideoSourceRequest) -> JSONResponse:
    try:
        state = engine.set_video_source(path=payload.path, loop=payload.loop)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except ValidationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, "video_source": state})


@app.post("/api/tracking_config")
def set_tracking_config(payload: TrackingConfigRequest) -> JSONResponse:
    state = engine.set_tracking_config(
        enabled=payload.enabled,
        model_path=payload.model_path,
        conf=payload.conf,
        tracker=payload.tracker,
        preprocess_mode=payload.preprocess_mode,
    )
    return JSONResponse({"ok": True, "tracking": state})


@app.get("/api/baseline_frame")
def baseline_frame() -> Response:
    return Response(content=engine.get_baseline_frame(), media_type="image/jpeg")


@app.get("/api/keyframe")
def keyframe() -> Response:
    return Response(content=engine.get_keyframe(), media_type="image/jpeg")


@app.get("/api/summary", response_model=SummaryResponse)
def summary() -> dict:
    return engine.get_summary()


@app.post("/api/batch_run")
def batch_run(payload: BatchRunRequest) -> JSONResponse:
    global _batch_state
    with _batch_lock:
        if _batch_state["running"]:
            return JSONResponse({"ok": False, "error": "Batch already running"}, status_code=409)
        _batch_state = {"running": True, "result": None, "error": None}

    def _worker() -> None:
        global _batch_state
        from app.batch import run_batch
        try:
            result = run_batch(
                dataset_dir=payload.dataset_dir,
                model_path=payload.model_path,
                tracker=payload.tracker,
                conf=payload.conf,
                link_kbps=payload.link_kbps,
                packet_loss=payload.packet_loss,
                semantic_ratio=payload.semantic_ratio,
                max_sequences=payload.max_sequences,
            )
            with _batch_lock:
                _batch_state = {"running": False, "result": result, "error": None}
        except Exception as exc:
            with _batch_lock:
                _batch_state = {"running": False, "result": None, "error": str(exc)}

    threading.Thread(target=_worker, daemon=True).start()
    return JSONResponse({"ok": True, "status": "started"})


@app.get("/api/batch_status")
def batch_status() -> JSONResponse:
    with _batch_lock:
        return JSONResponse(dict(_batch_state))
