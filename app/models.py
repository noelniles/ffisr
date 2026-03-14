from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LinkProfileRequest(BaseModel):
    link_kbps: int = Field(ge=50, le=20000)
    packet_loss: float = Field(ge=0.0, le=0.8, default=0.0)
    semantic_ratio: float = Field(
        default=0.55,
        ge=0.05,
        le=0.9,
        description="Fraction of total link allocated to semantic channel",
    )


class KeyframeRequest(BaseModel):
    track_id: int | None = None


class RFSilenceRequest(BaseModel):
    duration_s: float = Field(default=10.0, ge=1.0, le=600.0)


class VideoSourceRequest(BaseModel):
    path: str | None = None
    loop: bool = True


class TrackingConfigRequest(BaseModel):
    enabled: bool = True
    model_path: str = Field(default="yolo11n.pt", min_length=1)
    conf: float = Field(default=0.25, ge=0.05, le=0.95)
    tracker: str = Field(default="bytetrack.yaml", min_length=1)
    preprocess_mode: str = Field(default="visible")


class TrackUpdate(BaseModel):
    type: Literal["TRACK_UPDATE"] = "TRACK_UPDATE"
    track_id: int
    cls: str
    confidence: float
    velocity: float
    timestamp: float
    bbox: list[int] | None = None
    bbox_delta: list[int] | None = None
    embedding: list[int] | None = None
    gt_id: int | None = None


class EventMessage(BaseModel):
    type: Literal["EVENT"] = "EVENT"
    event_type: Literal["enter", "exit", "loiter", "anomaly"]
    track_id: int | None = None
    timestamp: float
    detail: str | None = None


class EngineStateResponse(BaseModel):
    tracks: list[dict[str, Any]]
    events: list[dict[str, Any]]
    bandwidth: dict[str, Any]
    utility: dict[str, Any]
    transport: dict[str, Any]
    rf_link: dict[str, Any]
    video_source: dict[str, Any]
    tracking: dict[str, Any]
    baseline_frame_id: int
    keyframe_frame_id: int


class SummaryResponse(BaseModel):
    summary_lines: list[str]
    bandwidth: dict[str, Any]
    utility: dict[str, Any]
    transport: dict[str, Any]


class BatchRunRequest(BaseModel):
    dataset_dir: str
    model_path: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
    conf: float = Field(default=0.25, ge=0.05, le=0.95)
    link_kbps: int = Field(default=1000, ge=50, le=20000)
    packet_loss: float = Field(default=0.02, ge=0.0, le=0.8)
    semantic_ratio: float = Field(default=0.65, ge=0.05, le=0.9)
    max_sequences: int = Field(default=5, ge=1, le=100)
