from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class PlatformState:
    """Vendor-neutral vehicle/navigation state."""

    timestamp_s: float
    source: str
    frame_id: str | None = None
    lat_deg: float | None = None
    lon_deg: float | None = None
    alt_m: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    groundspeed_mps: float | None = None
    mode: str | None = None
    armed: bool | None = None
    health: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackObservation:
    """Canonical track produced by any perception stack."""

    timestamp_s: float
    track_id: int
    cls: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    velocity_mps: float | None = None
    gt_id: int | None = None
    sensor_id: str | None = None
    embedding: list[int] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticEvent:
    """Canonical event message produced by perception/system logic."""

    timestamp_s: float
    event_type: str
    detail: str | None = None
    track_id: int | None = None
    severity: str = "info"
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KeyframePacket:
    """Reference image for the receiver, optionally attached to a track."""

    timestamp_s: float
    jpeg_bytes: bytes
    width: int
    height: int
    quality: int
    reason: str
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PerceptionBatch:
    """One perception output slice from an adapter poll."""

    timestamp_s: float
    source: str
    tracks: list[TrackObservation] = field(default_factory=list)
    events: list[SemanticEvent] = field(default_factory=list)
    keyframe: KeyframePacket | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticSnapshot:
    """Fused vehicle + perception state emitted by the integration pipeline."""

    timestamp_s: float
    perception: PerceptionBatch
    platform: PlatformState | None = None
    schema_version: str = "1.0"
    provenance: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class VehicleAdapter(Protocol):
    """Adapter contract for autopilot/vehicle state sources (PX4, ArduPilot, etc.)."""

    name: str

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def latest_state(self) -> PlatformState | None:
        ...


@runtime_checkable
class PerceptionAdapter(Protocol):
    """Adapter contract for perception systems (SightLine, custom model server, etc.)."""

    name: str

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def poll(self) -> PerceptionBatch | None:
        ...


@runtime_checkable
class TransportAdapter(Protocol):
    """Optional contract for non-emulator transport implementations."""

    name: str

    def send(self, channel: str, payload_bytes: int) -> bool:
        ...

    def stats(self) -> dict[str, dict[str, int]]:
        ...
