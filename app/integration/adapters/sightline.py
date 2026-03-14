from __future__ import annotations

import time
from collections import deque
from typing import Any

from app.integration.contracts import PerceptionAdapter, PerceptionBatch, SemanticEvent, TrackObservation


class SightLinePerceptionAdapter(PerceptionAdapter):
    """SightLine OEM metadata adapter.

    Feed it normalized packets from your SightLine ingest process.
    Each packet can include tracks/events and is converted to canonical contracts.
    """

    def __init__(self, source: str = "sightline", max_queue: int = 16) -> None:
        self.name = "sightline-perception"
        self._source = source
        self._queue: deque[PerceptionBatch] = deque(maxlen=max_queue)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self._queue.clear()

    def poll(self) -> PerceptionBatch | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def ingest_packet(self, payload: dict[str, Any]) -> None:
        timestamp_s = float(payload.get("timestamp_s", time.monotonic()))

        tracks: list[TrackObservation] = []
        for raw in payload.get("tracks", []):
            try:
                track = TrackObservation(
                    timestamp_s=float(raw.get("timestamp_s", timestamp_s)),
                    track_id=int(raw["track_id"]),
                    cls=str(raw.get("class", "unknown")),
                    confidence=float(raw.get("confidence", 0.0)),
                    bbox_xywh=_bbox_tuple(raw.get("bbox_xywh") or raw.get("bbox")),
                    velocity_mps=_as_optional_float(raw.get("velocity_mps") or raw.get("velocity")),
                    sensor_id=_as_optional_str(raw.get("sensor_id")),
                    embedding=_embedding_or_none(raw.get("embedding")),
                    attributes=_dict_or_empty(raw.get("attributes")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            tracks.append(track)

        events: list[SemanticEvent] = []
        for raw in payload.get("events", []):
            try:
                event = SemanticEvent(
                    timestamp_s=float(raw.get("timestamp_s", timestamp_s)),
                    event_type=str(raw.get("event_type", "unknown")),
                    detail=_as_optional_str(raw.get("detail")),
                    track_id=_as_optional_int(raw.get("track_id")),
                    severity=str(raw.get("severity", "info")),
                    source=self._source,
                    metadata=_dict_or_empty(raw.get("metadata")),
                )
            except (TypeError, ValueError):
                continue
            events.append(event)

        self._queue.append(
            PerceptionBatch(
                timestamp_s=timestamp_s,
                source=self._source,
                tracks=tracks,
                events=events,
                metadata=_dict_or_empty(payload.get("metadata")),
            )
        )


def _bbox_tuple(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be a 4-item list/tuple")
    return int(value[0]), int(value[1]), int(value[2]), int(value[3])


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _embedding_or_none(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    return [int(v) for v in value]
