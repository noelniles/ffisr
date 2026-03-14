from __future__ import annotations

import random
import time

from app.integration.contracts import PerceptionAdapter, PerceptionBatch, SemanticEvent, TrackObservation


class SimulatedPerceptionAdapter(PerceptionAdapter):
    """Lightweight adapter useful for adapter-pipeline testing without hardware."""

    def __init__(self, source: str = "simulated", seed: int = 7) -> None:
        self.name = "simulated-perception"
        self._source = source
        self._rng = random.Random(seed)
        self._track_id = 100

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def poll(self) -> PerceptionBatch | None:
        now = time.monotonic()
        n_tracks = self._rng.randint(1, 3)

        tracks: list[TrackObservation] = []
        for _ in range(n_tracks):
            self._track_id += 1
            tracks.append(
                TrackObservation(
                    timestamp_s=now,
                    track_id=self._track_id,
                    cls=self._rng.choice(["person", "vehicle", "bike"]),
                    confidence=round(self._rng.uniform(0.75, 0.98), 2),
                    bbox_xywh=(
                        self._rng.randint(10, 640),
                        self._rng.randint(10, 360),
                        self._rng.randint(28, 120),
                        self._rng.randint(28, 120),
                    ),
                    velocity_mps=round(self._rng.uniform(0.0, 22.0), 2),
                    sensor_id="sim-cam-1",
                )
            )

        events: list[SemanticEvent] = []
        if self._rng.random() < 0.2:
            events.append(
                SemanticEvent(
                    timestamp_s=now,
                    event_type="anomaly",
                    detail="synthetic low-confidence blip",
                    track_id=tracks[0].track_id if tracks else None,
                    severity="warn",
                    source=self._source,
                )
            )

        return PerceptionBatch(timestamp_s=now, source=self._source, tracks=tracks, events=events)
