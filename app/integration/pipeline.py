from __future__ import annotations

import time
from dataclasses import dataclass

from app.integration.contracts import PerceptionAdapter, SemanticSnapshot, TransportAdapter, VehicleAdapter


@dataclass(slots=True)
class PipelineStats:
    frames_polled: int = 0
    snapshots_emitted: int = 0
    last_emit_s: float = 0.0


class IntegrationPipeline:
    """Minimal fusion pipeline for vehicle + perception adapters.

    The pipeline emits canonical SemanticSnapshot objects that can be consumed
    by the existing engine/transport path or a future production sender.
    """

    def __init__(
        self,
        vehicle: VehicleAdapter,
        perception: PerceptionAdapter,
        transport: TransportAdapter | None = None,
        schema_version: str = "1.0",
    ) -> None:
        self.vehicle = vehicle
        self.perception = perception
        self.transport = transport
        self.schema_version = schema_version
        self.stats = PipelineStats()

    def start(self) -> None:
        self.vehicle.start()
        self.perception.start()

    def stop(self) -> None:
        self.perception.stop()
        self.vehicle.stop()

    def poll_snapshot(self) -> SemanticSnapshot | None:
        self.stats.frames_polled += 1
        batch = self.perception.poll()
        if batch is None:
            return None

        snapshot = SemanticSnapshot(
            timestamp_s=batch.timestamp_s,
            perception=batch,
            platform=self.vehicle.latest_state(),
            schema_version=self.schema_version,
            provenance={
                "vehicle": self.vehicle.name,
                "perception": self.perception.name,
            },
        )

        self.stats.snapshots_emitted += 1
        self.stats.last_emit_s = time.monotonic()
        return snapshot

    def transport_stats(self) -> dict[str, dict[str, int]]:
        if self.transport is None:
            return {}
        return self.transport.stats()
