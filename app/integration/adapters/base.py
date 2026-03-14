from __future__ import annotations

import time

from app.integration.contracts import PerceptionBatch, PerceptionAdapter, PlatformState, VehicleAdapter


class NullVehicleAdapter(VehicleAdapter):
    """Safe default vehicle adapter used for local testing and CI."""

    def __init__(self) -> None:
        self.name = "null-vehicle"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def latest_state(self) -> PlatformState | None:
        return None


class NullPerceptionAdapter(PerceptionAdapter):
    """Safe default perception adapter used for local testing and CI."""

    def __init__(self) -> None:
        self.name = "null-perception"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def poll(self) -> PerceptionBatch | None:
        return PerceptionBatch(timestamp_s=time.monotonic(), source=self.name)
