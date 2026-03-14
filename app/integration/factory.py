from __future__ import annotations

from typing import Any, Callable

from app.integration.adapters import (
    NullPerceptionAdapter,
    NullVehicleAdapter,
    PX4VehicleAdapter,
    SightLinePerceptionAdapter,
    SimulatedPerceptionAdapter,
)
from app.integration.config import AdapterSpec, IntegrationConfig
from app.integration.contracts import PerceptionAdapter, TransportAdapter, VehicleAdapter
from app.transport import LinkEmulator

VehicleBuilder = Callable[[dict[str, Any]], VehicleAdapter]
PerceptionBuilder = Callable[[dict[str, Any]], PerceptionAdapter]
TransportBuilder = Callable[[dict[str, Any]], TransportAdapter]


class IntegrationFactory:
    """Registry-based adapter factory for hardware-agnostic wiring."""

    def __init__(self) -> None:
        self._vehicle_builders: dict[str, VehicleBuilder] = {
            "null": lambda params: NullVehicleAdapter(),
            "px4": lambda params: PX4VehicleAdapter(
                source=str(params.get("source", "px4")),
                frame_id=str(params.get("frame_id", "NED")),
            ),
        }
        self._perception_builders: dict[str, PerceptionBuilder] = {
            "null": lambda params: NullPerceptionAdapter(),
            "sightline": lambda params: SightLinePerceptionAdapter(
                source=str(params.get("source", "sightline")),
                max_queue=int(params.get("max_queue", 16)),
            ),
            "simulated": lambda params: SimulatedPerceptionAdapter(
                source=str(params.get("source", "simulated")),
                seed=int(params.get("seed", 7)),
            ),
        }
        self._transport_builders: dict[str, TransportBuilder] = {
            "link_emulator": lambda params: LinkEmulator(
                link_kbps=int(params.get("link_kbps", 5000)),
                packet_loss=float(params.get("packet_loss", 0.02)),
                semantic_ratio=float(params.get("semantic_ratio", 0.55)),
            )
        }

    def register_vehicle(self, name: str, builder: VehicleBuilder) -> None:
        self._vehicle_builders[name] = builder

    def register_perception(self, name: str, builder: PerceptionBuilder) -> None:
        self._perception_builders[name] = builder

    def register_transport(self, name: str, builder: TransportBuilder) -> None:
        self._transport_builders[name] = builder

    def build_vehicle(self, spec: AdapterSpec) -> VehicleAdapter:
        builder = self._vehicle_builders.get(spec.driver)
        if builder is None:
            raise ValueError(f"Unknown vehicle adapter driver: {spec.driver}")
        return builder(spec.params)

    def build_perception(self, spec: AdapterSpec) -> PerceptionAdapter:
        builder = self._perception_builders.get(spec.driver)
        if builder is None:
            raise ValueError(f"Unknown perception adapter driver: {spec.driver}")
        return builder(spec.params)

    def build_transport(self, spec: AdapterSpec | None) -> TransportAdapter | None:
        if spec is None:
            return None
        builder = self._transport_builders.get(spec.driver)
        if builder is None:
            raise ValueError(f"Unknown transport adapter driver: {spec.driver}")
        return builder(spec.params)

    def build_from_config(
        self,
        config: IntegrationConfig,
    ) -> tuple[VehicleAdapter, PerceptionAdapter, TransportAdapter | None]:
        vehicle = self.build_vehicle(config.vehicle)
        perception = self.build_perception(config.perception)
        transport = self.build_transport(config.transport)
        return vehicle, perception, transport
