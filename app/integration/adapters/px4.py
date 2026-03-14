from __future__ import annotations

from typing import Any

from app.integration.contracts import PlatformState, VehicleAdapter


class PX4VehicleAdapter(VehicleAdapter):
    """PX4-facing adapter contract.

    This class intentionally avoids a hard dependency on pymavlink/mavsdk.
    Integrators can feed parsed PX4/MAVLink dictionaries via ingest_message().
    """

    def __init__(self, source: str = "px4", frame_id: str = "NED") -> None:
        self.name = "px4-vehicle"
        self._source = source
        self._frame_id = frame_id
        self._latest: PlatformState | None = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def latest_state(self) -> PlatformState | None:
        return self._latest

    def ingest_message(self, payload: dict[str, Any]) -> None:
        """Update adapter state from a normalized MAVLink dictionary."""

        self._latest = PlatformState(
            timestamp_s=float(payload.get("timestamp_s", 0.0)),
            source=self._source,
            frame_id=str(payload.get("frame_id", self._frame_id)),
            lat_deg=_as_optional_float(payload.get("lat_deg")),
            lon_deg=_as_optional_float(payload.get("lon_deg")),
            alt_m=_as_optional_float(payload.get("alt_m")),
            roll_deg=_as_optional_float(payload.get("roll_deg")),
            pitch_deg=_as_optional_float(payload.get("pitch_deg")),
            yaw_deg=_as_optional_float(payload.get("yaw_deg")),
            groundspeed_mps=_as_optional_float(payload.get("groundspeed_mps")),
            mode=_as_optional_str(payload.get("mode")),
            armed=_as_optional_bool(payload.get("armed")),
            health=_as_dict(payload.get("health")),
        )


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}
