from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass(slots=True)
class AdapterSpec:
    kind: str
    driver: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntegrationConfig:
    """Top-level integration configuration for hardware-agnostic adapter wiring."""

    vehicle: AdapterSpec
    perception: AdapterSpec
    transport: AdapterSpec | None = None
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntegrationConfig":
        if "vehicle" not in payload or "perception" not in payload:
            raise ValueError("integration config requires both 'vehicle' and 'perception'")

        vehicle = _parse_adapter_spec(payload["vehicle"], default_kind="vehicle")
        perception = _parse_adapter_spec(payload["perception"], default_kind="perception")

        transport_payload = payload.get("transport")
        transport = None
        if transport_payload is not None:
            transport = _parse_adapter_spec(transport_payload, default_kind="transport")

        schema_version = str(payload.get("schema_version", "1.0"))
        return cls(vehicle=vehicle, perception=perception, transport=transport, schema_version=schema_version)

    @classmethod
    def from_toml_file(cls, path: str | Path) -> "IntegrationConfig":
        file_path = Path(path)
        raw = file_path.read_bytes()
        parsed = tomllib.loads(raw.decode("utf-8"))
        data = parsed.get("integration")
        if not isinstance(data, dict):
            raise ValueError("expected [integration] table in TOML config")
        return cls.from_dict(data)


def _parse_adapter_spec(payload: dict[str, Any], default_kind: str) -> AdapterSpec:
    if not isinstance(payload, dict):
        raise ValueError("adapter spec must be a dictionary")

    driver = payload.get("driver")
    if not driver:
        raise ValueError("adapter spec requires a 'driver' entry")

    kind = str(payload.get("kind", default_kind))
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("adapter 'params' must be a dictionary")

    return AdapterSpec(kind=kind, driver=str(driver), params=params)
