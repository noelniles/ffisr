# Hardware Integration Layer

This package defines a hardware-agnostic adapter layer so the Feature-First ISR stack
can integrate with PX4 + SightLine now and stay portable to other systems later.

## File layout

- `contracts.py`
  - Canonical data contracts and adapter protocols.
- `config.py`
  - Typed integration config loader (`from_dict`, `from_toml_file`).
- `factory.py`
  - Registry/factory for adapter drivers.
- `pipeline.py`
  - Minimal fusion pipeline producing `SemanticSnapshot` objects.
- `adapters/px4.py`
  - PX4 vehicle adapter contract (MAVLink-normalized input).
- `adapters/sightline.py`
  - SightLine perception adapter contract (metadata packet input).
- `adapters/simulated.py`
  - Portable test adapter without hardware.
- `adapters/base.py`
  - Null adapters used as safe defaults.

## Integration approach

1. Ingest PX4/MAVLink packets externally and call `PX4VehicleAdapter.ingest_message(...)`.
2. Ingest SightLine packets externally and call `SightLinePerceptionAdapter.ingest_packet(...)`.
3. Use `IntegrationPipeline.poll_snapshot()` to get canonical snapshots.
4. Feed snapshots into your sender/transport path.

## Why this stays portable

- Core pipeline only depends on canonical contracts.
- Vendor-specific parsing is isolated inside adapter classes.
- Adapter selection is config-driven (`integration.example.toml`).
