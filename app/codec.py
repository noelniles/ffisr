from __future__ import annotations

import json
import math
import random
from typing import Any


def quantize_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 2)


def quantize_velocity(value: float) -> float:
    return round(max(0.0, value), 2)


def bbox_delta(current: list[int], previous: list[int] | None) -> tuple[list[int] | None, list[int]]:
    if previous is None:
        return None, current
    delta = [current[i] - previous[i] for i in range(4)]
    return delta, current


def reduced_embedding(track_id: int, timestamp: float, dims: int = 8) -> list[int]:
    seed = (track_id * 73856093) ^ int(timestamp * 10)
    rng = random.Random(seed)
    return [rng.randint(-127, 127) for _ in range(dims)]


def serialize_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def speed(vx: float, vy: float) -> float:
    return math.sqrt((vx * vx) + (vy * vy))
