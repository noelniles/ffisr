from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ChannelStats:
    attempted_packets: int = 0
    delivered_packets: int = 0
    dropped_packets: int = 0
    attempted_bytes: int = 0
    delivered_bytes: int = 0
    dropped_bytes: int = 0


class LinkEmulator:
    """Simple dual-channel token-bucket link with packet loss."""

    def __init__(
        self,
        link_kbps: int = 5000,
        packet_loss: float = 0.0,
        semantic_ratio: float = 0.25,
    ) -> None:
        self.link_kbps = link_kbps
        self.packet_loss = packet_loss
        self.semantic_ratio = semantic_ratio
        self._tokens = {"semantic": 0.0, "baseline": 0.0}
        self._last_refill = time.monotonic()
        self._stats: dict[str, ChannelStats] = defaultdict(ChannelStats)
        self._rf_silence_until = 0.0

    def set_profile(self, link_kbps: int, packet_loss: float, semantic_ratio: float) -> None:
        self.link_kbps = link_kbps
        self.packet_loss = packet_loss
        self.semantic_ratio = semantic_ratio

    def start_rf_silence(self, duration_s: float) -> None:
        now = time.monotonic()
        window_end = now + max(0.0, duration_s)
        self._rf_silence_until = max(self._rf_silence_until, window_end)

    def in_rf_silence(self) -> bool:
        return time.monotonic() < self._rf_silence_until

    def rf_silence_remaining(self) -> float:
        return max(0.0, self._rf_silence_until - time.monotonic())

    def _rates(self) -> dict[str, float]:
        total_bytes_per_sec = (self.link_kbps * 1000.0) / 8.0
        semantic = total_bytes_per_sec * self.semantic_ratio
        baseline = max(0.0, total_bytes_per_sec - semantic)
        return {"semantic": semantic, "baseline": baseline}

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now

        rates = self._rates()
        for channel in ("semantic", "baseline"):
            self._tokens[channel] += rates[channel] * elapsed
            max_bucket = rates[channel] * 2.0
            self._tokens[channel] = min(self._tokens[channel], max_bucket)

    def send(self, channel: str, payload_bytes: int) -> bool:
        if channel not in ("semantic", "baseline"):
            raise ValueError(f"Unknown channel: {channel}")

        self._refill()
        stats = self._stats[channel]
        stats.attempted_packets += 1
        stats.attempted_bytes += payload_bytes

        if self.in_rf_silence():
            stats.dropped_packets += 1
            stats.dropped_bytes += payload_bytes
            return False

        if random.random() < self.packet_loss:
            stats.dropped_packets += 1
            stats.dropped_bytes += payload_bytes
            return False

        if payload_bytes > self._tokens[channel]:
            stats.dropped_packets += 1
            stats.dropped_bytes += payload_bytes
            return False

        self._tokens[channel] -= payload_bytes
        stats.delivered_packets += 1
        stats.delivered_bytes += payload_bytes
        return True

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            channel: {
                "attempted_packets": stat.attempted_packets,
                "delivered_packets": stat.delivered_packets,
                "dropped_packets": stat.dropped_packets,
                "attempted_bytes": stat.attempted_bytes,
                "delivered_bytes": stat.delivered_bytes,
                "dropped_bytes": stat.dropped_bytes,
            }
            for channel, stat in self._stats.items()
        }
