from __future__ import annotations

from dataclasses import dataclass

from vidur.logger import init_logger

logger = init_logger(__name__)


@dataclass
class InterconnectStats:
    bytes_transferred: float
    latency_s: float
    hops: int


class InterconnectModel:
    """Simple latency/bandwidth model for chiplet interconnect."""

    def __init__(self, bandwidth_tbps: float, base_latency_ns: float) -> None:
        # Convert to bytes/second and seconds
        self._bandwidth_bytes = bandwidth_tbps * 1e12 / 8.0
        self._base_latency_s = base_latency_ns * 1e-9

    def estimate(self, bytes_requested: float, hops: int = 1, parallel_transfers: int = 1) -> InterconnectStats:
        if bytes_requested <= 0:
            return InterconnectStats(bytes_transferred=0.0, latency_s=0.0, hops=hops)

        effective_bandwidth = max(self._bandwidth_bytes / max(parallel_transfers, 1), 1e-6)
        transfer_time = bytes_requested / effective_bandwidth
        latency = hops * self._base_latency_s + transfer_time
        return InterconnectStats(bytes_transferred=bytes_requested, latency_s=latency, hops=hops)
