from __future__ import annotations

from dataclasses import dataclass

from vidur.logger import init_logger

logger = init_logger(__name__)


@dataclass
class InterconnectStats:
    bytes_transferred: float
    latency_s: float
    hops: int
    effective_bandwidth_gbps: float = 0.0
    congestion_factor: float = 1.0


class InterconnectModel:
    """Enhanced latency/bandwidth model for chiplet interconnect with NUMA awareness."""

    def __init__(self, bandwidth_tbps: float, base_latency_ns: float) -> None:
        # Convert to bytes/second and seconds
        self._bandwidth_bytes = bandwidth_tbps * 1e12 / 8.0
        self._base_latency_s = base_latency_ns * 1e-9
        self._congestion_history: List[float] = []
        self._max_history = 10

    def estimate(self, bytes_requested: float, hops: int = 1, parallel_transfers: int = 1) -> InterconnectStats:
        if bytes_requested <= 0:
            return InterconnectStats(
                bytes_transferred=0.0,
                latency_s=0.0,
                hops=hops,
                effective_bandwidth_gbps=0.0,
                congestion_factor=1.0
            )

        # Model congestion based on parallel transfers
        congestion_factor = self._calculate_congestion(parallel_transfers)

        # Calculate effective bandwidth with congestion
        base_bandwidth_per_transfer = self._bandwidth_bytes / max(parallel_transfers, 1)
        effective_bandwidth = base_bandwidth_per_transfer * congestion_factor
        effective_bandwidth = max(effective_bandwidth, 1e-6)

        # Calculate transfer time
        transfer_time = bytes_requested / effective_bandwidth

        # Calculate latency with hop penalty
        hop_penalty = self._calculate_hop_penalty(hops)
        latency = hops * self._base_latency_s * hop_penalty + transfer_time

        # Convert bandwidth to Gbps for reporting
        effective_bandwidth_gbps = (effective_bandwidth * 8) / 1e9

        return InterconnectStats(
            bytes_transferred=bytes_requested,
            latency_s=latency,
            hops=hops,
            effective_bandwidth_gbps=effective_bandwidth_gbps,
            congestion_factor=congestion_factor
        )

    def _calculate_congestion(self, parallel_transfers: int) -> float:
        """Calculate congestion factor based on parallel transfers."""
        if parallel_transfers <= 1:
            return 1.0
        elif parallel_transfers <= 4:
            return 0.9
        elif parallel_transfers <= 8:
            return 0.75
        else:
            return 0.6

    def _calculate_hop_penalty(self, hops: int) -> float:
        """Calculate latency penalty based on hop count."""
        if hops <= 1:
            return 1.0
        elif hops == 2:
            return 1.2
        elif hops == 3:
            return 1.5
        else:
            return 2.0

    def estimate_numa(
        self,
        bytes_requested: float,
        numa_distance: int,
        parallel_transfers: int = 1
    ) -> InterconnectStats:
        """Estimate interconnect performance with NUMA distance."""
        # Convert NUMA distance to hops
        if numa_distance <= 10:
            hops = 1
        elif numa_distance <= 20:
            hops = 2
        else:
            hops = 3

        return self.estimate(bytes_requested, hops, parallel_transfers)
