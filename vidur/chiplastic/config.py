from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChiplasticScalingThresholds:
    """Control thresholds for elastic scaling decisions."""

    memory_utilization_scale_up: float = 0.92
    memory_utilization_scale_down: float = 0.55
    prefill_latency_target_ms: float = 25.0
    decode_latency_target_ms: float = 6.0
    cooldown_steps: int = 4
    thermal_throttle_temp_c: float = field(
        default=75.0,
        metadata={"help": "Temperature threshold for thermal throttling (Celsius)"}
    )
    frequency_boost_threshold: float = field(
        default=0.85,
        metadata={"help": "Utilization threshold to trigger frequency boost"}
    )
    frequency_reduce_threshold: float = field(
        default=0.3,
        metadata={"help": "Utilization threshold to reduce frequency"}
    )


@dataclass
class ChiplasticHardwareConfig:
    """Describes the chiplet complex available to the simulator."""

    base_compute_dies: int = 2
    base_memory_dies: int = 2
    max_compute_dies: int = 8
    max_memory_dies: int = 8
    compute_tflops_per_die: float = 350.0
    memory_bandwidth_tbps_per_die: float = 0.65
    kv_blocks_per_die: int = 2048
    kv_block_dtype_bytes: int = 2
    compute_idle_power_w: float = 35.0
    compute_active_power_w: float = 170.0
    memory_idle_power_w: float = 9.0
    memory_active_power_w: float = 45.0
    remote_penalty_factor: float = 0.35
    interconnect_bandwidth_tbps: float = 1.6
    interconnect_latency_ns: float = 180.0
    # Enhanced hardware configs
    base_freq_ghz: float = field(
        default=1.5,
        metadata={"help": "Base frequency for compute dies (GHz)"}
    )
    boost_freq_ghz: float = field(
        default=2.5,
        metadata={"help": "Maximum boost frequency for compute dies (GHz)"}
    )
    min_freq_ghz: float = field(
        default=0.8,
        metadata={"help": "Minimum frequency for compute dies (GHz)"}
    )
    freq_steps: int = field(
        default=16,
        metadata={"help": "Number of frequency scaling steps"}
    )
    freq_transition_latency_us: float = field(
        default=10.0,
        metadata={"help": "Frequency transition latency (microseconds)"}
    )
    thermal_design_power_w: float = field(
        default=300.0,
        metadata={"help": "Thermal design power (watts)"}
    )
    max_temp_c: float = field(
        default=85.0,
        metadata={"help": "Maximum operating temperature (Celsius)"}
    )
    thermal_resistance: float = field(
        default=0.15,
        metadata={"help": "Thermal resistance (K/W)"}
    )
    thermal_capacitance: float = field(
        default=100.0,
        metadata={"help": "Thermal capacitance (J/K)"}
    )
    numa_nodes: int = field(
        default=4,
        metadata={"help": "Number of NUMA nodes"}
    )
    numa_local_latency_ns: float = field(
        default=50.0,
        metadata={"help": "NUMA local access latency (nanoseconds)"}
    )
    numa_remote_latency_ns: float = field(
        default=150.0,
        metadata={"help": "NUMA remote access latency (nanoseconds)"}
    )
    numa_far_latency_ns: float = field(
        default=300.0,
        metadata={"help": "NUMA far access latency (nanoseconds)"}
    )

    def __post_init__(self) -> None:
        if self.base_compute_dies <= 0 or self.base_memory_dies <= 0:
            raise ValueError("Base compute/memory dies must be positive")
        if self.max_compute_dies < self.base_compute_dies:
            raise ValueError("max_compute_dies must be >= base_compute_dies")
        if self.max_memory_dies < self.base_memory_dies:
            raise ValueError("max_memory_dies must be >= base_memory_dies")
        if self.kv_blocks_per_die <= 0:
            raise ValueError("kv_blocks_per_die must be positive")


@dataclass
class ChiplasticTuningConfig:
    """Aggregated tuning knobs for Chiplastic runtime."""

    hardware: ChiplasticHardwareConfig = field(
        default_factory=ChiplasticHardwareConfig
    )
    thresholds: ChiplasticScalingThresholds = field(
        default_factory=ChiplasticScalingThresholds
    )
    prefetch_effectiveness: float = 0.65
    dispatch_locality_bias: float = 0.7
    energy_reporting: bool = True
    enable_logging: bool = True
    # Enhanced tuning configs
    enable_frequency_scaling: bool = field(
        default=True,
        metadata={"help": "Enable dynamic frequency scaling"}
    )
    enable_thermal_management: bool = field(
        default=True,
        metadata={"help": "Enable thermal throttling and management"}
    )
    enable_numa_aware_placement: bool = field(
        default=True,
        metadata={"help": "Enable NUMA-aware memory placement"}
    )
    enable_granular_compute_control: bool = field(
        default=True,
        metadata={"help": "Enable per-component compute scaling"}
    )
    power_efficiency_mode: bool = field(
        default=False,
        metadata={"help": "Enable power efficiency optimizations"}
    )
    adaptive_scaling_alpha: float = field(
        default=0.25,
        metadata={"help": "EMA alpha for adaptive scaling decisions"}
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.prefetch_effectiveness <= 1.0):
            raise ValueError("prefetch_effectiveness must lie in [0, 1]")
        if not (0.0 <= self.dispatch_locality_bias <= 1.0):
            raise ValueError("dispatch_locality_bias must lie in [0, 1]")
