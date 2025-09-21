from dataclasses import dataclass, field


@dataclass
class ChiplasticScalingThresholds:
    """Control thresholds for elastic scaling decisions."""

    memory_utilization_scale_up: float = 0.92
    memory_utilization_scale_down: float = 0.55
    prefill_latency_target_ms: float = 25.0
    decode_latency_target_ms: float = 6.0
    cooldown_steps: int = 4


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

    def __post_init__(self) -> None:
        if not (0.0 <= self.prefetch_effectiveness <= 1.0):
            raise ValueError("prefetch_effectiveness must lie in [0, 1]")
        if not (0.0 <= self.dispatch_locality_bias <= 1.0):
            raise ValueError("dispatch_locality_bias must lie in [0, 1]")
