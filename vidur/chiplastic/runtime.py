from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

from vidur.chiplastic.config import ChiplasticTuningConfig
from vidur.chiplastic.interconnect import InterconnectModel
from vidur.chiplastic.memory import ChipletMemoryManager, RemoteAccessProfile
from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.logger import init_logger

logger = init_logger(__name__)


class StageType(Enum):
    PREFILL = auto()
    DECODE = auto()
    OTHER = auto()


class ScalingState(Enum):
    BASE = auto()
    MEMORY = auto()
    COMPUTE = auto()
    BANDWIDTH = auto()
    HYBRID = auto()  # Both compute and memory scaled


@dataclass
class ScalingDecision:
    state: ScalingState
    target_compute: int
    target_memory: int
    reason: str
    compute_frequencies: Dict[int, float] = None
    predicted_improvement: float = 0.0


@dataclass
class StageObservation:
    current_time: float
    stage_type: StageType
    execution_time: ExecutionTime
    batch_size: int
    total_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    memory_pressure: float
    allocated_blocks: int
    capacity_blocks: int
    remote_profile: RemoteAccessProfile
    remote_latency_ms: float
    remote_bytes: float
    local_bytes: float
    numa_distance: int = 10  # NUMA distance metric
    temperature_c: float = 25.0  # Current temperature
    power_w: float = 0.0  # Current power consumption
    compute_utilization: Dict[str, float] = None  # Per-component utilization

    @property
    def latency_ms(self) -> float:
        return self.execution_time.total_time * 1e3 + self.remote_latency_ms


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _scale_execution_time(
    execution_time: ExecutionTime,
    compute_scale: float,
    memory_scale: float,
) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=execution_time.num_layers,
        attention_rope_execution_time=
        execution_time.attention_rope_execution_time * memory_scale,
        attention_kv_cache_save_execution_time=
        execution_time.attention_kv_cache_save_execution_time * memory_scale,
        attention_decode_execution_time=
        execution_time.attention_decode_execution_time * memory_scale,
        attention_prefill_execution_time=
        execution_time.attention_prefill_execution_time * compute_scale,
        attention_layer_pre_proj_execution_time=
        execution_time.attention_pre_proj_time * compute_scale,
        attention_layer_post_proj_execution_time=
        execution_time.attention_post_proj_time * compute_scale,
        mlp_layer_up_proj_execution_time=
        execution_time.mlp_layer_up_proj_execution_time * compute_scale,
        mlp_layer_down_proj_execution_time=
        execution_time.mlp_layer_down_proj_execution_time * compute_scale,
        mlp_layer_act_execution_time=
        execution_time.mlp_layer_act_execution_time * compute_scale,
        attn_norm_time=execution_time.attn_norm_time * memory_scale,
        mlp_norm_time=execution_time.mlp_norm_time * compute_scale,
        add_time=execution_time.add_time * compute_scale,
        tensor_parallel_communication_time=
        execution_time.mlp_all_reduce_time * compute_scale,
        pipeline_parallel_communication_time=
        execution_time.pipeline_parallel_communication_time,
        schedule_time=execution_time.schedule_time,
        sampler_e2e_time=execution_time.sampler_e2e_time,
        prepare_inputs_e2e_time=execution_time.prepare_inputs_e2e_time,
        process_model_outputs_time=execution_time.process_model_outputs_time,
        ray_comm_time=execution_time.ray_comm_time,
    )


def _clone_execution_time(execution_time: ExecutionTime, **overrides) -> ExecutionTime:
    return ExecutionTime(
        num_layers_per_pipeline_stage=overrides.get(
            "num_layers_per_pipeline_stage", execution_time.num_layers
        ),
        attention_rope_execution_time=overrides.get(
            "attention_rope_execution_time", execution_time.attention_rope_execution_time
        ),
        attention_kv_cache_save_execution_time=overrides.get(
            "attention_kv_cache_save_execution_time",
            execution_time.attention_kv_cache_save_execution_time,
        ),
        attention_decode_execution_time=overrides.get(
            "attention_decode_execution_time",
            execution_time.attention_decode_execution_time,
        ),
        attention_prefill_execution_time=overrides.get(
            "attention_prefill_execution_time",
            execution_time.attention_prefill_execution_time,
        ),
        attention_layer_pre_proj_execution_time=overrides.get(
            "attention_layer_pre_proj_execution_time",
            execution_time.attention_pre_proj_time,
        ),
        attention_layer_post_proj_execution_time=overrides.get(
            "attention_layer_post_proj_execution_time",
            execution_time.attention_post_proj_time,
        ),
        mlp_layer_up_proj_execution_time=overrides.get(
            "mlp_layer_up_proj_execution_time",
            execution_time.mlp_layer_up_proj_execution_time,
        ),
        mlp_layer_down_proj_execution_time=overrides.get(
            "mlp_layer_down_proj_execution_time",
            execution_time.mlp_layer_down_proj_execution_time,
        ),
        mlp_layer_act_execution_time=overrides.get(
            "mlp_layer_act_execution_time",
            execution_time.mlp_layer_act_execution_time,
        ),
        attn_norm_time=overrides.get("attn_norm_time", execution_time.attn_norm_time),
        mlp_norm_time=overrides.get("mlp_norm_time", execution_time.mlp_norm_time),
        add_time=overrides.get("add_time", execution_time.add_time),
        tensor_parallel_communication_time=overrides.get(
            "tensor_parallel_communication_time",
            execution_time.mlp_all_reduce_time,
        ),
        pipeline_parallel_communication_time=overrides.get(
            "pipeline_parallel_communication_time",
            execution_time.pipeline_parallel_communication_time,
        ),
        schedule_time=overrides.get("schedule_time", execution_time.schedule_time),
        sampler_e2e_time=overrides.get("sampler_e2e_time", execution_time.sampler_e2e_time),
        prepare_inputs_e2e_time=overrides.get(
            "prepare_inputs_e2e_time", execution_time.prepare_inputs_e2e_time
        ),
        process_model_outputs_time=overrides.get(
            "process_model_outputs_time", execution_time.process_model_outputs_time
        ),
        ray_comm_time=overrides.get("ray_comm_time", execution_time.ray_comm_time),
    )


class ChiplasticController:
    def __init__(self, tuning: ChiplasticTuningConfig) -> None:
        self._tuning = tuning
        self.state = ScalingState.BASE
        self.active_compute = tuning.hardware.base_compute_dies
        self.active_memory = tuning.hardware.base_memory_dies
        self._cooldown = 0
        self._prefill_avg_ms = 0.0
        self._decode_avg_ms = 0.0
        self._alpha = tuning.adaptive_scaling_alpha
        # Enhanced state tracking
        self._compute_frequencies: Dict[int, float] = {}
        self._temperatures: Dict[int, float] = {}
        self._power_states: Dict[int, float] = {}
        self._numa_topology: Dict[int, Dict[int, int]] = {}
        self._initialize_enhanced_state()

    def _initialize_enhanced_state(self) -> None:
        """Initialize enhanced state tracking for compute resources."""
        for i in range(self._tuning.hardware.max_compute_dies):
            self._compute_frequencies[i] = self._tuning.hardware.base_freq_ghz
            self._temperatures[i] = 25.0
            self._power_states[i] = self._tuning.hardware.compute_idle_power_w

        # Initialize NUMA topology
        self._initialize_numa_topology()

    def _initialize_numa_topology(self) -> None:
        """Initialize NUMA distance matrix."""
        num_nodes = self._tuning.hardware.numa_nodes
        dies_per_node = self._tuning.hardware.max_compute_dies // num_nodes

        for i in range(self._tuning.hardware.max_compute_dies):
            self._numa_topology[i] = {}
            node_i = i // dies_per_node
            for j in range(self._tuning.hardware.max_memory_dies):
                node_j = j // dies_per_node
                if node_i == node_j:
                    self._numa_topology[i][j] = 10  # Local
                elif abs(node_i - node_j) == 1:
                    self._numa_topology[i][j] = 20  # Adjacent
                else:
                    self._numa_topology[i][j] = 30  # Remote

    def update(self, observation: StageObservation) -> ScalingDecision:
        self._update_latency_ema(observation)
        self._update_thermal_state(observation)

        logger.debug("[ChiplasticController] Stage: %s, Latency: %.2fms, Memory pressure: %.2f, Temp: %.1f°C",
                    observation.stage_type.name, observation.latency_ms,
                    observation.memory_pressure, observation.temperature_c)

        if self._cooldown > 0:
            self._cooldown -= 1
            logger.debug("[ChiplasticController] In cooldown period, remaining steps: %d", self._cooldown)
            return ScalingDecision(
                self._infer_state(),
                self.active_compute,
                self.active_memory,
                "cooldown",
                self._compute_frequencies
            )

        thresholds = self._tuning.thresholds
        hardware = self._tuning.hardware
        reason = "steady"
        target_compute = self.active_compute
        target_memory = self.active_memory
        predicted_improvement = 0.0

        # Consider thermal throttling
        if observation.temperature_c > thresholds.thermal_throttle_temp_c:
            logger.info("[ChiplasticController] Thermal throttling triggered: %.1f°C > %.1f°C",
                       observation.temperature_c, thresholds.thermal_throttle_temp_c)
            if self._tuning.enable_frequency_scaling:
                self._adjust_frequencies_for_thermal(observation)
                reason = "thermal_throttle"

        if observation.stage_type == StageType.PREFILL:
            if observation.latency_ms > thresholds.prefill_latency_target_ms:
                latency_ratio = observation.latency_ms / thresholds.prefill_latency_target_ms
                logger.info("[ChiplasticController] Prefill latency high: %.2fms > %.2fms (%.1fx target)",
                           observation.latency_ms, thresholds.prefill_latency_target_ms, latency_ratio)

                # Check if we should add compute dies based on severity
                should_add_compute = False
                dies_to_add = 0

                # Severe latency (>=2x) - aggressive scaling
                if (
                    self._tuning.aggressive_prefill_scaling
                    and latency_ratio >= thresholds.prefill_latency_severe_ratio
                    and self.active_compute < hardware.max_compute_dies
                ):
                    # Calculate optimal compute dies
                    efficiency = self._tuning.prefill_parallel_efficiency
                    needed_speedup = latency_ratio
                    current_power = self.active_compute
                    ideal_power = current_power * needed_speedup / efficiency
                    ideal_dies = min(int(ideal_power) + 1, hardware.max_compute_dies)
                    dies_to_add = min(ideal_dies - self.active_compute, 2)  # Add up to 2 at once

                    if dies_to_add > 0:
                        should_add_compute = True
                        logger.info("[ChiplasticController] SEVERE prefill latency (%.1fx) - adding %d compute dies: %d -> %d",
                                   latency_ratio, dies_to_add, self.active_compute, self.active_compute + dies_to_add)
                        target_compute = self.active_compute + dies_to_add
                        predicted_improvement = (dies_to_add * efficiency) / self.active_compute
                        reason = "prefill_latency_severe"

                # If not severe, try frequency boost first (but only if it's actually possible)
                if not should_add_compute and self._tuning.enable_frequency_scaling and self._can_boost_frequency():
                    current_max_freq = max(self._compute_frequencies.get(i, self._tuning.hardware.base_freq_ghz)
                                          for i in range(self.active_compute))
                    logger.info("[ChiplasticController] Boosting frequencies (current max: %.2f GHz)",
                               current_max_freq)
                    self._boost_compute_frequencies()
                    predicted_improvement = 0.15
                    reason = "prefill_frequency_boost"

                # If frequency boost not possible or not enough, add compute die
                elif not should_add_compute and self.active_compute < hardware.max_compute_dies:
                    logger.info("[ChiplasticController] Frequency boost exhausted, adding compute die: %d -> %d",
                               self.active_compute, self.active_compute + 1)
                    target_compute = self.active_compute + 1
                    predicted_improvement = self._tuning.prefill_parallel_efficiency / self.active_compute
                    reason = "prefill_latency"
                elif not should_add_compute:
                    logger.warning("[ChiplasticController] Cannot scale further - Max compute: %d, Max freq reached",
                                  self.active_compute)
        elif observation.stage_type == StageType.DECODE:
            remote_latency = observation.remote_latency_ms
            remote_fraction = observation.remote_profile.remote_fraction
            logger.debug("[ChiplasticController] Decode stage - Remote latency: %.2fms, Remote fraction: %.2f",
                        remote_latency, remote_fraction)

            if (
                observation.memory_pressure > thresholds.memory_utilization_scale_up
                and self.active_memory < hardware.max_memory_dies
            ):
                logger.info("[ChiplasticController] Memory pressure high: %.2f > %.2f, scaling memory %d -> %d",
                           observation.memory_pressure, thresholds.memory_utilization_scale_up,
                           self.active_memory, self.active_memory + 1)
                target_memory = self.active_memory + 1
                reason = "memory_pressure"
            elif (
                remote_latency > thresholds.decode_latency_target_ms
                and remote_fraction > 0.2
                and self.active_memory < hardware.max_memory_dies
            ):
                logger.info("[ChiplasticController] Remote access latency high: %.2fms, scaling memory %d -> %d",
                           remote_latency, self.active_memory, self.active_memory + 1)
                target_memory = self.active_memory + 1
                reason = "remote_latency"
            elif (
                observation.latency_ms > thresholds.decode_latency_target_ms
                and self.active_compute < hardware.max_compute_dies
                and self.active_memory > self.active_compute
            ):
                logger.info("[ChiplasticController] Decode latency high: %.2fms, adding compute for bandwidth",
                           observation.latency_ms)
                target_compute = self.active_compute + 1
                reason = "bandwidth_assist"

        if reason == "steady":
            # Try memory consolidation when utilization is low
            if (
                self._tuning.enable_memory_consolidation
                and observation.memory_pressure < self._tuning.consolidation_threshold
                and self.active_memory > hardware.base_memory_dies
                and self._cooldown == 0  # Only when not in cooldown
            ):
                # Check if we can consolidate to fewer dies
                total_blocks_needed = observation.allocated_blocks
                blocks_per_die = self._tuning.hardware.kv_blocks_per_die
                min_dies_needed = max(
                    (total_blocks_needed + blocks_per_die - 1) // blocks_per_die,
                    hardware.base_memory_dies
                )

                if min_dies_needed < self.active_memory:
                    # We can consolidate
                    logger.info("[ChiplasticController] Memory consolidation possible: %d dies -> %d dies",
                               self.active_memory, min_dies_needed)
                    target_memory = min_dies_needed
                    reason = "memory_consolidation"
                    predicted_improvement = (self.active_memory - target_memory) * \
                                          hardware.memory_active_power_w / 1000.0  # Power savings
                else:
                    # Try normal scale down
                    logger.info("[ChiplasticController] Scaling down memory: %d -> %d",
                               self.active_memory, max(hardware.base_memory_dies, self.active_memory - 1))
                    target_memory = max(hardware.base_memory_dies, self.active_memory - 1)
                    reason = "memory_scale_down"
            elif (
                self._prefill_avg_ms < thresholds.prefill_latency_target_ms * 0.5  # More conservative
                and self._decode_avg_ms < thresholds.decode_latency_target_ms * 0.5
                and self.active_compute > hardware.base_compute_dies
                and self._cooldown == 0  # Only when not in cooldown
            ):
                logger.info("[ChiplasticController] Low latency, scaling down compute: %d -> %d",
                           self.active_compute, max(hardware.base_compute_dies, self.active_compute - 1))
                target_compute = max(hardware.base_compute_dies, self.active_compute - 1)
                reason = "compute_scale_down"

        if target_compute != self.active_compute or target_memory != self.active_memory:
            logger.info("[ChiplasticController] Scaling decision: Compute %d->%d, Memory %d->%d, Reason: %s",
                       self.active_compute, target_compute, self.active_memory, target_memory, reason)
            self.active_compute = target_compute
            self.active_memory = target_memory
            self.state = self._infer_state()
            self._cooldown = thresholds.cooldown_steps
            return ScalingDecision(
                self.state,
                target_compute,
                target_memory,
                reason,
                self._compute_frequencies,
                predicted_improvement
            )

        self.state = self._infer_state()
        return ScalingDecision(
            self.state,
            target_compute,
            target_memory,
            reason,
            self._compute_frequencies,
            predicted_improvement
        )

    def _update_latency_ema(self, observation: StageObservation) -> None:
        latency = observation.latency_ms
        if observation.stage_type == StageType.PREFILL:
            self._prefill_avg_ms = (
                self._alpha * latency + (1 - self._alpha) * self._prefill_avg_ms
                if self._prefill_avg_ms
                else latency
            )
        elif observation.stage_type == StageType.DECODE:
            self._decode_avg_ms = (
                self._alpha * latency + (1 - self._alpha) * self._decode_avg_ms
                if self._decode_avg_ms
                else latency
            )

    def _infer_state(self) -> ScalingState:
        base_compute = self._tuning.hardware.base_compute_dies
        base_memory = self._tuning.hardware.base_memory_dies
        if self.active_compute == base_compute and self.active_memory == base_memory:
            return ScalingState.BASE
        if self.active_memory > base_memory and self.active_compute == base_compute:
            return ScalingState.MEMORY
        if self.active_compute > base_compute and self.active_memory == base_memory:
            return ScalingState.COMPUTE
        if self.active_compute > base_compute and self.active_memory > base_memory:
            return ScalingState.HYBRID
        return ScalingState.BANDWIDTH

    def _update_thermal_state(self, observation: StageObservation) -> None:
        """Update thermal state based on power consumption."""
        if not self._tuning.enable_thermal_management:
            return

        dt = 0.001  # Time step
        for i in range(self.active_compute):
            power = self._calculate_die_power(i, observation)
            temp_rise = (power * self._tuning.hardware.thermal_resistance -
                        (self._temperatures[i] - 25.0)) * dt / self._tuning.hardware.thermal_capacitance
            self._temperatures[i] = _clamp(
                self._temperatures[i] + temp_rise,
                25.0,
                self._tuning.hardware.max_temp_c
            )

    def _calculate_die_power(self, die_id: int, observation: StageObservation) -> float:
        """Calculate power consumption for a compute die."""
        if die_id >= self.active_compute:
            return self._tuning.hardware.compute_idle_power_w

        base_power = self._tuning.hardware.compute_active_power_w
        freq_ratio = self._compute_frequencies[die_id] / self._tuning.hardware.base_freq_ghz

        # Power scales with frequency squared (simplified model)
        power = base_power * (freq_ratio ** 2)

        # Scale by utilization
        if observation.compute_utilization:
            avg_util = sum(observation.compute_utilization.values()) / len(observation.compute_utilization)
            power *= avg_util

        return power

    def _can_boost_frequency(self) -> bool:
        """Check if frequency boost is possible."""
        # Check thermal headroom
        max_temp = max(self._temperatures[i] for i in range(self.active_compute))
        if max_temp >= self._tuning.thresholds.thermal_throttle_temp_c:
            return False

        # Check if any die can still be boosted
        for i in range(self.active_compute):
            current_freq = self._compute_frequencies.get(i, self._tuning.hardware.base_freq_ghz)
            if current_freq < self._tuning.hardware.boost_freq_ghz * 0.95:  # Allow 5% margin
                return True

        return False

    def _boost_compute_frequencies(self) -> None:
        """Boost compute frequencies for active dies."""
        boosted_any = False
        for i in range(self.active_compute):
            current_freq = self._compute_frequencies[i]
            max_freq = self._tuning.hardware.boost_freq_ghz

            # Only boost if there's headroom
            if current_freq < max_freq * 0.95:  # 5% margin
                new_freq = min(current_freq * 1.2, max_freq)
                logger.debug("[ChiplasticController] Boosting die %d frequency: %.2f -> %.2f GHz",
                            i, current_freq, new_freq)
                self._compute_frequencies[i] = new_freq
                boosted_any = True
            else:
                logger.debug("[ChiplasticController] Die %d already at max frequency: %.2f GHz",
                            i, current_freq)

        if not boosted_any:
            logger.info("[ChiplasticController] All dies already at maximum frequency")

    def _adjust_frequencies_for_thermal(self, observation: StageObservation) -> None:
        """Adjust frequencies to manage thermal constraints."""
        for i in range(self.active_compute):
            if self._temperatures[i] > self._tuning.thresholds.thermal_throttle_temp_c:
                old_freq = self._compute_frequencies[i]
                self._compute_frequencies[i] *= 0.9
                self._compute_frequencies[i] = max(
                    self._compute_frequencies[i],
                    self._tuning.hardware.min_freq_ghz
                )
                logger.info("[ChiplasticController] Thermal throttling die %d: %.2f -> %.2f GHz (Temp: %.1f°C)",
                           i, old_freq, self._compute_frequencies[i], self._temperatures[i])


class ChiplasticRuntime:
    def __init__(
        self,
        replica_id: int,
        tuning: ChiplasticTuningConfig,
        num_initial_blocks: int,
        memory_manager: ChipletMemoryManager,
    ) -> None:
        self._replica_id = replica_id
        self._tuning = tuning
        self._controller = ChiplasticController(tuning)
        self._history: list[Dict[str, float]] = []
        self._base_blocks = num_initial_blocks
        self._memory_manager = memory_manager
        self._interconnect = InterconnectModel(
            bandwidth_tbps=tuning.hardware.interconnect_bandwidth_tbps,
            base_latency_ns=tuning.hardware.interconnect_latency_ns,
        )
        self._energy_joules = 0.0
        self._compute_energy_j = 0.0
        self._memory_energy_j = 0.0
        self._dtype_bytes = tuning.hardware.kv_block_dtype_bytes

    @property
    def active_compute(self) -> int:
        return self._controller.active_compute

    @property
    def active_memory(self) -> int:
        return self._controller.active_memory

    @property
    def state(self) -> ScalingState:
        return self._controller.state

    def on_stage_scheduled(
        self,
        now: float,
        batch: Batch,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
        replica_scheduler,
    ) -> ExecutionTime:
        stage_type = self._infer_stage_type(batch)
        allocated_blocks = replica_scheduler.num_allocated_blocks
        capacity_blocks = max(self._memory_manager.total_blocks, 1)
        pressure = allocated_blocks / capacity_blocks

        remote_profile = self._memory_manager.remote_profile(
            batch.request_ids,
            active_compute=self.active_compute,
            active_memory=self.active_memory,
        )
        stage_bytes = self._estimate_stage_bytes(batch, stage_type, replica_scheduler)
        remote_bytes_raw = stage_bytes * remote_profile.remote_fraction
        effective_remote_bytes = remote_bytes_raw * (1.0 - self._tuning.prefetch_effectiveness)
        local_bytes = max(stage_bytes - remote_bytes_raw, 0.0)
        remote_stats = self._interconnect.estimate(
            bytes_requested=effective_remote_bytes,
            hops=1,
            parallel_transfers=max(1, self.active_compute),
        )
        remote_latency_ms = remote_stats.latency_s * 1e3

        observation = StageObservation(
            current_time=now,
            stage_type=stage_type,
            execution_time=execution_time,
            batch_size=batch.size,
            total_tokens=batch.total_num_tokens,
            num_prefill_tokens=batch.num_prefill_tokens,
            num_decode_tokens=batch.num_decode_tokens,
            memory_pressure=pressure,
            allocated_blocks=allocated_blocks,
            capacity_blocks=capacity_blocks,
            remote_profile=remote_profile,
            remote_latency_ms=remote_latency_ms,
            remote_bytes=effective_remote_bytes,
            local_bytes=local_bytes,
        )

        prev_compute = self.active_compute
        prev_memory = self.active_memory
        decision = self._controller.update(observation)
        if decision.reason not in {"steady", "cooldown"}:
            self._apply_scaling(decision, replica_scheduler, prev_compute, prev_memory)

        adjusted_execution = self._adjust_execution_time(
            observation,
            replica_scheduler,
        )
        if self._tuning.enable_logging:
            self._record_history(now, observation, adjusted_execution)
        return adjusted_execution

    def write_metrics(self, output_dir: str) -> None:
        if not self._tuning.enable_logging or not self._history:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"chiplastic_replica_{self._replica_id}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._history, handle, indent=2)

        summary = {
            "replica_id": self._replica_id,
            "total_energy_joules": self._energy_joules,
            "compute_energy_joules": self._compute_energy_j,
            "memory_energy_joules": self._memory_energy_j,
            "final_active_compute": self.active_compute,
            "final_active_memory": self.active_memory,
            "history_entries": len(self._history),
        }
        if self._history:
            summary["duration_seconds"] = self._history[-1]["time"]
            if summary["duration_seconds"] > 0:
                summary["avg_power_watts"] = (
                    self._energy_joules / summary["duration_seconds"]
                )
        summary_path = os.path.join(
            output_dir, f"chiplastic_replica_{self._replica_id}_energy.json"
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _apply_scaling(
        self,
        decision,  # ScalingDecision
        replica_scheduler,
        prev_compute: int,
        prev_memory: int,
    ) -> None:
        target_memory = decision.target_memory
        memory_applied = True
        if decision.target_memory != prev_memory:
            delta = decision.target_memory - prev_memory
            if delta > 0:
                self._grow_memory(replica_scheduler, delta)
            elif delta < 0:
                memory_applied = self._shrink_memory(replica_scheduler, -delta)
                if not memory_applied:
                    target_memory = prev_memory

        self._controller.active_memory = target_memory
        self._controller.active_compute = decision.target_compute
        self._controller.state = self._controller._infer_state()

    def _grow_memory(self, replica_scheduler, count: int) -> bool:
        added = self._memory_manager.add_helper_dies(count)
        if added:
            replica_scheduler._config.num_blocks = self._memory_manager.total_blocks
        return added == count

    def _shrink_memory(self, replica_scheduler, count: int) -> bool:
        # First try to consolidate memory if enabled and possible
        if self._tuning.enable_memory_consolidation and self._memory_manager.can_consolidate_memory():
            freed = self._memory_manager.consolidate_memory()
            if freed > 0:
                replica_scheduler._config.num_blocks = self._memory_manager.total_blocks
                logger.debug("Consolidated memory and freed %d dies", freed)
                return freed >= count

        # If consolidation wasn't enough, check for directly removable dies
        removable_count = self._memory_manager.get_removable_helper_dies_count()
        if removable_count == 0:
            return False  # Cannot shrink, all dies have allocated blocks

        actual_count = min(count, removable_count)
        removed = self._memory_manager.remove_helper_dies(actual_count)
        if removed:
            replica_scheduler._config.num_blocks = self._memory_manager.total_blocks
        return removed == actual_count

    def _adjust_execution_time(
        self,
        observation: StageObservation,
        replica_scheduler,
    ) -> ExecutionTime:
        hardware = self._tuning.hardware
        stage_type = observation.stage_type

        base_compute_capacity = (
            hardware.base_compute_dies * hardware.compute_tflops_per_die
        )
        active_compute_capacity = (
            self.active_compute * hardware.compute_tflops_per_die
        )
        remote_ratio = observation.remote_profile.remote_fraction

        dispatch_modifier = 1.0 - remote_ratio * (1.0 - self._tuning.dispatch_locality_bias)
        effective_compute_capacity = max(active_compute_capacity * dispatch_modifier, 1e-3)
        compute_scale = base_compute_capacity / effective_compute_capacity
        compute_scale = _clamp(compute_scale, 0.35, 2.5)

        base_bandwidth = hardware.base_memory_dies * hardware.memory_bandwidth_tbps_per_die
        active_bandwidth = max(
            self.active_memory * hardware.memory_bandwidth_tbps_per_die, 1e-3
        )
        prefetch_modifier = 1.0 - remote_ratio * (1.0 - self._tuning.prefetch_effectiveness)
        effective_bandwidth = active_bandwidth * prefetch_modifier
        memory_scale = base_bandwidth / max(effective_bandwidth, 1e-3)
        memory_scale = _clamp(memory_scale, 0.35, 3.0)

        if stage_type == StageType.PREFILL:
            # Prefill is compute-bound and scales efficiently with more compute
            # Use configured parallel efficiency factor
            parallel_efficiency = self._tuning.prefill_parallel_efficiency
            effective_compute_scale = compute_scale * parallel_efficiency

            logger.debug("[ChiplasticRuntime] Prefill scaling - Base scale: %.2fx, Parallel efficiency: %.2f, Effective: %.2fx, Active dies: %d",
                        compute_scale, parallel_efficiency, effective_compute_scale, self.active_compute)

            # Apply the scaling
            adjusted = _scale_execution_time(
                observation.execution_time,
                compute_scale=effective_compute_scale,
                memory_scale=1.0,
            )

            # Log expected speedup
            if self.active_compute > self._tuning.hardware.base_compute_dies:
                speedup = self._tuning.hardware.base_compute_dies / (self.active_compute * parallel_efficiency)
                logger.debug("[ChiplasticRuntime] Prefill speedup from compute scaling: %.2fx", 1.0/speedup)
        elif stage_type == StageType.DECODE:
            adjusted = _scale_execution_time(
                observation.execution_time,
                compute_scale=1.0,
                memory_scale=memory_scale,
            )
        else:
            adjusted = observation.execution_time

        remote_latency_s = max(observation.remote_latency_ms, 0.0) / 1e3
        if remote_latency_s > 0:
            penalty_us = remote_latency_s * 1e6
            if stage_type == StageType.DECODE:
                adjusted = _clone_execution_time(
                    adjusted,
                    attention_decode_execution_time=
                    adjusted.attention_decode_execution_time + penalty_us,
                )
            elif stage_type == StageType.PREFILL:
                adjusted = _clone_execution_time(
                    adjusted,
                    attention_prefill_execution_time=
                    adjusted.attention_prefill_execution_time + penalty_us,
                )

        if self._tuning.energy_reporting:
            self._accumulate_energy(adjusted.total_time)
        return adjusted

    def _estimate_stage_bytes(self, batch: Batch, stage_type: StageType, replica_scheduler) -> float:
        model_config = getattr(replica_scheduler, "model_config", None)
        hidden_dim = getattr(model_config, "embedding_dim", 4096)
        num_layers = getattr(model_config, "num_layers", 1)

        dtype_bytes = self._dtype_bytes

        if stage_type == StageType.PREFILL:
            tokens = max(batch.num_prefill_tokens, 0)
            bytes_per_token = hidden_dim * dtype_bytes * 3  # Q, K, V projections
        elif stage_type == StageType.DECODE:
            tokens = max(batch.num_decode_tokens, batch.size)
            bytes_per_token = hidden_dim * dtype_bytes * 2  # K and V fetches
        else:
            return 0.0

        return float(tokens * bytes_per_token * num_layers)

    def _accumulate_energy(self, stage_time_s: float) -> None:
        hardware = self._tuning.hardware
        compute_energy = (
            self.active_compute * hardware.compute_active_power_w * stage_time_s
        )
        memory_energy = (
            self.active_memory * hardware.memory_active_power_w * stage_time_s
        )
        self._compute_energy_j += compute_energy
        self._memory_energy_j += memory_energy
        self._energy_joules = self._compute_energy_j + self._memory_energy_j

    def _record_history(
        self,
        now: float,
        observation: StageObservation,
        adjusted_execution: ExecutionTime,
    ) -> None:
        entry = {
            "time": now,
            "stage": observation.stage_type.name,
            "base_time_ms": observation.execution_time.total_time * 1e3,
            "adjusted_time_ms": adjusted_execution.total_time * 1e3,
            "state": self.state.name,
            "active_compute": float(self.active_compute),
            "active_memory": float(self.active_memory),
            "memory_pressure": observation.memory_pressure,
            "remote_fraction": observation.remote_profile.remote_fraction,
            "remote_latency_ms": observation.remote_latency_ms,
            "remote_bytes": observation.remote_bytes,
            "local_bytes": observation.local_bytes,
            "energy_joules_total": self._energy_joules,
            "compute_energy_joules_total": self._compute_energy_j,
            "memory_energy_joules_total": self._memory_energy_j,
        }
        duration = now if now > 0 else None
        if duration:
            entry["avg_power_watts"] = self._energy_joules / duration
        self._history.append(entry)

    @staticmethod
    def _infer_stage_type(batch: Batch) -> StageType:
        if batch.num_prefill_tokens > 0 and batch.num_decode_tokens == 0:
            return StageType.PREFILL
        if batch.num_prefill_tokens == 0 and batch.num_decode_tokens > 0:
            return StageType.DECODE
        if batch.num_prefill_tokens > 0 and batch.num_decode_tokens > 0:
            # Mixed batches occur during streaming decode; treat as decode-dominant
            return StageType.DECODE
        return StageType.OTHER

    def _calculate_numa_distance(self) -> int:
        """Calculate average NUMA distance for active resources."""
        if not self._tuning.enable_numa_aware_placement:
            return 10

        total_distance = 0
        count = 0

        for compute_die in range(self.active_compute):
            for memory_die in range(self.active_memory):
                distance = self._numa_distances.get(compute_die, {}).get(memory_die, 10)
                total_distance += distance
                count += 1

        return total_distance // max(count, 1) if count > 0 else 10

    def _numa_distance_to_hops(self, distance: int) -> int:
        """Convert NUMA distance to hop count."""
        if distance <= 10:
            return 1
        elif distance <= 20:
            return 2
        else:
            return 3

    def _estimate_compute_utilization(
        self, batch: Batch, stage_type: StageType, execution_time: ExecutionTime
    ) -> Dict[str, float]:
        """Estimate utilization of different compute components."""
        if not execution_time:
            return {}

        total_time = execution_time.total_time
        if total_time <= 0:
            return {}

        utilization = {}

        if stage_type == StageType.PREFILL:
            utilization["attention"] = execution_time.attention_prefill_execution_time / total_time
            utilization["mlp"] = (
                execution_time.mlp_layer_up_proj_execution_time +
                execution_time.mlp_layer_down_proj_execution_time
            ) / total_time
        elif stage_type == StageType.DECODE:
            utilization["attention"] = execution_time.attention_decode_execution_time / total_time
            utilization["mlp"] = execution_time.mlp_layer_act_execution_time / total_time
        else:
            utilization["attention"] = 0.0
            utilization["mlp"] = 0.0

        utilization["norm"] = (
            execution_time.attn_norm_time + execution_time.mlp_norm_time
        ) / total_time
        utilization["communication"] = execution_time.mlp_all_reduce_time / total_time

        self._compute_utilization = utilization
        return utilization

    def _get_max_temperature(self) -> float:
        """Get maximum temperature across active compute dies."""
        if not self._tuning.enable_thermal_management:
            return 25.0

        return max(
            self._controller._temperatures.get(i, 25.0)
            for i in range(self.active_compute)
        )

    def _calculate_total_power(self) -> float:
        """Calculate total power consumption."""
        total_power = 0.0

        # Compute power
        for i in range(self.active_compute):
            total_power += self._controller._power_states.get(
                i, self._tuning.hardware.compute_idle_power_w
            )

        # Memory power
        total_power += self.active_memory * self._tuning.hardware.memory_active_power_w

        return total_power

    def _calculate_effective_compute_capacity(self) -> float:
        """Calculate effective compute capacity with frequency scaling."""
        if not self._tuning.enable_frequency_scaling:
            return self.active_compute * self._tuning.hardware.compute_tflops_per_die

        total_capacity = 0.0
        for i in range(self.active_compute):
            freq_ratio = (
                self._controller._compute_frequencies.get(i, self._tuning.hardware.base_freq_ghz) /
                self._tuning.hardware.base_freq_ghz
            )
            total_capacity += self._tuning.hardware.compute_tflops_per_die * freq_ratio

        return total_capacity

    def _calculate_numa_bandwidth_modifier(self, numa_distance: int) -> float:
        """Calculate bandwidth modifier based on NUMA distance."""
        if not self._tuning.enable_numa_aware_placement:
            return 1.0

        # Local access = 1.0, adjacent = 0.8, remote = 0.5
        if numa_distance <= 10:
            return 1.0
        elif numa_distance <= 20:
            return 0.8
        else:
            return 0.5

    def _scale_execution_time_granular(
        self,
        execution_time: ExecutionTime,
        compute_scale: float,
        memory_scale: float,
        utilization: Dict[str, float],
    ) -> ExecutionTime:
        """Apply granular scaling based on component utilization."""
        if not utilization:
            # Fallback to regular scaling
            return _scale_execution_time(execution_time, compute_scale, memory_scale)

        # Scale different components based on their resource requirements
        attention_scale = compute_scale if utilization.get("attention", 0) > 0.5 else memory_scale
        mlp_scale = compute_scale
        norm_scale = (compute_scale + memory_scale) / 2
        comm_scale = 1.0  # Communication doesn't scale with compute/memory

        return ExecutionTime(
            num_layers_per_pipeline_stage=execution_time.num_layers,
            attention_rope_execution_time=execution_time.attention_rope_execution_time * memory_scale,
            attention_kv_cache_save_execution_time=execution_time.attention_kv_cache_save_execution_time * memory_scale,
            attention_decode_execution_time=execution_time.attention_decode_execution_time * attention_scale,
            attention_prefill_execution_time=execution_time.attention_prefill_execution_time * attention_scale,
            attention_layer_pre_proj_execution_time=execution_time.attention_pre_proj_time * compute_scale,
            attention_layer_post_proj_execution_time=execution_time.attention_post_proj_time * compute_scale,
            mlp_layer_up_proj_execution_time=execution_time.mlp_layer_up_proj_execution_time * mlp_scale,
            mlp_layer_down_proj_execution_time=execution_time.mlp_layer_down_proj_execution_time * mlp_scale,
            mlp_layer_act_execution_time=execution_time.mlp_layer_act_execution_time * mlp_scale,
            attn_norm_time=execution_time.attn_norm_time * norm_scale,
            mlp_norm_time=execution_time.mlp_norm_time * norm_scale,
            add_time=execution_time.add_time * compute_scale,
            tensor_parallel_communication_time=execution_time.mlp_all_reduce_time * comm_scale,
            pipeline_parallel_communication_time=execution_time.pipeline_parallel_communication_time,
            schedule_time=execution_time.schedule_time,
            sampler_e2e_time=execution_time.sampler_e2e_time,
            prepare_inputs_e2e_time=execution_time.prepare_inputs_e2e_time,
            process_model_outputs_time=execution_time.process_model_outputs_time,
            ray_comm_time=execution_time.ray_comm_time,
        )
