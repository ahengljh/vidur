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


@dataclass
class ScalingDecision:
    state: ScalingState
    target_compute: int
    target_memory: int
    reason: str


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
        self._alpha = 0.25

    def update(self, observation: StageObservation) -> ScalingDecision:
        self._update_latency_ema(observation)
        if self._cooldown > 0:
            self._cooldown -= 1
            return ScalingDecision(self._infer_state(), self.active_compute, self.active_memory, "cooldown")

        thresholds = self._tuning.thresholds
        hardware = self._tuning.hardware
        reason = "steady"
        target_compute = self.active_compute
        target_memory = self.active_memory

        if observation.stage_type == StageType.PREFILL:
            if (
                observation.latency_ms > thresholds.prefill_latency_target_ms
                and self.active_compute < hardware.max_compute_dies
            ):
                target_compute = self.active_compute + 1
                reason = "prefill_latency"
        elif observation.stage_type == StageType.DECODE:
            remote_latency = observation.remote_latency_ms
            remote_fraction = observation.remote_profile.remote_fraction
            if (
                observation.memory_pressure > thresholds.memory_utilization_scale_up
                and self.active_memory < hardware.max_memory_dies
            ):
                target_memory = self.active_memory + 1
                reason = "memory_pressure"
            elif (
                remote_latency > thresholds.decode_latency_target_ms
                and remote_fraction > 0.2
                and self.active_memory < hardware.max_memory_dies
            ):
                target_memory = self.active_memory + 1
                reason = "remote_latency"
            elif (
                observation.latency_ms > thresholds.decode_latency_target_ms
                and self.active_compute < hardware.max_compute_dies
                and self.active_memory > self.active_compute
            ):
                target_compute = self.active_compute + 1
                reason = "bandwidth_assist"

        if reason == "steady":
            if (
                observation.memory_pressure < thresholds.memory_utilization_scale_down
                and self.active_memory > hardware.base_memory_dies
            ):
                target_memory = max(hardware.base_memory_dies, self.active_memory - 1)
                reason = "memory_scale_down"
            elif (
                self._prefill_avg_ms < thresholds.prefill_latency_target_ms * 0.7
                and self.active_compute > hardware.base_compute_dies
            ):
                target_compute = max(hardware.base_compute_dies, self.active_compute - 1)
                reason = "compute_scale_down"

        if target_compute != self.active_compute or target_memory != self.active_memory:
            self.active_compute = target_compute
            self.active_memory = target_memory
            self.state = self._infer_state()
            self._cooldown = thresholds.cooldown_steps
            return ScalingDecision(self.state, target_compute, target_memory, reason)

        self.state = self._infer_state()
        return ScalingDecision(self.state, target_compute, target_memory, reason)

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
        return ScalingState.BANDWIDTH


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

    def _apply_scaling(
        self,
        decision: ScalingDecision,
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
        removed = self._memory_manager.remove_helper_dies(count)
        if removed:
            replica_scheduler._config.num_blocks = self._memory_manager.total_blocks
        return removed == count

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
            adjusted = _scale_execution_time(
                observation.execution_time,
                compute_scale=compute_scale,
                memory_scale=1.0,
            )
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
        self._energy_joules += compute_energy + memory_energy

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
        }
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
