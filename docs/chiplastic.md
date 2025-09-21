# Chiplastic Extensions

This repository includes an implementation of **Chiplastic**, the elastic chiplet
resource management scheme described in *Chiplastic: Hardware-Software Co-Design
for Elastic LLM Inference on Chiplet GPUs*.

## Key Components

- **ChiplasticSarathiReplicaScheduler** (`ReplicaSchedulerType.CHIPLASTIC_SARATHI`)
  augments the standard Sarathi scheduler with a `ChiplasticRuntime` that tracks
  memory pressure and per-stage latency, triggers independent scaling, and
  adjusts execution time predictions to reflect NUMA-aware execution.
- **ChiplasticGlobalScheduler** (`GlobalSchedulerType.CHIPLASTIC`) orchestrates
  round-robin request routing and exports per-replica Chiplastic traces on
  simulation completion.
- **ChiplasticRuntime** models elastic scaling states (base, memory, compute,
  bandwidth), now tracks per-stage local/remote KV utilisation, interconnect
  latency, and energy, logging telemetry to
  `simulator_output/chiplastic_replica_<ID>.json`.

## Configuration

Select the new schedulers via CLI or configuration files:

```bash
python -m vidur.main \
  --cluster_config_global_scheduler_config_type chiplastic \
  --cluster_config_replica_scheduler_config_type chiplastic_sarathi
```

`ChiplasticSarathiSchedulerConfig` exposes an embedded
`chiplastic` field (see `vidur/chiplastic/config.py`) for tuning thresholds,
hardware limits, and prefetch/dispatch effectiveness.

### Example overrides

```bash
python -m vidur.main \
  --cluster_config_replica_scheduler_config_chiplastic_hardware_max_compute_dies 6 \
  --cluster_config_replica_scheduler_config_chiplastic_thresholds_prefill_latency_target_ms 18
```

## Output

For each replica, a JSON trace captures:

- Stage type, base vs adjusted latency (ms)
- Active compute/memory die counts and scaling state
- Memory pressure and cumulative energy estimate (J)
- Remote KV fraction, effective remote bytes transferred, and per-stage
  interconnect latency, enabling latency breakdown plots for the paper.

These traces are written alongside standard Vidur metrics and can be consumed
for the evaluations outlined in the Chiplastic paper (tail latency, throughput,
energy breakdown, ablations over scaling states, etc.).

## Experiments

1. **Baseline vs Chiplastic** – run identical workloads once with
   `ReplicaSchedulerType.SARATHI` + `GlobalSchedulerType.ROUND_ROBIN` and once
   with the Chiplastic variants to measure TTFT/TPOT improvements and tail
   latency reduction.
2. **Scaling Ablations** – vary `max_compute_dies` or `max_memory_dies` to study
   memory-only, compute-only, and full bandwidth scaling.
3. **Prefetch/Dispatch Sensitivity** – sweep `prefetch_effectiveness` and
   `dispatch_locality_bias` to quantify benefits of NUMA-aware optimizations.
4. **Energy Analysis** – leverage the exported energy counters to report energy
   per generated token and compute/memory power draw across scaling states.

The provided runtime hooks and metrics simplify scripting these experiments in
`sim.py` or bespoke notebooks.
