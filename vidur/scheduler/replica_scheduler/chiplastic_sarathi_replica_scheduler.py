from typing import Optional

from vidur.chiplastic import ChiplasticRuntime, ChiplasticTuningConfig
from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)


class ChiplasticSarathiReplicaScheduler(SarathiReplicaScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = getattr(self._config, "chiplastic", ChiplasticTuningConfig())
        self._chiplastic_runtime = ChiplasticRuntime(
            replica_id=self._replica_id,
            tuning=config,
            num_initial_blocks=self._config.num_blocks,
        )

    def maybe_apply_chiplastic(
        self,
        now: float,
        batch: Batch,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
    ) -> Optional[ExecutionTime]:
        return self._chiplastic_runtime.on_stage_scheduled(
            now,
            batch,
            batch_stage,
            execution_time,
            self,
        )

    def write_chiplastic_metrics(self, output_dir: str) -> None:
        self._chiplastic_runtime.write_metrics(output_dir)
