from typing import Dict, List, Optional

from vidur.chiplastic import (
    ChipletMemoryManager,
    ChiplasticRuntime,
    ChiplasticTuningConfig,
)
from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)


class ChiplasticSarathiReplicaScheduler(SarathiReplicaScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tuning: ChiplasticTuningConfig = getattr(
            self._config, "chiplastic", ChiplasticTuningConfig()
        )
        dtype_bytes = self._tuning.hardware.kv_block_dtype_bytes
        self._memory_manager = ChipletMemoryManager(
            tuning=self._tuning,
            block_size_tokens=self._config.block_size,
            dtype_bytes=dtype_bytes,
        )
        self._config.num_blocks = self._memory_manager.total_blocks
        self._request_to_blocks: Dict[int, List[str]] = {}
        self._model_config = self._replica_config.model_config

        self._chiplastic_runtime = ChiplasticRuntime(
            replica_id=self._replica_id,
            tuning=self._tuning,
            num_initial_blocks=self._config.num_blocks,
            memory_manager=self._memory_manager,
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

    @property
    def memory_manager(self) -> ChipletMemoryManager:
        return self._memory_manager

    @property
    def model_config(self):
        return self._model_config

    def get_request_blocks(self, request_id: int) -> List[str]:
        return list(self._request_to_blocks.get(request_id, []))

    def allocate(self, request_id: int, num_blocks: int) -> None:  # type: ignore[override]
        allocated_blocks = self._memory_manager.allocate(
            request_id, num_blocks, locality_bias=self._tuning.dispatch_locality_bias
        )
        if allocated_blocks:
            self._request_to_blocks.setdefault(request_id, []).extend(allocated_blocks)
        super().allocate(request_id, num_blocks)

    def free(self, *request_ids: int) -> None:  # type: ignore[override]
        for request_id in request_ids:
            self._memory_manager.free(request_id)
            self._request_to_blocks.pop(request_id, None)
        super().free(*request_ids)
