from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)


class ChiplasticGlobalScheduler(RoundRobinGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        return super().schedule()

    def on_simulation_end(self) -> None:
        output_dir = self._config.metrics_config.output_dir
        for scheduler in self._replica_schedulers.values():
            writer = getattr(scheduler, "write_chiplastic_metrics", None)
            if callable(writer):
                writer(output_dir)
