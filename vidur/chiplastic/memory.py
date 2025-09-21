from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from vidur.chiplastic.config import ChiplasticTuningConfig
from vidur.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MemoryBlock:
    block_id: str
    die_id: int


@dataclass
class RemoteAccessProfile:
    remote_fraction: float
    local_fraction: float
    remote_blocks: int
    total_blocks: int


@dataclass
class MemoryDie:
    die_id: int
    capacity_blocks: int
    tier: str  # "base" or "helper"
    free_blocks: List[str] = field(default_factory=list)

    @property
    def free_count(self) -> int:
        return len(self.free_blocks)


class ChipletMemoryManager:
    """Tracks KV cache block placement across memory chiplets."""

    def __init__(self, tuning: ChiplasticTuningConfig, block_size_tokens: int, dtype_bytes: int = 2) -> None:
        self._tuning = tuning
        self._block_size_tokens = block_size_tokens
        self._dtype_bytes = dtype_bytes
        self._dies: Dict[int, MemoryDie] = {}
        self._allocations: Dict[int, List[str]] = {}
        self._block_home: Dict[str, int] = {}
        self._next_block_id = 0
        self._base_memory_dies = tuning.hardware.base_memory_dies
        self._blocks_per_die = max(1, tuning.hardware.kv_blocks_per_die)

        for _ in range(self._base_memory_dies):
            self._add_die(tier="base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def block_size_bytes(self) -> int:
        return self._block_size_tokens * self._dtype_bytes

    @property
    def total_blocks(self) -> int:
        return sum(d.capacity_blocks for d in self._dies.values())

    @property
    def free_blocks(self) -> int:
        return sum(d.free_count for d in self._dies.values())

    @property
    def allocations(self) -> Dict[int, List[str]]:
        return self._allocations

    def allocate(self, request_id: int, num_blocks: int, locality_bias: float = 0.7) -> List[str]:
        if num_blocks <= 0:
            return []

        chosen_blocks: List[str] = []
        preferred_dies = self._local_die_ids()
        helper_dies = [die_id for die_id in self._dies if die_id not in preferred_dies]

        for _ in range(num_blocks):
            block = self._try_allocate_from(preferred_dies)
            if not block and helper_dies:
                block = self._try_allocate_from(helper_dies)
            if not block:
                raise RuntimeError("Insufficient KV cache capacity for allocation")
            chosen_blocks.append(block.block_id)

        self._allocations.setdefault(request_id, []).extend(chosen_blocks)
        return chosen_blocks

    def free(self, request_id: int) -> None:
        block_ids = self._allocations.pop(request_id, [])
        for block_id in block_ids:
            die_id = self._block_home.get(block_id)
            if die_id is None:
                continue
            die = self._dies[die_id]
            die.free_blocks.append(block_id)

    def add_helper_dies(self, count: int) -> int:
        added = 0
        for _ in range(count):
            self._add_die(tier="helper")
            added += 1
        if added:
            logger.info("Added %s helper memory dies", added)
        return added

    def remove_helper_dies(self, count: int) -> int:
        removable = [die_id for die_id, die in sorted(self._dies.items(), reverse=True) if die.tier == "helper" and die.free_count == die.capacity_blocks]
        removed = 0
        for die_id in removable:
            if removed == count:
                break
            self._remove_die(die_id)
            removed += 1
        if removed < count:
            logger.warning("Requested removal of %s helper dies but only %s were free", count, removed)
        return removed

    def remote_profile(self, request_ids: Sequence[int], active_compute: int, active_memory: int) -> RemoteAccessProfile:
        block_ids: List[str] = []
        for rid in request_ids:
            block_ids.extend(self._allocations.get(rid, []))
        total_blocks = len(block_ids)
        if total_blocks == 0:
            return RemoteAccessProfile(remote_fraction=0.0, local_fraction=1.0, remote_blocks=0, total_blocks=0)

        local_die_cutoff = min(active_compute, active_memory, len(self._dies))
        local_die_ids = set(sorted(self._dies.keys())[:local_die_cutoff])

        remote_blocks = sum(1 for block_id in block_ids if self._block_home.get(block_id) not in local_die_ids)
        remote_fraction = remote_blocks / total_blocks
        return RemoteAccessProfile(
            remote_fraction=remote_fraction,
            local_fraction=1.0 - remote_fraction,
            remote_blocks=remote_blocks,
            total_blocks=total_blocks,
        )

    def blocks_assigned_to(self, die_id: int) -> int:
        allocated = 0
        for blocks in self._allocations.values():
            allocated += sum(1 for block in blocks if self._block_home.get(block) == die_id)
        die = self._dies.get(die_id)
        free = die.free_count if die else 0
        return allocated + free

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_die(self, tier: str) -> None:
        die_id = len(self._dies)
        blocks = [self._new_block_id(die_id) for _ in range(self._blocks_per_die)]
        die = MemoryDie(die_id=die_id, capacity_blocks=len(blocks), tier=tier, free_blocks=blocks)
        self._dies[die_id] = die

    def _remove_die(self, die_id: int) -> None:
        die = self._dies.pop(die_id, None)
        if not die:
            return
        for block_id in die.free_blocks:
            self._block_home.pop(block_id, None)

    def _new_block_id(self, die_id: int) -> str:
        block_id = f"kv_block_{self._next_block_id}"
        self._next_block_id += 1
        self._block_home[block_id] = die_id
        return block_id

    def _local_die_ids(self) -> List[int]:
        return [die_id for die_id, die in sorted(self._dies.items()) if die.tier == "base"]

    def _try_allocate_from(self, die_ids: Sequence[int]) -> Optional[MemoryBlock]:
        for die_id in die_ids:
            die = self._dies.get(die_id)
            if not die or not die.free_blocks:
                continue
            block_id = die.free_blocks.pop()
            return MemoryBlock(block_id=block_id, die_id=die_id)
        return None
