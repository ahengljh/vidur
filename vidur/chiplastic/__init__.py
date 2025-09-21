"""Chiplastic runtime support for Vidur.

This package contains the core control/measurement primitives
needed to simulate elastic chiplet-aware execution described in
Chiplastic (HPCA'26 submission).
"""

from vidur.chiplastic.config import (
    ChiplasticHardwareConfig,
    ChiplasticScalingThresholds,
    ChiplasticTuningConfig,
)
from vidur.chiplastic.memory import ChipletMemoryManager, RemoteAccessProfile
from vidur.chiplastic.runtime import ChiplasticRuntime, ScalingState
from vidur.chiplastic.interconnect import InterconnectModel

__all__ = [
    "ChiplasticHardwareConfig",
    "ChiplasticScalingThresholds",
    "ChiplasticTuningConfig",
    "ChiplasticRuntime",
    "ScalingState",
    "ChipletMemoryManager",
    "RemoteAccessProfile",
    "InterconnectModel",
]
