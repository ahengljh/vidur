"""Core configuration helpers for the Chiplastic extensions."""

from vidur.chiplastic.config import (
    ChiplasticHardwareConfig,
    ChiplasticScalingThresholds,
    ChiplasticTuningConfig,
)
from vidur.chiplastic.memory import ChipletMemoryManager, RemoteAccessProfile
from vidur.chiplastic.interconnect import InterconnectModel

__all__ = [
    "ChiplasticHardwareConfig",
    "ChiplasticScalingThresholds",
    "ChiplasticTuningConfig",
    "ChipletMemoryManager",
    "RemoteAccessProfile",
    "InterconnectModel",
]
