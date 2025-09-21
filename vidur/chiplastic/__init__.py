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
from vidur.chiplastic.runtime import ChiplasticRuntime, ScalingState

__all__ = [
    "ChiplasticHardwareConfig",
    "ChiplasticScalingThresholds",
    "ChiplasticTuningConfig",
    "ChiplasticRuntime",
    "ScalingState",
]
