"""MJCF model builder for the T1 robot.

Generates model variants from the canonical assets/t1/t1_robot.xml.
"""

from __future__ import annotations

from .model_builder_config import BodySubset, GhostType, ModelConfig
from .model_builder_core import T1ModelBuilder, resolve_model_path

__all__ = [
    "BodySubset",
    "GhostType",
    "ModelConfig",
    "T1ModelBuilder",
    "resolve_model_path",
]
