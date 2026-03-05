"""Configuration helpers for the T1 MJCF model builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, Sequence


class BodySubset(str, Enum):
    """Subset selection for the main robot body."""

    FULL = "FULL"
    UPPER_BODY = "UPPER_BODY"
    HEAD_ONLY = "HEAD_ONLY"
    LEFT_HAND_ONLY = "LEFT_HAND_ONLY"
    RIGHT_HAND_ONLY = "RIGHT_HAND_ONLY"


class GhostType(str, Enum):
    """Ghost visualization variants."""

    NONE = "NONE"
    TARGETS = "TARGETS"
    IK_ROBOT = "IK_ROBOT"
    TARGETS_AND_IK = "TARGETS_AND_IK"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for building T1 MJCF variants."""

    body_subset: BodySubset = BodySubset.FULL
    ghost_type: GhostType = GhostType.NONE
    enable_physics: bool = True
    fix_base: bool = False
    enable_collisions: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ModelConfig":
        """Build a ModelConfig from a YAML-friendly dictionary."""
        if data is None:
            return cls()

        body_subset = _coerce_body_subset(data.get("body_subset"))
        ghost_type = _coerce_ghost_type(data.get("ghost_type"))
        enable_physics = bool(data.get("enable_physics", True))
        fix_base = bool(data.get("fix_base", False))

        enable_collisions = bool(data.get("enable_collisions", False))
        if "disable_collisions" in data:
            enable_collisions = not bool(data.get("disable_collisions"))

        return cls(
            body_subset=body_subset,
            ghost_type=ghost_type,
            enable_physics=enable_physics,
            fix_base=fix_base,
            enable_collisions=enable_collisions,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-friendly dict."""
        return {
            "body_subset": self.body_subset.value,
            "ghost_type": self.ghost_type.value,
            "enable_physics": self.enable_physics,
            "fix_base": self.fix_base,
            "enable_collisions": self.enable_collisions,
        }

    def with_ghosts(self, ghost_type: GhostType) -> "ModelConfig":
        """Return a copy with a new ghost type."""
        return replace(self, ghost_type=ghost_type)

    def with_body_subset(self, body_subset: BodySubset) -> "ModelConfig":
        """Return a copy with a new body subset."""
        return replace(self, body_subset=body_subset)


def _coerce_body_subset(value: Any) -> BodySubset:
    if isinstance(value, BodySubset):
        return value
    if value is None:
        return BodySubset.FULL
    text = str(value).strip().upper()
    aliases = {
        "UPPER": "UPPER_BODY",
        "UPPERBODY": "UPPER_BODY",
        "FULL_BODY": "FULL",
        "HEAD": "HEAD_ONLY",
        "HAND_LEFT": "LEFT_HAND_ONLY",
        "LEFT_HAND": "LEFT_HAND_ONLY",
        "HAND_RIGHT": "RIGHT_HAND_ONLY",
        "RIGHT_HAND": "RIGHT_HAND_ONLY",
    }
    text = aliases.get(text, text)
    return BodySubset(text)


def _coerce_ghost_type(value: Any) -> GhostType:
    if isinstance(value, GhostType):
        return value
    if value is None:
        return GhostType.NONE
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        types = {_coerce_ghost_type(item) for item in value}
        if GhostType.TARGETS in types and GhostType.IK_ROBOT in types:
            return GhostType.TARGETS_AND_IK
        if GhostType.TARGETS in types:
            return GhostType.TARGETS
        if GhostType.IK_ROBOT in types:
            return GhostType.IK_ROBOT
        return GhostType.NONE
    text = str(value).strip().upper()
    aliases = {
        "IK": "IK_ROBOT",
        "TARGETS+IK": "TARGETS_AND_IK",
        "TARGETS_AND_IK_ROBOT": "TARGETS_AND_IK",
    }
    text = aliases.get(text, text)
    return GhostType(text)
