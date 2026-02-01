"""MJCF model builder for the T1 robot.

Generates model variants from the canonical assets/t1/t1_robot.xml.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import copy
import os
import shutil
from typing import Any, Dict, Optional, Sequence
import hashlib
import json
import xml.etree.ElementTree as ET

# Track generated directories for cleanup on exit
_generated_dirs: set[Path] = set()


def _cleanup_generated_dirs() -> None:
    """Remove all generated directories on exit."""
    for path in _generated_dirs:
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
            except Exception:
                pass  # Best-effort cleanup


atexit.register(_cleanup_generated_dirs)


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


class T1ModelBuilder:
    """Build MJCF variants from the canonical T1 robot model."""

    def __init__(
        self,
        base_xml_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        if base_xml_path is None:
            base_xml_path = Path(__file__).resolve().parent / "t1_robot.xml"

        self.base_xml_path = Path(base_xml_path)
        if not self.base_xml_path.exists():
            raise FileNotFoundError(
                f"Base MJCF not found: {self.base_xml_path}"
            )

        if output_dir is None:
            output_dir = self.base_xml_path.parent / "generated"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _generated_dirs.add(self.output_dir)

    def build(self, config: ModelConfig, output_path: Optional[Path] = None) -> Path:
        """Build a variant MJCF file and return its path."""
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        joint_specs = _collect_joint_specs(root)

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MJCF missing <worldbody> element")

        trunk = _find_body(worldbody, "Trunk")
        if trunk is None:
            raise ValueError("MJCF missing Trunk body")

        _apply_fix_base(trunk, config.fix_base)
        worldbody = _apply_body_subset(
            root, worldbody, trunk, config.body_subset
        )
        if config.body_subset in (BodySubset.FULL, BodySubset.UPPER_BODY):
            _apply_ghosts(
                worldbody, config.ghost_type, joint_specs, self.base_xml_path
            )
        _strip_unwanted_nodes(worldbody)
        _apply_physics(root, worldbody, config.enable_physics)
        if not config.enable_collisions:
            _disable_collisions(worldbody)

        if output_path is None:
            output_path = self.output_dir / _build_filename(
                self.base_xml_path, config
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _rewrite_paths(root, self.base_xml_path, output_path)

        ET.indent(tree, space="    ")
        tree.write(output_path, encoding="utf-8", xml_declaration=False)
        return output_path


def resolve_model_path(
    model_path: str,
    model_config: Optional[Any] = None,
) -> str:
    """Resolve a model path, generating a variant if model_config is provided."""
    if model_config is None:
        return model_path

    base_path = Path(model_path)
    if not base_path.exists():
        # Handle stale absolute paths that still include assets/...
        marker = f"assets{os.sep}"
        text = str(model_path)
        if marker in text:
            suffix = text.split(marker, 1)[1]
            try:
                from minerva_bringup.paths import resolve_path as package_resolve_path
            except Exception:  # noqa: BLE001
                package_resolve_path = None
            if package_resolve_path is not None:
                candidate = package_resolve_path(Path("assets") / suffix)
                if candidate.exists():
                    base_path = Path(candidate)

    if isinstance(model_config, ModelConfig):
        config = model_config
    else:
        config = ModelConfig.from_dict(model_config)
    builder = T1ModelBuilder(base_xml_path=base_path)
    return str(builder.build(config))


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


def _build_filename(base_path: Path, config: ModelConfig) -> str:
    body_tag = config.body_subset.value.lower()
    ghost_tag = config.ghost_type.value.lower().replace("_and_", "_")
    phys_tag = "phys" if config.enable_physics else "nophys"
    base_tag = "fixed" if config.fix_base else "floating"
    coll_tag = "coll" if config.enable_collisions else "nocoll"
    payload = json.dumps(config.to_dict(), sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    stem = f"{base_path.stem}_{body_tag}_{ghost_tag}_{phys_tag}_{base_tag}_{coll_tag}_{digest}"
    return f"{stem}.xml"


def _find_body(root: ET.Element, body_name: str) -> Optional[ET.Element]:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _collect_joint_specs(root: ET.Element) -> Dict[str, Dict[str, str]]:
    specs: Dict[str, Dict[str, str]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        if not name:
            continue
        spec: Dict[str, str] = {}
        axis = joint.get("axis")
        if axis:
            spec["axis"] = axis
        joint_range = joint.get("range")
        if joint_range:
            spec["range"] = joint_range
        limited = joint.get("limited")
        if limited:
            spec["limited"] = limited
        if spec:
            specs[name] = spec
    return specs


def _apply_fix_base(trunk: ET.Element, fix_base: bool) -> None:
    freejoints = [child for child in trunk if child.tag == "freejoint"]
    if fix_base:
        for joint in freejoints:
            trunk.remove(joint)
        return

    if freejoints:
        return

    insert_idx = 0
    for idx, child in enumerate(list(trunk)):
        if child.tag in ("inertial", "site"):
            insert_idx = idx + 1
    trunk.insert(insert_idx, ET.Element("freejoint"))


def _reparent_legs_to_trunk(trunk: ET.Element) -> None:
    """Reparent leg bodies from Waist to Trunk.

    This makes legs stay fixed when the Waist joint rotates, which is
    needed for upper body teleoperation where only the torso should yaw.

    The leg body positions are adjusted to account for the Waist body's
    position offset relative to Trunk.
    """
    waist_body = None
    for child in trunk:
        if child.tag == "body" and child.get("name") == "Waist":
            waist_body = child
            break

    if waist_body is None:
        return

    # Get Waist position relative to Trunk
    waist_pos_str = waist_body.get("pos", "0 0 0")
    waist_pos = [float(x) for x in waist_pos_str.split()]

    # Find and move leg bodies from Waist to Trunk
    leg_body_names = {"Hip_Pitch_Left", "Hip_Pitch_Right"}
    legs_to_move = []
    for child in list(waist_body):
        if child.tag == "body" and child.get("name") in leg_body_names:
            legs_to_move.append(child)

    for leg_body in legs_to_move:
        # Remove from Waist
        waist_body.remove(leg_body)

        # Adjust position: add Waist offset to leg position
        leg_pos_str = leg_body.get("pos", "0 0 0")
        leg_pos = [float(x) for x in leg_pos_str.split()]
        new_pos = [
            leg_pos[0] + waist_pos[0],
            leg_pos[1] + waist_pos[1],
            leg_pos[2] + waist_pos[2],
        ]
        leg_body.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")

        # Add to Trunk
        trunk.append(leg_body)


def _swap_trunk_waist_hierarchy(trunk: ET.Element, worldbody: ET.Element) -> ET.Element:
    """Swap Trunk and Waist hierarchy - make Waist the fixed base.

    Original structure:
        Trunk (root, pos=0 0 0.7)
        ├── H1, AL1, AR1 (upper body)
        └── Waist (pos=0.0625 0 -0.1155, has yaw joint)
            └── Legs

    New structure:
        Waist (root, fixed base at pos=0.0625 0 0.5845)
        ├── Trunk (has yaw joint, pos=0 0 0.1155)
        │   └── H1, AL1, AR1 (upper body, positions unchanged relative to Trunk)
        └── Legs (positions adjusted to be relative to Waist)

    Returns:
        The new root body (Waist)
    """
    # Find Waist body
    waist_body = None
    for child in trunk:
        if child.tag == "body" and child.get("name") == "Waist":
            waist_body = child
            break

    if waist_body is None:
        return trunk

    # Get original positions
    trunk_pos_str = trunk.get("pos", "0 0 0.7")
    trunk_pos = [float(x) for x in trunk_pos_str.split()]
    waist_rel_pos_str = waist_body.get("pos", "0.0625 0 -0.1155")
    waist_rel_pos = [float(x) for x in waist_rel_pos_str.split()]

    # Calculate absolute Waist position (Waist is at Trunk pos + relative offset)
    waist_world_pos = [
        trunk_pos[0] + waist_rel_pos[0],
        trunk_pos[1] + waist_rel_pos[1],
        trunk_pos[2] + waist_rel_pos[2],
    ]

    # Calculate Trunk position relative to Waist (inverse of waist relative pos)
    trunk_rel_to_waist = [
        -waist_rel_pos[0],
        -waist_rel_pos[1],
        -waist_rel_pos[2],
    ]

    # Remove Waist from Trunk
    trunk.remove(waist_body)

    # Detect if Trunk has freejoint (for floating base mode) before we remove it
    trunk_freejoint = trunk.find("freejoint")
    has_freejoint = trunk_freejoint is not None
    for fj in trunk.findall("freejoint"):
        trunk.remove(fj)

    # Create new Waist as root (copy inertial and geom, remove old joint)
    new_waist = ET.Element("body", {"name": "Waist", "pos": f"{waist_world_pos[0]} {waist_world_pos[1]} {waist_world_pos[2]}"})

    # Copy inertial from old Waist
    old_inertial = waist_body.find("inertial")
    if old_inertial is not None:
        new_waist.append(copy.deepcopy(old_inertial))

    # Add freejoint to new Waist root if Trunk had one (floating base mode)
    if has_freejoint:
        new_waist.append(ET.Element("freejoint"))

    # Copy geoms from old Waist (but not the joint - we'll move it to Trunk)
    for elem in waist_body:
        if elem.tag == "geom":
            new_waist.append(copy.deepcopy(elem))

    # Get the old yaw joint from Waist (we'll move it to Trunk)
    old_yaw_joint = waist_body.find("joint[@name='Waist']")

    # Set Trunk position relative to Waist
    trunk.set("pos", f"{trunk_rel_to_waist[0]} {trunk_rel_to_waist[1]} {trunk_rel_to_waist[2]}")

    # Add yaw joint to Trunk (now named Trunk_yaw for clarity, or reuse Waist name)
    if old_yaw_joint is not None:
        # Insert joint after inertial
        inertial_idx = 0
        for idx, child in enumerate(list(trunk)):
            if child.tag == "inertial":
                inertial_idx = idx + 1
        # Keep the same joint name for compatibility with control
        new_joint = copy.deepcopy(old_yaw_joint)
        # Set joint position to Waist center in Trunk's frame.
        # This ensures rotation happens about the Waist joint axis, not Trunk origin.
        # Waist center in Trunk frame = -trunk_rel_to_waist = waist_rel_pos
        new_joint.set("pos", f"{waist_rel_pos[0]} {waist_rel_pos[1]} {waist_rel_pos[2]}")
        trunk.insert(inertial_idx, new_joint)

    # Move legs from old Waist to new Waist
    leg_body_names = {"Hip_Pitch_Left", "Hip_Pitch_Right"}
    for child in list(waist_body):
        if child.tag == "body" and child.get("name") in leg_body_names:
            # Position is already relative to old Waist, keep it
            new_waist.append(child)

    # Attach Trunk as child of new Waist
    new_waist.append(trunk)

    # Replace Trunk in worldbody with new Waist
    worldbody.remove(trunk)
    worldbody.append(new_waist)

    return new_waist


def _apply_body_subset(
    root: ET.Element,
    worldbody: ET.Element,
    trunk: ET.Element,
    body_subset: BodySubset,
) -> ET.Element:
    if body_subset == BodySubset.HEAD_ONLY:
        _strip_actuators_and_sensors(root)
        _filter_asset_includes(root, keep_base=True, keep_hands=False)
        new_worldbody = _build_head_worldbody(trunk)
        _replace_worldbody(root, worldbody, new_worldbody)
        return new_worldbody

    if body_subset in (BodySubset.LEFT_HAND_ONLY, BodySubset.RIGHT_HAND_ONLY):
        _strip_actuators_and_sensors(root)
        _filter_asset_includes(root, keep_base=False, keep_hands=True)
        side = "left" if body_subset == BodySubset.LEFT_HAND_ONLY else "right"
        new_worldbody = _build_hand_worldbody(trunk, side)
        _replace_worldbody(root, worldbody, new_worldbody)
        return new_worldbody

    _replace_actuator_include(root, body_subset)
    
    if body_subset == BodySubset.FULL:
        # Swap Trunk/Waist hierarchy: Waist becomes fixed base, Trunk rotates
        new_root = _swap_trunk_waist_hierarchy(trunk, worldbody)
        # Now new_root is Waist. Legs are already children of Waist.
        # We need to find Trunk (now child of Waist) for collision setup
        trunk_body = new_root.find("body[@name='Trunk']")
        if trunk_body is not None:
            _ensure_upper_body_collision_geoms(trunk_body)
        _add_ghost_leg_bodies(worldbody)
        return worldbody

    if body_subset != BodySubset.UPPER_BODY:
        return worldbody

    # For UPPER_BODY (IK model): swap hierarchy so Trunk pose can be tracked
    new_root = _swap_trunk_waist_hierarchy(trunk, worldbody)
    
    # Remove legs from new Waist root (not needed for upper body IK)
    leg_body_names = {"Hip_Pitch_Left", "Hip_Pitch_Right"}
    for child in list(new_root):
        if child.tag == "body" and child.get("name") in leg_body_names:
            new_root.remove(child)

    trunk_body = new_root.find("body[@name='Trunk']")
    if trunk_body is not None:
        _ensure_upper_body_collision_geoms(trunk_body)
    _add_ghost_leg_bodies(worldbody)
    return worldbody


def _strip_actuators_and_sensors(root: ET.Element) -> None:
    remove_files = {
        "include/t1_actuators_full.xml",
        "include/t1_actuators_upper.xml",
        "include/t1_sensors_common.xml",
    }
    for child in list(root):
        if child.tag == "include" and child.get("file") in remove_files:
            root.remove(child)
        if child.tag in ("actuator", "sensor"):
            root.remove(child)


def _filter_asset_includes(
    root: ET.Element, keep_base: bool, keep_hands: bool
) -> None:
    for child in list(root):
        if child.tag != "include":
            continue
        file_attr = child.get("file")
        if file_attr == "include/t1_assets_base.xml" and not keep_base:
            root.remove(child)
        if file_attr == "include/t1_assets_hands.xml" and not keep_hands:
            root.remove(child)


def _replace_worldbody(
    root: ET.Element, old_worldbody: ET.Element, new_worldbody: ET.Element
) -> None:
    children = list(root)
    try:
        idx = children.index(old_worldbody)
    except ValueError:
        root.append(new_worldbody)
        return
    root.remove(old_worldbody)
    root.insert(idx, new_worldbody)


def _build_head_worldbody(trunk: ET.Element) -> ET.Element:
    head = _find_body(trunk, "H1")
    if head is None:
        raise ValueError("HEAD_ONLY requires body H1 in the base model")

    head_copy = copy.deepcopy(head)
    head_copy.set("pos", "0 0 0")

    base = ET.Element("body", {"name": "head_base", "pos": "0 0 0"})
    base.append(head_copy)

    worldbody = ET.Element("worldbody")
    worldbody.append(base)
    return worldbody


def _build_hand_worldbody(trunk: ET.Element, side: str) -> ET.Element:
    body_name = f"{side}_inspire_hand"
    hand = _find_body(trunk, body_name)
    if hand is None:
        raise ValueError(
            f"{body_name} not found in base model for {side} hand"
        )

    hand_copy = copy.deepcopy(hand)
    hand_copy.set("pos", "0 0 0")

    worldbody = ET.Element("worldbody")
    worldbody.append(hand_copy)
    return worldbody


def _replace_actuator_include(root: ET.Element, body_subset: BodySubset) -> None:
    actuator_files = {"include/t1_actuators_full.xml", "include/t1_actuators_upper.xml"}
    includes = [child for child in list(root) if child.tag == "include"]
    for child in includes:
        if child.get("file") in actuator_files:
            root.remove(child)

    actuator_file = "include/t1_actuators_upper.xml"
    if body_subset == BodySubset.FULL:
        actuator_file = "include/t1_actuators_full.xml"

    new_include = ET.Element("include", {"file": actuator_file})
    insert_idx = None
    for idx, child in enumerate(list(root)):
        if child.tag == "include" and child.get("file") == "include/t1_sensors_common.xml":
            insert_idx = idx
            break
    if insert_idx is None:
        root.append(new_include)
    else:
        root.insert(insert_idx, new_include)


def _apply_ghosts(
    worldbody: ET.Element,
    ghost_type: GhostType,
    joint_specs: Optional[Dict[str, Dict[str, str]]] = None,
    base_xml_path: Optional[Path] = None,
) -> None:
    ghost_includes = {
        "include/target_ghosts.xml",
    }
    for child in list(worldbody):
        if child.tag == "include" and child.get("file") in ghost_includes:
            worldbody.remove(child)

    # Remove any previously generated ghost bodies
    for child in list(worldbody):
        if child.tag != "body":
            continue
        if child.get("name") in ("Trunk_des_ghost", "Trunk_ik_ghost"):
            worldbody.remove(child)

    hand_templates: Dict[str, ET.Element] = {}
    for side in ("left", "right"):
        hand_body = _find_body(worldbody, f"{side}_inspire_hand")
        if hand_body is not None:
            hand_templates[side] = copy.deepcopy(hand_body)

    if ghost_type in (GhostType.TARGETS, GhostType.TARGETS_AND_IK):
        worldbody.append(
            ET.Element("include", {"file": "include/target_ghosts.xml"})
        )
        _add_target_ghost_variants(worldbody, base_xml_path, ("raw", "kf"))
    if ghost_type in (GhostType.IK_ROBOT, GhostType.TARGETS_AND_IK):
        _add_ik_ghost_robot(worldbody, joint_specs, hand_templates)
        # Generate desired ghost robot programmatically (blue, transparent)
        _add_desired_ghost_robot(worldbody, joint_specs, hand_templates)
        # Generate posture ghost robot programmatically (purple, transparent)
        _add_posture_ghost_robot(worldbody, joint_specs, hand_templates)
        # Generate warmstart ghost robot programmatically (orange, transparent)
        _add_warmstart_ghost_robot(worldbody, joint_specs, hand_templates)


def _clone_body_with_suffix(body: ET.Element, suffix: str) -> ET.Element:
    """Clone a ghost body and rename all _ghost nodes with a suffix."""
    cloned = copy.deepcopy(body)
    for elem in cloned.iter():
        name = elem.get("name")
        if name and "_ghost" in name:
            elem.set("name", name.replace("_ghost", f"_{suffix}_ghost"))
    return cloned


def _load_body_from_include(
    xml_path: Path, body_name: str
) -> Optional[ET.Element]:
    """Load a named body from an include XML file."""
    if not xml_path.exists():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _add_target_ghost_variants(
    worldbody: ET.Element,
    base_xml_path: Optional[Path],
    suffixes: Sequence[str],
) -> None:
    """Clone target ghost bodies to create raw/KF variants without new XMLs."""
    if base_xml_path is None:
        return

    base_dir = base_xml_path.parent
    target_ghosts = base_dir / "include" / "target_ghosts.xml"
    hand_dir = (target_ghosts.parent / ".." / ".." / "hands").resolve()
    head_body = _load_body_from_include(target_ghosts, "H2_ghost")

    hand_files = [
        (hand_dir / "ghost_hand_left.xml", "left_inspire_hand_ghost"),
        (hand_dir / "ghost_hand_right.xml", "right_inspire_hand_ghost"),
    ]

    for suffix in suffixes:
        # Head and hands are authored in separate include files.
        if head_body is not None:
            worldbody.append(_clone_body_with_suffix(head_body, suffix))

        for hand_path, root_name in hand_files:
            hand_body = _load_body_from_include(hand_path, root_name)
            if hand_body is None:
                continue
            worldbody.append(_clone_body_with_suffix(hand_body, suffix))


def _apply_physics(root: ET.Element, worldbody: ET.Element, enable: bool) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        insert_idx = 0
        for idx, child in enumerate(list(root)):
            if child.tag in ("compiler", "size"):
                insert_idx = idx + 1
        root.insert(insert_idx, option)

    option.set("integrator", "Euler")
    option.set("gravity", "0 0 -9.81" if enable else "0 0 0")

    ground = _find_geom(worldbody, "ground")
    if ground is not None:
        if enable:
            ground.set("contype", "1")
            ground.set("conaffinity", "1")
        else:
            ground.set("contype", "0")
            ground.set("conaffinity", "0")


def _disable_collisions(worldbody: ET.Element) -> None:
    for geom in worldbody.iter("geom"):
        if geom.get("name") == "ground":
            continue
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def _strip_unwanted_nodes(worldbody: ET.Element) -> None:
    """Remove extra ghost collision meshes from generated variants."""
    body_names = {
        "ghost_left_thigh",
        "ghost_left_shank",
        "ghost_right_thigh",
        "ghost_right_shank",
    }
    geom_names = {
        "left_hand_col",
        "right_hand_col",
    }
    _remove_named_children(worldbody, "body", body_names)
    _remove_named_children(worldbody, "geom", geom_names)


def _remove_named_children(
    parent: ET.Element, tag: str, names: set[str]
) -> None:
    for child in list(parent):
        if child.tag == tag and child.get("name") in names:
            parent.remove(child)
            continue
        _remove_named_children(child, tag, names)


def _find_geom(worldbody: ET.Element, name: str) -> Optional[ET.Element]:
    for geom in worldbody.iter("geom"):
        if geom.get("name") == name:
            return geom
    return None


def _rewrite_paths(
    root: ET.Element, base_xml_path: Path, output_path: Path
) -> None:
    base_dir = base_xml_path.parent.resolve()
    output_dir = output_path.parent.resolve()
    if base_dir == output_dir:
        return

    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir and not Path(meshdir).is_absolute():
            mesh_src = (base_dir / meshdir).resolve()
            if mesh_src.exists():
                rel_meshdir = os.path.relpath(mesh_src, output_dir)
                if not rel_meshdir.endswith("/"):
                    rel_meshdir += "/"
                compiler.set("meshdir", rel_meshdir)

    for include in root.iter("include"):
        file_attr = include.get("file")
        if not file_attr:
            continue
        if Path(file_attr).is_absolute():
            continue
        src_path = (base_dir / file_attr).resolve()
        if not src_path.exists():
            continue
        rel_path = os.path.relpath(src_path, output_dir)
        include.set("file", Path(rel_path).as_posix())


def _ensure_upper_body_collision_geoms(trunk: ET.Element) -> None:
    specs = [
        {
            "name": "trunk_col",
            "parent": "Trunk",
            "type": "box",
            "size": "0.075 0.1 0.15",
            "pos": "0.06 0 0.12",
        },
        {
            "name": "head_col",
            "parent": "H2",
            "type": "sphere",
            "size": "0.08",
            "pos": "0.01 0 0.11",
        },
        {
            "name": "left_shoulder_col",
            "parent": "AL2",
            "type": "sphere",
            "size": "0.05",
            "pos": "0 0.03 0",
        },
        {
            "name": "left_upper_arm_col",
            "parent": "AL3",
            "type": "cylinder",
            "size": "0.03 0.08",
            "pos": "0 0.10 0",  # Shifted toward elbow, away from shoulder
            "quat": "0.707105 0.707108 0 0",
        },
        {
            "name": "left_forearm_col",
            "parent": "AL4",
            "type": "cylinder",
            "size": "0.04 0.08",
            "pos": "0 0.04 0",  # Extended toward elbow to cover joint
            "quat": "0.707105 0.707108 0 0",
        },
        {
            "name": "left_hand_col",
            "parent": "left_inspire_hand",
            "type": "capsule",
            "size": "0.04 0.06",
            "pos": "0 0.06 0",
            "quat": "0.707105 0.707108 0 0",
        },
        {
            "name": "right_shoulder_col",
            "parent": "AR2",
            "type": "sphere",
            "size": "0.05",
            "pos": "0 -0.03 0",
        },
        {
            "name": "right_upper_arm_col",
            "parent": "AR3",
            "type": "cylinder",
            "size": "0.03 0.08",
            "pos": "0 -0.10 0",  # Shifted toward elbow, away from shoulder
            "quat": "0.707105 0.707108 0 0",
        },
        {
            "name": "right_forearm_col",
            "parent": "AR4",
            "type": "cylinder",
            "size": "0.04 0.08",
            "pos": "0 -0.04 0",  # Extended toward elbow to cover joint
            "quat": "0.707105 0.707108 0 0",
        },
        {
            "name": "right_hand_col",
            "parent": "right_inspire_hand",
            "type": "capsule",
            "size": "0.04 0.06",
            "pos": "0 -0.06 0",
            "quat": "0.707105 0.707108 0 0",
        },
    ]

    for spec in specs:
        parent = trunk if spec["parent"] == "Trunk" else _find_body(trunk, spec["parent"])
        if parent is None:
            continue
        if any(
            child.tag == "geom" and child.get("name") == spec["name"]
            for child in parent
        ):
            continue

        geom = ET.Element("geom", {
            "name": spec["name"],
            "type": spec["type"],
            "size": spec["size"],
            "pos": spec["pos"],
            "rgba": "0 0 0 0",
        })
        if "quat" in spec:
            geom.set("quat", spec["quat"])
        parent.append(geom)


def _add_ghost_leg_bodies(worldbody: ET.Element) -> None:
    legs = [
        {
            "body": "ghost_left_thigh",
            "pos": "0.04 0.106 0.45",
            "geom": "ghost_left_thigh_col",
            "size": "0.055 0.10",
        },
        {
            "body": "ghost_left_shank",
            "pos": "0.035 0.106 0.21",
            "geom": "ghost_left_shank_col",
            "size": "0.045 0.12",
        },
        {
            "body": "ghost_right_thigh",
            "pos": "0.04 -0.106 0.45",
            "geom": "ghost_right_thigh_col",
            "size": "0.055 0.10",
        },
        {
            "body": "ghost_right_shank",
            "pos": "0.035 -0.106 0.21",
            "geom": "ghost_right_shank_col",
            "size": "0.045 0.12",
        },
    ]

    existing = {body.get("name") for body in worldbody.iter("body")}
    for spec in legs:
        if spec["body"] in existing:
            continue
        body = ET.Element(
            "body",
            {"name": spec["body"], "mocap": "true", "pos": spec["pos"]},
        )
        geom = ET.Element(
            "geom",
            {
                "name": spec["geom"],
                "type": "capsule",
                "size": spec["size"],
                "rgba": "0 0 0 0",
                "contype": "0",
                "conaffinity": "0",
            },
        )
        body.append(geom)
        worldbody.append(body)


def _add_ik_ghost_robot(
    worldbody: ET.Element,
    joint_specs: Dict[str, Dict[str, str]],
    hand_templates: Optional[Dict[str, ET.Element]] = None,
) -> None:
    """Add an articulated ghost robot for visualizing IK solution (green)."""
    GREEN_RGBA = "1.0 0.8 0.2 0.4"
    upper_body = _build_articulated_ghost(
        base_name="Trunk_ik_ghost",
        base_pos="0 0 0.7",
        freejoint_name="Trunk_ik_ghost_free",
        suffix="_ik_ghost",
        rgba=GREEN_RGBA,
        joint_specs=joint_specs,
        hand_templates=hand_templates,
    )
    worldbody.append(upper_body)


def _add_desired_ghost_robot(
    worldbody: ET.Element,
    joint_specs: Dict[str, Dict[str, str]],
    hand_templates: Optional[Dict[str, ET.Element]] = None,
) -> None:
    """Add an articulated ghost robot for visualizing commanded state (blue).
    
    This clones the IK ghost structure but with blue transparent styling.
    Joint names use _des_ghost suffix instead of _ik_ghost.
    """
    # Blue transparent color for desired state
    BLUE_RGBA = "0.3 0.5 1.0 0.5"
    
    # Upper body structure (mirrors the IK ghost layout)
    upper_body = _build_articulated_ghost(
        base_name="Trunk_des_ghost",
        base_pos="0 0 0.7",
        freejoint_name="Trunk_des_ghost_free",
        suffix="_des_ghost",
        rgba=BLUE_RGBA,
        joint_specs=joint_specs,
        hand_templates=hand_templates,
    )
    worldbody.append(upper_body)


def _add_posture_ghost_robot(
    worldbody: ET.Element,
    joint_specs: Dict[str, Dict[str, str]],
    hand_templates: Optional[Dict[str, ET.Element]] = None,
) -> None:
    """Add an articulated ghost robot for visualizing posture reference (purple).
    
    Shows the default posture task target pose.
    Joint names use _posture_ghost suffix.
    """
    # Purple transparent color for posture reference
    PURPLE_RGBA = "0.7 0.3 1.0 0.4"
    
    upper_body = _build_articulated_ghost(
        base_name="Trunk_posture_ghost",
        base_pos="0 0 0.7",
        freejoint_name="Trunk_posture_ghost_free",
        suffix="_posture_ghost",
        rgba=PURPLE_RGBA,
        joint_specs=joint_specs,
        hand_templates=hand_templates,
    )
    worldbody.append(upper_body)


def _add_warmstart_ghost_robot(
    worldbody: ET.Element,
    joint_specs: Dict[str, Dict[str, str]],
    hand_templates: Optional[Dict[str, ET.Element]] = None,
) -> None:
    """Add an articulated ghost robot for visualizing warmstart result (orange).
    
    Shows the IK result after Stage 1 (before collision refinement).
    Joint names use _warmstart_ghost suffix.
    """
    # Orange transparent color for warmstart result
    ORANGE_RGBA = "1.0 0.6 0.2 0.4"
    
    upper_body = _build_articulated_ghost(
        base_name="Trunk_warmstart_ghost",
        base_pos="0 0 0.7",
        freejoint_name="Trunk_warmstart_ghost_free",
        suffix="_warmstart_ghost",
        rgba=ORANGE_RGBA,
        joint_specs=joint_specs,
        hand_templates=hand_templates,
    )
    worldbody.append(upper_body)


def _build_articulated_ghost(
    base_name: str,
    base_pos: str,
    freejoint_name: str,
    suffix: str,
    rgba: str,
    joint_specs: Dict[str, Dict[str, str]],
    hand_templates: Optional[Dict[str, ET.Element]] = None,
) -> ET.Element:
    """Build an articulated ghost robot tree.
    
    This creates a complete upper body kinematic chain with the specified
    suffix and color for visualization purposes.
    """
    def make_body(name: str, pos: str, quat: Optional[str] = None) -> ET.Element:
        attrib = {"name": name + suffix, "pos": pos}
        if quat:
            attrib["quat"] = quat
        body = ET.Element("body", attrib)
        body.append(ET.Element("inertial", {
            "pos": "0 0 0", "mass": "1e-4", "diaginertia": "1e-6 1e-6 1e-6"
        }))
        return body
    
    def add_joint(body: ET.Element, name: str, pos: str = "0 0 0") -> None:
        if joint_specs is None:
            raise ValueError("joint_specs is required for ghost generation")
        if name not in joint_specs:
            raise ValueError(f"Joint '{name}' not found in XML joint specs")
        spec = joint_specs[name]
        attrib = {
            "name": name + suffix,
            "pos": pos,
            "axis": spec.get("axis", "0 0 1"),
            "limited": spec.get("limited", "true"),
        }
        if "range" in spec:
            attrib["range"] = spec["range"]
        body.append(ET.Element("joint", attrib))
    
    def add_mesh_geom(body: ET.Element, mesh: str, quat: Optional[str] = None) -> None:
        attrib = {
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "rgba": rgba,
            "mesh": mesh,
        }
        if quat:
            attrib["quat"] = quat
        body.append(ET.Element("geom", attrib))

    def clone_hand_template(side: str) -> Optional[ET.Element]:
        if not hand_templates:
            return None
        template = hand_templates.get(side)
        if template is None:
            return None
        cloned = copy.deepcopy(template)
        for elem in cloned.iter():
            name = elem.get("name")
            if name:
                elem.set("name", f"{name}{suffix}")
            if elem.tag == "geom":
                # Only color visual geoms (group 1), keep collision primitives invisible
                if elem.get("group") == "1":
                    elem.set("rgba", rgba)
                else:
                    elem.set("rgba", "0 0 0 0")
        return cloned
    
    # Build the kinematic tree - NEW STRUCTURE:
    # Waist (root, fixed base) -> Trunk (yaw joint) -> Upper body
    # Legs are children of Waist (stay fixed when Trunk rotates)
    
    # base_pos is the original Trunk position (e.g. "0 0 0.7")
    # Calculate Waist position: Trunk pos + Waist offset (0.0625 0 -0.1155)
    trunk_pos = [float(x) for x in base_pos.split()]
    waist_offset = [0.0625, 0, -0.1155]
    waist_world_pos = [
        trunk_pos[0] + waist_offset[0],
        trunk_pos[1] + waist_offset[1],
        trunk_pos[2] + waist_offset[2],
    ]
    waist_pos_str = f"{waist_world_pos[0]} {waist_world_pos[1]} {waist_world_pos[2]}"
    
    # Waist as root body (fixed base)
    waist = make_body("Waist", waist_pos_str)
    waist.insert(1, ET.Element("freejoint", {"name": freejoint_name}))
    add_mesh_geom(waist, "Waist")
    
    # Trunk as child of Waist with yaw joint
    # Position relative to Waist: inverse of waist offset
    trunk_rel_pos = f"{-waist_offset[0]} {-waist_offset[1]} {-waist_offset[2]}"
    trunk = make_body("Trunk", trunk_rel_pos)
    # Waist joint at Waist center in Trunk's frame for correct rotation pivot
    waist_joint_pos = f"{waist_offset[0]} {waist_offset[1]} {waist_offset[2]}"
    add_joint(trunk, "Waist", pos=waist_joint_pos)

    add_mesh_geom(trunk, "Trunk")
    
    # Head chain - child of Trunk (original positions relative to Trunk)
    h1 = make_body("H1", "0.0625 0 0.243")
    add_joint(h1, "AAHead_yaw")
    add_mesh_geom(h1, "H1")
    
    h2 = make_body("H2", "0 0 0.06185")
    add_joint(h2, "Head_pitch")
    add_mesh_geom(h2, "H2")
    h1.append(h2)
    trunk.append(h1)
    
    # Left arm chain - child of Trunk
    al1 = make_body("AL1", "0.0575 0.1063 0.219", "1 0 0.000440565 0")
    add_joint(al1, "Left_Shoulder_Pitch")
    add_mesh_geom(al1, "AL1")
    
    al2 = make_body("AL2", "0 0.047 0")
    add_joint(al2, "Left_Shoulder_Roll")
    add_mesh_geom(al2, "AL2")
    
    al3 = make_body("AL3", "0.00025 0.0605 0")
    add_joint(al3, "Left_Elbow_Pitch")
    add_mesh_geom(al3, "AL3")
    
    al4 = make_body("AL4", "0 0.1471 0")
    add_joint(al4, "Left_Elbow_Yaw")
    add_mesh_geom(al4, "AL4")
    
    al5 = make_body("AL5", "0 0.105 0.00025")
    add_joint(al5, "Left_Wrist_Pitch")
    add_mesh_geom(al5, "AL5")
    
    al6 = make_body("AL6", "0 0.042 0")
    add_joint(al6, "Left_Wrist_Yaw")
    add_mesh_geom(al6, "AL6")
    
    left_hand = clone_hand_template("left")
    if left_hand is None:
        left_hand = make_body("left_inspire_hand", "-0.00001093 0.10132774 -0.00093692")
        add_joint(left_hand, "Left_Hand_Roll")
        add_mesh_geom(left_hand, "left_hand_base_link", "0.707107 -0.707107 0 0")
    
    al6.append(left_hand)
    al5.append(al6)
    al4.append(al5)
    al3.append(al4)
    al2.append(al3)
    al1.append(al2)
    trunk.append(al1)
    
    # Right arm chain - child of Trunk
    ar1 = make_body("AR1", "0.0575 -0.1063 0.219", "1 0 0.000440565 0")
    add_joint(ar1, "Right_Shoulder_Pitch")
    add_mesh_geom(ar1, "AR1")
    
    ar2 = make_body("AR2", "0 -0.047 0")
    add_joint(ar2, "Right_Shoulder_Roll")
    add_mesh_geom(ar2, "AR2")
    
    ar3 = make_body("AR3", "0.00025 -0.0605 0")
    add_joint(ar3, "Right_Elbow_Pitch")
    add_mesh_geom(ar3, "AR3")
    
    ar4 = make_body("AR4", "0 -0.1471 0")
    add_joint(ar4, "Right_Elbow_Yaw")
    add_mesh_geom(ar4, "AR4")
    
    ar5 = make_body("AR5", "0 -0.105 0.00025")
    add_joint(ar5, "Right_Wrist_Pitch")
    add_mesh_geom(ar5, "AR5")
    
    ar6 = make_body("AR6", "0 -0.042 0")
    add_joint(ar6, "Right_Wrist_Yaw")
    add_mesh_geom(ar6, "AR6")
    
    right_hand = clone_hand_template("right")
    if right_hand is None:
        right_hand = make_body("right_inspire_hand", "0.00001093 -0.10132774 -0.00093692")
        add_joint(right_hand, "Right_Hand_Roll")
        add_mesh_geom(right_hand, "right_hand_base_link", "0.707107 0.707107 0 0")
    
    ar6.append(right_hand)
    ar5.append(ar6)
    ar4.append(ar5)
    ar3.append(ar4)
    ar2.append(ar3)
    ar1.append(ar2)
    trunk.append(ar1)
    
    # Attach Trunk (with upper body) to Waist
    waist.append(trunk)
    
    # Left leg chain - child of Waist (stays fixed when Trunk rotates)
    hip_pitch_l = make_body("Hip_Pitch_Left", "0 0.106 0")
    add_joint(hip_pitch_l, "Left_Hip_Pitch")
    add_mesh_geom(hip_pitch_l, "Hip_Pitch_Left")
    
    hip_roll_l = make_body("Hip_Roll_Left", "0 0 -0.02")
    add_joint(hip_roll_l, "Left_Hip_Roll")
    add_mesh_geom(hip_roll_l, "Hip_Roll_Left")
    
    hip_yaw_l = make_body("Hip_Yaw_Left", "0 0 -0.081854")
    add_joint(hip_yaw_l, "Left_Hip_Yaw")
    add_mesh_geom(hip_yaw_l, "Hip_Yaw_Left")
    
    shank_l = make_body("Shank_Left", "-0.014 0 -0.134")
    add_joint(shank_l, "Left_Knee_Pitch")
    add_mesh_geom(shank_l, "Shank_Left")
    
    ankle_cross_l = make_body("Ankle_Cross_Left", "0 0 -0.28")
    add_joint(ankle_cross_l, "Left_Ankle_Pitch")
    add_mesh_geom(ankle_cross_l, "Ankle_Cross_Left")
    
    left_foot = make_body("left_foot_link", "0 0.00025 -0.012")
    add_joint(left_foot, "Left_Ankle_Roll")
    add_mesh_geom(left_foot, "left_foot_link")
    
    ankle_cross_l.append(left_foot)
    shank_l.append(ankle_cross_l)
    hip_yaw_l.append(shank_l)
    hip_roll_l.append(hip_yaw_l)
    hip_pitch_l.append(hip_roll_l)
    waist.append(hip_pitch_l)
    
    # Right leg chain - child of Waist (stays fixed when Trunk rotates)
    hip_pitch_r = make_body("Hip_Pitch_Right", "0 -0.106 0")
    add_joint(hip_pitch_r, "Right_Hip_Pitch")
    add_mesh_geom(hip_pitch_r, "Hip_Pitch_Right")
    
    hip_roll_r = make_body("Hip_Roll_Right", "0 0 -0.02")
    add_joint(hip_roll_r, "Right_Hip_Roll")
    add_mesh_geom(hip_roll_r, "Hip_Roll_Right")
    
    hip_yaw_r = make_body("Hip_Yaw_Right", "0 0 -0.081854")
    add_joint(hip_yaw_r, "Right_Hip_Yaw")
    add_mesh_geom(hip_yaw_r, "Hip_Yaw_Right")
    
    shank_r = make_body("Shank_Right", "-0.014 0 -0.134")
    add_joint(shank_r, "Right_Knee_Pitch")
    add_mesh_geom(shank_r, "Shank_Right")
    
    ankle_cross_r = make_body("Ankle_Cross_Right", "0 0 -0.28")
    add_joint(ankle_cross_r, "Right_Ankle_Pitch")
    add_mesh_geom(ankle_cross_r, "Ankle_Cross_Right")
    
    right_foot = make_body("right_foot_link", "0 -0.00025 -0.012")
    add_joint(right_foot, "Right_Ankle_Roll")
    add_mesh_geom(right_foot, "right_foot_link")
    
    ankle_cross_r.append(right_foot)
    shank_r.append(ankle_cross_r)
    hip_yaw_r.append(shank_r)
    hip_roll_r.append(hip_yaw_r)
    hip_pitch_r.append(hip_roll_r)
    waist.append(hip_pitch_r)
    
    return waist


__all__ = [
    "BodySubset",
    "GhostType",
    "ModelConfig",
    "T1ModelBuilder",
    "resolve_model_path",
]
