"""Worldbody and collision helpers for the T1 MJCF model builder."""

from __future__ import annotations

import copy
from typing import Optional
import xml.etree.ElementTree as ET

from .model_builder_config import BodySubset


def _find_body(root: ET.Element, body_name: str) -> Optional[ET.Element]:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _find_geom(worldbody: ET.Element, name: str) -> Optional[ET.Element]:
    for geom in worldbody.iter("geom"):
        if geom.get("name") == name:
            return geom
    return None


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

    # Remove finger child bodies from both inspire_hand bodies.
    # Keeps the inspire_hand body (holds hand_col) but strips all
    # finger kinematic chains (24 joints total).
    _strip_finger_bodies(new_root)

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
    actuator_files = {
        "include/t1_actuators_full.xml",
        "include/t1_actuators_upper.xml",
        "include/t1_actuators_ik.xml",
    }
    includes = [child for child in list(root) if child.tag == "include"]
    for child in includes:
        if child.get("file") in actuator_files:
            root.remove(child)

    if body_subset == BodySubset.FULL:
        actuator_file = "include/t1_actuators_full.xml"
    else:
        actuator_file = "include/t1_actuators_upper.xml"

    new_include = ET.Element("include", {"file": actuator_file})
    insert_idx = None
    for idx, child in enumerate(list(root)):
        if (
            child.tag == "include"
            and child.get("file") == "include/t1_sensors_common.xml"
        ):
            insert_idx = idx
            break
    if insert_idx is None:
        root.append(new_include)
    else:
        root.insert(insert_idx, new_include)


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


def _swap_trunk_waist_hierarchy(
    trunk: ET.Element, worldbody: ET.Element
) -> ET.Element:
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
    new_waist = ET.Element(
        "body",
        {
            "name": "Waist",
            "pos": (
                f"{waist_world_pos[0]} {waist_world_pos[1]} {waist_world_pos[2]}"
            ),
        },
    )

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
    trunk.set(
        "pos",
        f"{trunk_rel_to_waist[0]} {trunk_rel_to_waist[1]} {trunk_rel_to_waist[2]}",
    )

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
        new_joint.set(
            "pos",
            f"{waist_rel_pos[0]} {waist_rel_pos[1]} {waist_rel_pos[2]}",
        )
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


def _strip_unwanted_nodes(worldbody: ET.Element) -> None:
    """Remove extra ghost collision meshes from generated variants."""
    body_names = {
        "ghost_left_thigh",
        "ghost_left_shank",
        "ghost_right_thigh",
        "ghost_right_shank",
    }
    _remove_named_children(worldbody, "body", body_names)


def _remove_named_children(
    parent: ET.Element, tag: str, names: set[str]
) -> None:
    for child in list(parent):
        if child.tag == tag and child.get("name") in names:
            parent.remove(child)
            continue
        _remove_named_children(child, tag, names)


def _strip_finger_bodies(root: ET.Element) -> None:
    """Remove finger child bodies from inspire_hand bodies.

    Keeps the inspire_hand body itself (it holds the hand
    collision geom) but strips all finger kinematic chains.
    """
    _FINGER_TOKENS = ("thumb", "index", "middle", "ring", "little")
    for hand_name in ("left_inspire_hand", "right_inspire_hand"):
        hand_body = _find_body(root, hand_name)
        if hand_body is None:
            continue
        for child in list(hand_body):
            if child.tag != "body":
                continue
            child_name = (child.get("name") or "").lower()
            if any(tok in child_name for tok in _FINGER_TOKENS):
                hand_body.remove(child)


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
        parent = trunk if spec["parent"] == "Trunk" else _find_body(
            trunk, spec["parent"]
        )
        if parent is None:
            continue
        if any(
            child.tag == "geom" and child.get("name") == spec["name"]
            for child in parent
        ):
            continue

        geom = ET.Element(
            "geom",
            {
                "name": spec["name"],
                "type": spec["type"],
                "size": spec["size"],
                "pos": spec["pos"],
                "rgba": "0 0 0 0",
            },
        )
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
