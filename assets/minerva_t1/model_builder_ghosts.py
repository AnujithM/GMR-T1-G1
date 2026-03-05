"""Ghost-model helpers for the T1 MJCF model builder."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Sequence
import xml.etree.ElementTree as ET

from .model_builder_assembly import _find_body
from .model_builder_config import GhostType


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
        if child.get("name") in (
            "Trunk_des_ghost",
            "Trunk_ik_ghost",
            "Trunk_warmstart_ghost",
        ):
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
        for variant, rgba in _GHOST_VARIANTS:
            suffix = f"_{variant}_ghost"
            body = _build_articulated_ghost(
                base_name=f"Trunk{suffix}",
                base_pos="0 0 0.7",
                freejoint_name=f"Trunk{suffix}_free",
                suffix=suffix,
                rgba=rgba,
                joint_specs=joint_specs,
                hand_templates=hand_templates,
            )
            worldbody.append(body)


# (variant_name, rgba) for each articulated ghost robot.
_GHOST_VARIANTS = [
    ("ik",  "1.0 0.8 0.2 0.4"),  # green/gold
    ("des", "0.3 0.5 1.0 0.5"),  # blue
    ("warmstart", "1.0 0.6 0.2 0.4"),  # orange
]


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
        body.append(
            ET.Element(
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "1e-4",
                    "diaginertia": "1e-6 1e-6 1e-6",
                },
            )
        )
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
    waist_pos_str = (
        f"{waist_world_pos[0]} {waist_world_pos[1]} {waist_world_pos[2]}"
    )

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
        left_hand = make_body(
            "left_inspire_hand", "-0.00001093 0.10132774 -0.00093692"
        )
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
        right_hand = make_body(
            "right_inspire_hand", "0.00001093 -0.10132774 -0.00093692"
        )
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
