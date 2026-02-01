"""Parity tests between t1_robot.urdf and t1_robot.xml.

Validates that the URDF (for training) and XML (for deployment) have
consistent dynamics where applicable, while allowing for differences
in finger inertia handling (zero in URDF, tiny in XML).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ASSETS_DIR = Path(__file__).parent.parent
URDF_PATH = ASSETS_DIR / "t1_robot.urdf"
XML_PATH = ASSETS_DIR / "t1_robot.xml"

EXPECTED_INSPIRE_HAND_MASS = 0.62  # Gripper mass (merged from hand_link + inspire_hand)
MASS_TOLERANCE = 0.01
INERTIA_TOLERANCE = 1e-4
QUAT_ANGLE_TOLERANCE_DEG = 5.0  # Max angle difference for principal axes


def parse_urdf_links(urdf_path: Path) -> dict:
    """Parse URDF and return dict of link name -> mass."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {}
    for link in root.findall(".//link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                links[name] = float(mass_elem.get("value"))
    return links


def parse_urdf_inertias(urdf_path: Path) -> dict:
    """Parse URDF and return dict of link name -> inertia tensor (6 values).

    Returns:
        Dict mapping link name to (ixx, iyy, izz, ixy, ixz, iyz).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    inertias = {}
    for link in root.findall(".//link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is not None:
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.get("ixx", 0))
                iyy = float(inertia_elem.get("iyy", 0))
                izz = float(inertia_elem.get("izz", 0))
                ixy = float(inertia_elem.get("ixy", 0))
                ixz = float(inertia_elem.get("ixz", 0))
                iyz = float(inertia_elem.get("iyz", 0))
                inertias[name] = (ixx, iyy, izz, ixy, ixz, iyz)
    return inertias


def inertia_to_principal(ixx: float, iyy: float, izz: float,
                         ixy: float, ixz: float, iyz: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize inertia tensor to get principal moments and orientation.

    Args:
        ixx, iyy, izz: Diagonal components of inertia tensor.
        ixy, ixz, iyz: Off-diagonal components.

    Returns:
        Tuple of (principal_inertia, quaternion_wxyz).
        principal_inertia: Sorted principal moments [I1, I2, I3].
        quaternion_wxyz: Rotation from body frame to principal frame (w, x, y, z).
    """
    I = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz]
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(I)
    # Sort by eigenvalue ascending
    idx = np.argsort(eigenvalues)
    principal_inertia = eigenvalues[idx]
    R = eigenvectors[:, idx]
    # Ensure right-handed
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    # Convert to quaternion (scipy returns xyzw, we want wxyz)
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return principal_inertia, quat_wxyz


def quaternion_angle_diff(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angle difference between two quaternions in degrees.

    Args:
        q1, q2: Quaternions in wxyz format.

    Returns:
        Angle difference in degrees (0-180).
    """
    # Normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # Handle quaternion double-cover
    dot = np.abs(np.dot(q1, q2))
    dot = min(dot, 1.0)
    angle_rad = 2 * np.arccos(dot)
    return np.degrees(angle_rad)


def parse_urdf_joints(urdf_path: Path) -> dict:
    """Parse URDF and return dict of joint name -> (lower, upper, type)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        if jtype == "revolute":
            limit = joint.find("limit")
            if limit is not None:
                lower = float(limit.get("lower", 0))
                upper = float(limit.get("upper", 0))
                joints[name] = (lower, upper, jtype)
    return joints


class TestURDFValidity:
    """Tests for URDF file validity."""

    def test_urdf_parses(self):
        """URDF parses without errors."""
        tree = ET.parse(URDF_PATH)
        root = tree.getroot()
        assert root.tag == "robot"

    def test_urdf_link_count(self):
        """URDF has 54 links (28 body + 26 finger/hand, hand_link merged into inspire_hand)."""
        tree = ET.parse(URDF_PATH)
        root = tree.getroot()
        links = root.findall(".//link")
        assert len(links) == 54


class TestXMLValidity:
    """Tests for MuJoCo XML file validity."""

    def test_xml_loads(self):
        """XML loads in MuJoCo without errors."""
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        assert model is not None


class TestHandMassParity:
    """Hand mass should match between URDF and XML (hand_link merged into inspire_hand)."""

    def test_left_inspire_hand_urdf(self):
        """Left inspire_hand (gripper) mass is 0.62 kg in URDF."""
        links = parse_urdf_links(URDF_PATH)
        assert abs(links["left_inspire_hand"] - EXPECTED_INSPIRE_HAND_MASS) < MASS_TOLERANCE

    def test_right_inspire_hand_urdf(self):
        """Right inspire_hand (gripper) mass is 0.62 kg in URDF."""
        links = parse_urdf_links(URDF_PATH)
        assert abs(links["right_inspire_hand"] - EXPECTED_INSPIRE_HAND_MASS) < MASS_TOLERANCE

    def test_left_inspire_hand_xml(self):
        """Left inspire_hand (gripper) mass is 0.62 kg in XML."""
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_inspire_hand")
        assert abs(model.body_mass[idx] - EXPECTED_INSPIRE_HAND_MASS) < MASS_TOLERANCE

    def test_right_inspire_hand_xml(self):
        """Right inspire_hand (gripper) mass is 0.62 kg in XML."""
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_inspire_hand")
        assert abs(model.body_mass[idx] - EXPECTED_INSPIRE_HAND_MASS) < MASS_TOLERANCE


class TestFingerMass:
    """Finger mass handling differs: zero in URDF, tiny in XML."""

    def test_urdf_fingers_zero(self):
        """All finger links have zero mass in URDF."""
        links = parse_urdf_links(URDF_PATH)
        finger_kw = ["thumb", "index", "middle", "ring", "little"]
        for name, mass in links.items():
            if any(kw in name for kw in finger_kw):
                assert mass == 0.0, f"{name} should have zero mass"

    def test_xml_fingers_negligible(self):
        """Finger masses in XML are negligible (< 0.01 kg)."""
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        finger_kw = ["thumb", "index", "middle", "ring", "little"]
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and any(kw in name for kw in finger_kw):
                assert model.body_mass[i] < 0.01, f"{name} mass too high"


class TestActuatedJoints:
    """Actuated joints should have matching limits."""

    def test_main_joint_count(self):
        """Both have 29 actuated DOF (excluding finger joints)."""
        urdf_joints = parse_urdf_joints(URDF_PATH)
        main_joints = {k: v for k, v in urdf_joints.items() 
                       if not any(x in k for x in ["thumb", "index", "middle", "ring", "little"])}
        assert len(main_joints) == 29

    def test_joint_limits_match(self):
        """All 29 main joint limits match between URDF and XML."""
        urdf_joints = parse_urdf_joints(URDF_PATH)
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        
        for name, (u_lower, u_upper, _) in urdf_joints.items():
            if any(x in name for x in ["thumb", "index", "middle", "ring", "little"]):
                continue
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx < 0:
                continue
            x_lower, x_upper = model.jnt_range[idx]
            assert abs(u_lower - x_lower) < 0.01, f"{name} lower mismatch"
            assert abs(u_upper - x_upper) < 0.01, f"{name} upper mismatch"


class TestMassParity:
    """Non-finger body masses should match."""

    def test_non_finger_masses_match(self):
        """All non-finger link/body masses match within tolerance."""
        urdf_links = parse_urdf_links(URDF_PATH)
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        
        finger_kw = ["thumb", "index", "middle", "ring", "little", "inspire"]
        for name, urdf_mass in urdf_links.items():
            if any(kw in name for kw in finger_kw):
                continue
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx < 0:
                continue
            xml_mass = model.body_mass[idx]
            assert abs(urdf_mass - xml_mass) < 0.01, (
                f"{name}: URDF={urdf_mass:.4f}, XML={xml_mass:.4f}"
            )


class TestInertiaParity:
    """Inertia tensors should match between URDF and XML."""

    def test_non_finger_principal_inertias_match(self):
        """Principal inertia values match within tolerance for main bodies."""
        urdf_inertias = parse_urdf_inertias(URDF_PATH)
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))

        finger_kw = ["thumb", "index", "middle", "ring", "little", "inspire"]
        mismatches = []

        for name, urdf_tensor in urdf_inertias.items():
            if any(kw in name for kw in finger_kw):
                continue
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx < 0:
                continue

            # URDF principal inertia (computed from 6-element tensor)
            urdf_principal, _ = inertia_to_principal(*urdf_tensor)

            # XML stores diagonalized inertia directly in body_inertia
            xml_diaginertia = model.body_inertia[idx]

            # Compare sorted principal values
            for i, (u_val, x_val) in enumerate(zip(
                sorted(urdf_principal), sorted(xml_diaginertia)
            )):
                if abs(u_val - x_val) > INERTIA_TOLERANCE:
                    mismatches.append(
                        f"{name} I[{i}]: URDF={u_val:.6f}, XML={x_val:.6f}"
                    )

        assert not mismatches, "Inertia mismatches:\n" + "\n".join(mismatches)

    def test_non_finger_inertia_tensors_equivalent(self):
        """Full 3x3 inertia tensors are dynamically equivalent.

        This checks that the reconstructed inertia matrices match, not just
        the quaternions. Different (quat, diaginertia) pairs can represent
        the same physical tensor.
        """
        urdf_inertias = parse_urdf_inertias(URDF_PATH)
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))

        finger_kw = ["thumb", "index", "middle", "ring", "little", "inspire"]
        results = []
        mismatches = []

        for name, urdf_tensor in urdf_inertias.items():
            if any(kw in name for kw in finger_kw):
                continue
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx < 0:
                continue

            # URDF: direct 3x3 matrix
            ixx, iyy, izz, ixy, ixz, iyz = urdf_tensor
            I_urdf = np.array([
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz]
            ])

            # XML: reconstruct from diaginertia + quat
            xml_diag = model.body_inertia[idx]
            xml_quat_wxyz = model.body_iquat[idx]
            quat_xyzw = np.array([
                xml_quat_wxyz[1], xml_quat_wxyz[2],
                xml_quat_wxyz[3], xml_quat_wxyz[0]
            ])
            R = Rotation.from_quat(quat_xyzw).as_matrix()
            I_xml = R @ np.diag(xml_diag) @ R.T

            # Compare tensors
            urdf_norm = np.linalg.norm(I_urdf, 'fro')
            xml_norm = np.linalg.norm(I_xml, 'fro')
            abs_diff = np.linalg.norm(I_urdf - I_xml, 'fro')
            rel_err = abs_diff / max(urdf_norm, 1e-12)

            status = "OK" if rel_err < 0.01 else "DIFF"
            results.append(
                f"{name:<25} {urdf_norm:>12.6f} {xml_norm:>12.6f} "
                f"{abs_diff:>12.6f} {rel_err*100:>10.2f}% {status}"
            )
            if rel_err > 0.01:
                mismatches.append(name)

        # Build detailed report for failure message
        header = (
            f"{'Body':<25} {'URDF norm':>12} {'XML norm':>12} "
            f"{'Abs Diff':>12} {'Rel Err':>11} Status\n"
            + "-" * 85
        )
        report = header + "\n" + "\n".join(results)

        assert not mismatches, f"Tensor mismatches:\n{report}"

    def test_hand_inertia_tensors_equivalent(self):
        """Inspire hand 3x3 inertia tensors match between URDF and XML."""
        urdf_inertias = parse_urdf_inertias(URDF_PATH)
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))

        for hand in ["left_inspire_hand", "right_inspire_hand"]:
            urdf_tensor = urdf_inertias[hand]
            ixx, iyy, izz, ixy, ixz, iyz = urdf_tensor
            I_urdf = np.array([
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz]
            ])

            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand)
            xml_diag = model.body_inertia[idx]
            xml_quat_wxyz = model.body_iquat[idx]
            quat_xyzw = np.array([
                xml_quat_wxyz[1], xml_quat_wxyz[2],
                xml_quat_wxyz[3], xml_quat_wxyz[0]
            ])
            R = Rotation.from_quat(quat_xyzw).as_matrix()
            I_xml = R @ np.diag(xml_diag) @ R.T

            rel_err = np.linalg.norm(I_urdf - I_xml) / np.linalg.norm(I_urdf)
            assert rel_err < 0.01, (
                f"{hand} inertia tensor mismatch: {rel_err*100:.2f}% error"
            )

