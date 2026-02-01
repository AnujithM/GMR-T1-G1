"""Tests for T1 model builder ghost generation."""

import pytest
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from minerva_bringup.assets.t1.model_builder import (
    T1ModelBuilder,
    ModelConfig,
    BodySubset,
    GhostType,
)


class TestModelBuilder:
    """Tests for T1ModelBuilder class."""

    def test_build_with_ghost_creates_ik_and_desired_ghosts(self) -> None:
        """Ghost generation should create both IK and desired ghost robots."""
        builder = T1ModelBuilder()
        config = ModelConfig(ghost_type=GhostType.IK_ROBOT)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.xml"
            result = builder.build(config, output_path)
            
            content = result.read_text()
            assert "Trunk_ik_ghost" in content
            assert "Trunk_des_ghost" in content

    def test_build_without_ghost_has_no_ghosts(self) -> None:
        """Building without ghost type should not include ghost robots."""
        builder = T1ModelBuilder()
        config = ModelConfig(ghost_type=GhostType.NONE)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.xml"
            result = builder.build(config, output_path)
            
            content = result.read_text()
            assert "Trunk_ik_ghost" not in content
            assert "Trunk_des_ghost" not in content

    def test_ghost_joints_use_xml_limits(self) -> None:
        """Ghost joint limits should come from the main XML, not hardcoded."""
        builder = T1ModelBuilder()
        config = ModelConfig(ghost_type=GhostType.IK_ROBOT)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.xml"
            result = builder.build(config, output_path)
            
            tree = ET.parse(result)
            root = tree.getroot()

            base_tree = ET.parse(builder.base_xml_path)
            base_root = base_tree.getroot()
            expected_range = None
            for joint in base_root.iter("joint"):
                if joint.get("name") == "Left_Shoulder_Pitch":
                    expected_range = joint.get("range")
                    break
            assert expected_range is not None
            exp_low, exp_high = map(float, expected_range.split())
            
            for joint in root.iter("joint"):
                if joint.get("name") == "Left_Shoulder_Pitch_ik_ghost":
                    range_attr = joint.get("range")
                    assert range_attr is not None
                    low, high = map(float, range_attr.split())
                    assert abs(low - exp_low) < 1e-6
                    assert abs(high - exp_high) < 1e-6
                    return
            
            pytest.fail("Left_Shoulder_Pitch_ik_ghost joint not found")

    def test_ghost_has_all_upper_body_joints(self) -> None:
        """Ghost robot should have all expected upper body joints."""
        builder = T1ModelBuilder()
        config = ModelConfig(ghost_type=GhostType.IK_ROBOT)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.xml"
            result = builder.build(config, output_path)
            
            tree = ET.parse(result)
            root = tree.getroot()
            
            joint_names = [j.get("name") for j in root.iter("joint")]
            
            expected_ghost_joints = [
                "AAHead_yaw_ik_ghost",
                "Head_pitch_ik_ghost",
                "Left_Shoulder_Pitch_ik_ghost",
                "Left_Shoulder_Roll_ik_ghost", 
                "Left_Elbow_Pitch_ik_ghost",
                "Left_Elbow_Yaw_ik_ghost",
                "Left_Wrist_Pitch_ik_ghost",
                "Left_Wrist_Yaw_ik_ghost",
                "Left_Hand_Roll_ik_ghost",
                "Right_Shoulder_Pitch_ik_ghost",
                "Right_Shoulder_Roll_ik_ghost",
                "Right_Elbow_Pitch_ik_ghost",
                "Right_Elbow_Yaw_ik_ghost",
                "Right_Wrist_Pitch_ik_ghost",
                "Right_Wrist_Yaw_ik_ghost",
                "Right_Hand_Roll_ik_ghost",
            ]
            
            for expected in expected_ghost_joints:
                assert expected in joint_names, f"Missing ghost joint: {expected}"

    def test_upper_body_subset_removes_legs(self) -> None:
        """UPPER_BODY subset should not include leg bodies but keeps Waist."""
        builder = T1ModelBuilder()
        config = ModelConfig(body_subset=BodySubset.UPPER_BODY)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.xml"
            result = builder.build(config, output_path)
            
            content = result.read_text()
            assert "Trunk" in content
            assert "Waist" in content  # Waist kept for IK control
            # Leg bodies should be removed
            assert "Left_Hip_Pitch" not in content
            assert "Right_Hip_Pitch" not in content
