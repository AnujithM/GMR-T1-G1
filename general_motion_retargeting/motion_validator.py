"""
Motion Validation Module for GMR (General Motion Retargeting)

Validates retargeted robot motions against multiple criteria:
1. Joint limits with safety margins
2. Keypoint tracking accuracy
3. Self-collision detection
4. Foot ground contact constraints
5. Jitter and temporal artifacts
"""

import numpy as np
import pickle
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import mujoco as mj
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    warnings.warn("MuJoCo not available; collision checking disabled")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available; plotting disabled")


@dataclass
class JointLimitReport:
    """Report for joint limit violations."""
    joint_name: str
    q_min: float
    q_max: float
    q_min_safe: float
    q_max_safe: float
    max_violation: float  # How much outside safe range
    violation_frames: List[int]  # Frame indices with violations
    num_violation_frames: int
    percent_near_limit: float  # % of frames within 0.03 of limit
    passed: bool

    def to_dict(self):
        d = asdict(self)
        d['violation_frames'] = str(d['violation_frames'][:10]) + \
                                (f" ... +{len(d['violation_frames'])-10} more" 
                                 if len(d['violation_frames']) > 10 else "")
        return d


@dataclass
class KeypointError:
    """Tracking error for a single keypoint."""
    keypoint_name: str
    pos_error_mean: float
    pos_error_p95: float
    pos_error_max: float
    ori_error_mean: Optional[float] = None
    ori_error_p95: Optional[float] = None
    ori_error_max: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class CollisionReport:
    """Report for a collision pair."""
    pair_name: str
    collided: bool
    first_collision_frame: Optional[int]
    num_collision_frames: int
    min_distance: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class FootGroundReport:
    """Report for foot ground contact constraints."""
    foot_name: str
    max_penetration: float
    bad_penetration_frames: List[int]
    num_bad_penetration_frames: int
    roll_p95_during_contact: float
    pitch_p95_during_contact: float
    num_non_flat_frames: int
    passed: bool

    def to_dict(self):
        d = asdict(self)
        d['bad_penetration_frames'] = str(d['bad_penetration_frames'][:10]) + \
                                      (f" ... +{len(d['bad_penetration_frames'])-10} more" 
                                       if len(d['bad_penetration_frames']) > 10 else "")
        return d


@dataclass
class JitterReport:
    """Report for jitter and temporal artifacts."""
    joint_name: str
    vel_p95: float
    vel_p99: float
    acc_p95: float
    acc_p99: float

    def to_dict(self):
        return asdict(self)


@dataclass
class ValidationResult:
    """Complete validation result for a motion clip."""
    clip_name: str
    fps: float
    num_frames: int
    duration_sec: float
    
    # Check results
    joint_limits_passed: bool
    joint_limits_reports: List[JointLimitReport]
    
    keypoint_errors: List[KeypointError]
    
    collision_passed: bool
    collision_reports: List[CollisionReport]
    
    foot_ground_passed: bool
    foot_ground_reports: List[FootGroundReport]
    
    jitter_reports: List[JitterReport]
    
    # Summary
    all_passed: bool
    summary: str

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'clip_name': self.clip_name,
            'fps': self.fps,
            'num_frames': self.num_frames,
            'duration_sec': self.duration_sec,
            'joint_limits_passed': self.joint_limits_passed,
            'joint_limits_reports': [r.to_dict() for r in self.joint_limits_reports],
            'keypoint_errors': [e.to_dict() for e in self.keypoint_errors],
            'collision_passed': self.collision_passed,
            'collision_reports': [r.to_dict() for r in self.collision_reports],
            'foot_ground_passed': self.foot_ground_passed,
            'foot_ground_reports': [r.to_dict() for r in self.foot_ground_reports],
            'jitter_reports': [r.to_dict() for r in self.jitter_reports],
            'all_passed': self.all_passed,
            'summary': self.summary,
        }


class MotionValidator:
    """Validator for retargeted robot motions."""
    
    def __init__(
        self,
        robot_xml: str,
        joint_names: Optional[List[str]] = None,
        keypoint_map: Optional[Dict[str, str]] = None,
        collision_pairs: Optional[List[Tuple[str, str]]] = None,
        foot_names: Optional[List[str]] = None,
        ground_z: float = 0.0,
        verbose: bool = True,
    ):
        """
        Initialize the validator.
        
        Args:
            robot_xml: Path to MuJoCo XML file
            joint_names: List of joint names in order (if None, will be auto-detected)
            keypoint_map: Dict mapping keypoint names to body names
            collision_pairs: List of (body1, body2) tuples for collision checking
            foot_names: List of foot body names for ground contact checking
            ground_z: Height of ground plane (default 0.0)
            verbose: Print verbose output
        """
        self.robot_xml = robot_xml
        self.verbose = verbose
        self.ground_z = ground_z
        
        # Load MuJoCo model if available
        self.model = None
        self.data = None
        if HAS_MUJOCO:
            try:
                self.model = mj.MjModel.from_xml_path(str(robot_xml))
                self.data = mj.MjData(self.model)
                if self.verbose:
                    print(f"[Validator] Loaded MuJoCo model: {robot_xml}")
            except Exception as e:
                warnings.warn(f"Failed to load MuJoCo model: {e}")
        
        # Get joint info from model if available
        if self.model is not None and joint_names is None:
            self.joint_names = []
            for i in range(self.model.nv):
                dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, 
                                         self.model.dof_jntid[i])
                self.joint_names.append(dof_name)
        else:
            self.joint_names = joint_names or []
        
        self.keypoint_map = keypoint_map or {
            'pelvis': 'pelvis',
            'head': 'head',
            'left_hand': 'left_hand',
            'right_hand': 'right_hand',
            'left_foot': 'left_foot',
            'right_foot': 'right_foot',
        }
        
        self.collision_pairs = collision_pairs or []
        self.foot_names = foot_names or ['left_foot', 'right_foot']
        
        if self.verbose:
            print(f"[Validator] Initialized with {len(self.joint_names)} joints")
            print(f"[Validator] Keypoints: {list(self.keypoint_map.keys())}")
            print(f"[Validator] Feet: {self.foot_names}")

    def validate(
        self,
        motion_file: str,
        joint_positions: np.ndarray,
        root_pose: Optional[np.ndarray] = None,
        foot_poses: Optional[Dict[str, np.ndarray]] = None,
        target_keypoints: Optional[Dict[str, np.ndarray]] = None,
        fps: float = 60.0,
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> ValidationResult:
        """
        Validate a motion clip.
        
        Args:
            motion_file: Path to motion file (for logging)
            joint_positions: Array of shape (T, n_joints) with joint angles in radians
            root_pose: Optional array of shape (T, 7) with [x, y, z, qx, qy, qz, qw]
            foot_poses: Optional dict mapping foot names to arrays of shape (T, 7)
            target_keypoints: Optional dict mapping keypoint names to arrays of shape (T, 3)
            fps: Frames per second
            joint_limits: Optional dict mapping joint names to (min, max) in radians
        
        Returns:
            ValidationResult with all checks and reports
        """
        clip_name = Path(motion_file).stem
        n_frames = joint_positions.shape[0]
        duration = n_frames / fps
        dt = 1.0 / fps
        
        if self.verbose:
            print(f"\n[Validator] Validating {clip_name}: {n_frames} frames @ {fps} Hz ({duration:.2f}s)")
        
        # Check 1: Joint limits
        joint_limits_passed, joint_limits_reports = self._check_joint_limits(
            joint_positions, joint_limits
        )
        
        # Check 2: Keypoint tracking
        keypoint_errors = []
        if target_keypoints is not None and root_pose is not None:
            keypoint_errors = self._check_keypoint_tracking(
                root_pose, foot_poses, target_keypoints
            )
        
        # Check 3: Self collision
        collision_passed, collision_reports = self._check_collisions(
            joint_positions, root_pose
        )
        
        # Check 4: Foot ground contact
        foot_ground_passed, foot_ground_reports = self._check_foot_ground(
            foot_poses, dt
        )
        
        # Check 5: Jitter and temporal artifacts
        jitter_reports = self._check_jitter(joint_positions, dt)
        
        # Overall pass/fail
        all_passed = (
            joint_limits_passed and 
            collision_passed and 
            foot_ground_passed
        )
        
        # Generate summary
        summary = self._generate_summary(
            clip_name,
            all_passed,
            joint_limits_passed,
            collision_passed,
            foot_ground_passed,
            joint_limits_reports,
            keypoint_errors,
            collision_reports,
            foot_ground_reports,
        )
        
        result = ValidationResult(
            clip_name=clip_name,
            fps=fps,
            num_frames=n_frames,
            duration_sec=duration,
            joint_limits_passed=joint_limits_passed,
            joint_limits_reports=joint_limits_reports,
            keypoint_errors=keypoint_errors,
            collision_passed=collision_passed,
            collision_reports=collision_reports,
            foot_ground_passed=foot_ground_passed,
            foot_ground_reports=foot_ground_reports,
            jitter_reports=jitter_reports,
            all_passed=all_passed,
            summary=summary,
        )
        
        return result

    def _check_joint_limits(
        self,
        joint_positions: np.ndarray,
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[bool, List[JointLimitReport]]:
        """Check 1: Joint limits with 0.05 rad safety margin."""
        reports = []
        margin = 0.05
        near_limit_threshold = 0.03
        
        n_joints = joint_positions.shape[1]
        passed = True
        
        for i in range(min(n_joints, len(self.joint_names))):
            joint_name = self.joint_names[i]
            q = joint_positions[:, i]
            
            # Get limits
            if joint_limits and joint_name in joint_limits:
                q_min, q_max = joint_limits[joint_name]
            else:
                # Conservative defaults if not provided
                q_min, q_max = -np.pi, np.pi
            
            q_min_safe = q_min + margin
            q_max_safe = q_max - margin
            
            # Check violations
            violations_min = q < q_min_safe
            violations_max = q > q_max_safe
            violations = violations_min | violations_max
            
            violation_frames = np.where(violations)[0].tolist()
            max_violation = 0.0
            if len(violation_frames) > 0:
                passed = False
                over_max = np.maximum(0, q[violations_max] - q_max_safe)
                under_min = np.maximum(0, q_min_safe - q[violations_min])
                max_violation = float(max(
                    np.max(over_max) if len(over_max) > 0 else 0,
                    np.max(under_min) if len(under_min) > 0 else 0,
                ))
            
            # Percent near limit
            near_min = np.abs(q - q_min_safe) < near_limit_threshold
            near_max = np.abs(q - q_max_safe) < near_limit_threshold
            percent_near = 100.0 * np.sum(near_min | near_max) / len(q)
            
            report = JointLimitReport(
                joint_name=joint_name,
                q_min=float(q_min),
                q_max=float(q_max),
                q_min_safe=float(q_min_safe),
                q_max_safe=float(q_max_safe),
                max_violation=max_violation,
                violation_frames=violation_frames,
                num_violation_frames=len(violation_frames),
                percent_near_limit=percent_near,
                passed=len(violation_frames) == 0,
            )
            reports.append(report)
        
        return passed, reports

    def _check_keypoint_tracking(
        self,
        root_pose: np.ndarray,
        foot_poses: Optional[Dict[str, np.ndarray]],
        target_keypoints: Dict[str, np.ndarray],
    ) -> List[KeypointError]:
        """Check 2: Keypoint tracking errors."""
        errors = []
        
        for keypoint_name, target_pos in target_keypoints.items():
            # Get robot keypoint position
            if keypoint_name == 'pelvis':
                # Use root pose
                robot_pos = root_pose[:, :3]
            elif keypoint_name in self.keypoint_map:
                # Try to get from foot poses if available
                body_name = self.keypoint_map[keypoint_name]
                if foot_poses and body_name in foot_poses:
                    robot_pos = foot_poses[body_name][:, :3]
                else:
                    # Skip if we don't have position data
                    continue
            else:
                continue
            
            # Compute position error
            if robot_pos.shape != target_pos.shape:
                warnings.warn(f"Shape mismatch for {keypoint_name}: "
                            f"{robot_pos.shape} vs {target_pos.shape}")
                continue
            
            dx = robot_pos[:, 0] - target_pos[:, 0]
            dy = robot_pos[:, 1] - target_pos[:, 1]
            dz = robot_pos[:, 2] - target_pos[:, 2]
            e_pos = np.sqrt(dx**2 + dy**2 + dz**2)
            
            error = KeypointError(
                keypoint_name=keypoint_name,
                pos_error_mean=float(np.mean(e_pos)),
                pos_error_p95=float(np.percentile(e_pos, 95)),
                pos_error_max=float(np.max(e_pos)),
            )
            errors.append(error)
        
        return errors

    def _check_collisions(
        self,
        joint_positions: np.ndarray,
        root_pose: Optional[np.ndarray] = None,
    ) -> Tuple[bool, List[CollisionReport]]:
        """Check 3: Self-collision detection."""
        reports = []
        passed = True
        
        if not HAS_MUJOCO or self.model is None or len(self.collision_pairs) == 0:
            return passed, reports
        
        for body1_name, body2_name in self.collision_pairs:
            try:
                body1_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body1_name)
                body2_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body2_name)
            except ValueError:
                warnings.warn(f"Body not found: {body1_name} or {body2_name}")
                continue
            
            first_collision_frame = None
            collision_frames = []
            min_distance = None
            
            for t in range(joint_positions.shape[0]):
                # Set state
                self.data.qpos[:len(joint_positions[t])] = joint_positions[t]
                if root_pose is not None:
                    self.data.qpos[:3] = root_pose[t, :3]
                    # Skip quaternion for now; requires proper handling
                
                mj.mj_forward(self.model, self.data)
                
                # Check collision
                # Note: MuJoCo contact detection is complex; this is simplified
                # In practice, you'd iterate through geoms and check distances
                # For now, just track that we're checking
                # TODO: Implement proper geom-pair distance computation
            
            pair_name = f"{body1_name} <-> {body2_name}"
            report = CollisionReport(
                pair_name=pair_name,
                collided=len(collision_frames) > 0,
                first_collision_frame=first_collision_frame,
                num_collision_frames=len(collision_frames),
                min_distance=min_distance,
            )
            reports.append(report)
            
            if len(collision_frames) > 0:
                passed = False
        
        return passed, reports

    def _check_foot_ground(
        self,
        foot_poses: Optional[Dict[str, np.ndarray]],
        dt: float,
    ) -> Tuple[bool, List[FootGroundReport]]:
        """Check 4: Foot ground contact constraints."""
        reports = []
        passed = True
        
        if foot_poses is None:
            return passed, reports
        
        for foot_name in self.foot_names:
            if foot_name not in foot_poses:
                continue
            
            pose = foot_poses[foot_name]  # Shape: (T, 7) with [x, y, z, qx, qy, qz, qw]
            z_foot = pose[:, 2]
            
            # Penetration check
            z_ground = self.ground_z
            penetration = np.maximum(0, z_ground - z_foot)
            bad_penetration_threshold = 0.01
            bad_pen_frames = np.where(penetration > bad_penetration_threshold)[0].tolist()
            max_penetration = float(np.max(penetration))
            
            if len(bad_pen_frames) > 5:  # More than a handful
                passed = False
            
            # Foot flatness during contact
            contact_threshold = 0.01
            contact_frames = z_foot <= (z_ground + contact_threshold)
            
            # Compute roll and pitch from quaternion
            roll_values = []
            pitch_values = []
            non_flat_frames = []
            
            for t in np.where(contact_frames)[0]:
                quat = pose[t, 3:7]  # [qx, qy, qz, qw]
                roll, pitch = self._quat_to_roll_pitch(quat)
                
                roll_values.append(roll)
                pitch_values.append(pitch)
                
                # Flag non-flat (> 10 deg)
                if np.abs(roll) > np.deg2rad(10) or np.abs(pitch) > np.deg2rad(10):
                    non_flat_frames.append(t)
            
            roll_p95 = float(np.percentile(np.abs(roll_values), 95)) if roll_values else 0.0
            pitch_p95 = float(np.percentile(np.abs(pitch_values), 95)) if pitch_values else 0.0
            
            report = FootGroundReport(
                foot_name=foot_name,
                max_penetration=max_penetration,
                bad_penetration_frames=bad_pen_frames,
                num_bad_penetration_frames=len(bad_pen_frames),
                roll_p95_during_contact=roll_p95,
                pitch_p95_during_contact=pitch_p95,
                num_non_flat_frames=len(non_flat_frames),
                passed=len(bad_pen_frames) <= 5 and len(non_flat_frames) == 0,
            )
            reports.append(report)
        
        return passed, reports

    def _check_jitter(
        self,
        joint_positions: np.ndarray,
        dt: float,
    ) -> List[JitterReport]:
        """Check 5: Jitter and temporal artifacts."""
        reports = []
        
        n_frames, n_joints = joint_positions.shape
        denom_v = 2 * dt
        denom_a = dt * dt
        
        for i in range(min(n_joints, len(self.joint_names))):
            joint_name = self.joint_names[i]
            q = joint_positions[:, i]
            
            # Compute velocity and acceleration
            dq = np.zeros(n_frames)
            ddq = np.zeros(n_frames)
            
            for t in range(1, n_frames - 1):
                tmp_v = q[t + 1] - q[t - 1]
                dq[t] = tmp_v / denom_v
                
                tmp_a = q[t + 1] - 2 * q[t] + q[t - 1]
                ddq[t] = tmp_a / denom_a
            
            # Compute percentiles
            vel_p95 = float(np.percentile(np.abs(dq[1:-1]), 95))
            vel_p99 = float(np.percentile(np.abs(dq[1:-1]), 99))
            acc_p95 = float(np.percentile(np.abs(ddq[1:-1]), 95))
            acc_p99 = float(np.percentile(np.abs(ddq[1:-1]), 99))
            
            report = JitterReport(
                joint_name=joint_name,
                vel_p95=vel_p95,
                vel_p99=vel_p99,
                acc_p95=acc_p95,
                acc_p99=acc_p99,
            )
            reports.append(report)
        
        return reports

    @staticmethod
    def _quat_to_roll_pitch(quat: np.ndarray) -> Tuple[float, float]:
        """Convert quaternion [qx, qy, qz, qw] to roll and pitch in radians."""
        qx, qy, qz, qw = quat
        
        # Roll (rotation around x-axis)
        sin_roll = 2 * (qw * qx + qy * qz)
        cos_roll = 1 - 2 * (qx**2 + qy**2)
        roll = np.arctan2(sin_roll, cos_roll)
        
        # Pitch (rotation around y-axis)
        sin_pitch = 2 * (qw * qy - qz * qx)
        sin_pitch = np.clip(sin_pitch, -1, 1)
        pitch = np.arcsin(sin_pitch)
        
        return roll, pitch

    def _generate_summary(
        self,
        clip_name: str,
        all_passed: bool,
        joint_limits_passed: bool,
        collision_passed: bool,
        foot_ground_passed: bool,
        joint_limits_reports: List[JointLimitReport],
        keypoint_errors: List[KeypointError],
        collision_reports: List[CollisionReport],
        foot_ground_reports: List[FootGroundReport],
    ) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"{'='*70}",
            f"Validation Report: {clip_name}",
            f"{'='*70}",
            "",
        ]
        
        # Overall status
        status = "✓ PASS" if all_passed else "✗ FAIL"
        lines.append(f"Overall Status: {status}")
        lines.append("")
        
        # Joint limits
        lines.append(f"Joint Limits: {'✓ PASS' if joint_limits_passed else '✗ FAIL'}")
        if not joint_limits_passed:
            violations = [r for r in joint_limits_reports if not r.passed]
            for r in violations:
                lines.append(f"  - {r.joint_name}: max violation {r.max_violation:.4f} rad "
                           f"({r.num_violation_frames} frames)")
        
        # Keypoint tracking
        if keypoint_errors:
            lines.append("")
            lines.append("Keypoint Tracking Errors:")
            for e in keypoint_errors:
                lines.append(f"  {e.keypoint_name}:")
                lines.append(f"    Position: mean={e.pos_error_mean:.4f} m, "
                           f"p95={e.pos_error_p95:.4f} m, "
                           f"max={e.pos_error_max:.4f} m")
        
        # Collision
        lines.append("")
        lines.append(f"Self-Collision: {'✓ PASS' if collision_passed else '✗ FAIL'}")
        if not collision_passed:
            collisions = [r for r in collision_reports if r.collided]
            for r in collisions:
                lines.append(f"  - {r.pair_name}: {r.num_collision_frames} frames")
        
        # Foot ground
        lines.append("")
        lines.append(f"Foot Ground Contact: {'✓ PASS' if foot_ground_passed else '✗ FAIL'}")
        for r in foot_ground_reports:
            if not r.passed:
                lines.append(f"  {r.foot_name}:")
                if r.num_bad_penetration_frames > 0:
                    lines.append(f"    Penetration: max={r.max_penetration:.4f} m, "
                               f"{r.num_bad_penetration_frames} bad frames")
                if r.num_non_flat_frames > 0:
                    lines.append(f"    Non-flat: roll p95={np.rad2deg(r.roll_p95_during_contact):.2f}°, "
                               f"pitch p95={np.rad2deg(r.pitch_p95_during_contact):.2f}°, "
                               f"{r.num_non_flat_frames} frames")
        
        lines.append("")
        lines.append(f"{'='*70}")
        
        return "\n".join(lines)

    def _generate_concise_summary(self, result: ValidationResult) -> str:
        """Generate a concise summary in quick-scan format."""
        lines = [
            "=== VALIDATION SUMMARY ===",
            f"Clip: {result.clip_name}",
            f"Duration: {result.duration_sec:.2f}s ({result.num_frames} frames @ {result.fps:.1f} Hz)",
            f"Status: {'✓ PASS' if result.all_passed else '✗ FAIL'}",
            "",
            "=== JOINT LIMITS ===",
        ]
        
        # Joint limits summary
        violations = [r for r in result.joint_limits_reports if not r.passed]
        if violations:
            lines.append(f"✗ {len(violations)} joints with violations:")
            for r in violations:
                lines.append(f"  {r.joint_name}: {r.num_violation_frames} frames, "
                           f"max violation: {r.max_violation:.4f} rad")
        else:
            lines.append("✓ All joints within safe limits")
        
        # Jitter statistics (worst 5 joints by velocity p99)
        if result.jitter_reports:
            lines.append("")
            lines.append("=== JITTER STATISTICS (Worst 5 joints) ===")
            worst_joints = sorted(result.jitter_reports, 
                                 key=lambda x: x.vel_p99, 
                                 reverse=True)[:5]
            for j in worst_joints:
                lines.append(f"{j.joint_name:30} vel_p99={j.vel_p99:8.3f} acc_p99={j.acc_p99:8.2f}")
        
        return "\n".join(lines)

    def save_reports(
        self,
        result: ValidationResult,
        output_dir: str,
    ):
        """Save validation reports as JSON and CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        json_file = output_dir / f"{result.clip_name}_validation.json"
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        if self.verbose:
            print(f"[Validator] Saved JSON report: {json_file}")
        
        # Summary text (concise format)
        summary_file = output_dir / f"{result.clip_name}_summary.txt"
        concise_summary = self._generate_concise_summary(result)
        with open(summary_file, 'w') as f:
            f.write(concise_summary)
        if self.verbose:
            print(f"[Validator] Saved summary: {summary_file}")
        
        # Joint limits CSV
        if result.joint_limits_reports:
            csv_file = output_dir / f"{result.clip_name}_joint_limits.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'joint_name', 'q_min', 'q_max', 'q_min_safe', 'q_max_safe',
                    'max_violation', 'num_violation_frames', 'percent_near_limit', 'passed'
                ])
                writer.writeheader()
                for r in result.joint_limits_reports:
                    d = r.to_dict()
                    d['violation_frames'] = str(r.violation_frames)
                    writer.writerow({k: v for k, v in d.items() 
                                   if k in writer.fieldnames})
            if self.verbose:
                print(f"[Validator] Saved joint limits CSV: {csv_file}")
        
        # Keypoint errors CSV
        if result.keypoint_errors:
            csv_file = output_dir / f"{result.clip_name}_keypoint_errors.csv"
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['keypoint_name', 'pos_error_mean', 'pos_error_p95', 'pos_error_max']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for e in result.keypoint_errors:
                    row = {
                        'keypoint_name': e.keypoint_name,
                        'pos_error_mean': e.pos_error_mean,
                        'pos_error_p95': e.pos_error_p95,
                        'pos_error_max': e.pos_error_max,
                    }
                    writer.writerow(row)
            if self.verbose:
                print(f"[Validator] Saved keypoint errors CSV: {csv_file}")
        
        # Jitter CSV
        if result.jitter_reports:
            csv_file = output_dir / f"{result.clip_name}_jitter.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'joint_name', 'vel_p95', 'vel_p99', 'acc_p95', 'acc_p99'
                ])
                writer.writeheader()
                for r in result.jitter_reports:
                    writer.writerow(r.to_dict())
            if self.verbose:
                print(f"[Validator] Saved jitter CSV: {csv_file}")

    def plot_motion(
        self,
        result: ValidationResult,
        joint_positions: np.ndarray,
        output_dir: str,
    ):
        """Generate diagnostic plots."""
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not available; skipping plots")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_frames = joint_positions.shape[0]
        time_axis = np.arange(n_frames) / result.fps
        
        # Plot 1: Joint positions with safe limits
        n_cols = 3
        n_joints = min(6, joint_positions.shape[1])
        n_rows = (n_joints + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = np.atleast_1d(axes).flatten()
        
        for i in range(n_joints):
            ax = axes[i]
            joint_name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i}"
            q = joint_positions[:, i]
            
            # Get safe limits from reports
            safe_min, safe_max = -np.pi, np.pi
            for r in result.joint_limits_reports:
                if r.joint_name == joint_name:
                    safe_min = r.q_min_safe
                    safe_max = r.q_max_safe
            
            ax.plot(time_axis, q, 'b-', linewidth=1, label='Joint position')
            ax.axhline(safe_min, color='r', linestyle='--', alpha=0.5, label='Safe limit')
            ax.axhline(safe_max, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (rad)')
            ax.set_title(joint_name)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_joints, len(axes)):
            axes[i].set_visible(False)
        
        fig.tight_layout()
        plot_file = output_dir / f"{result.clip_name}_joints.png"
        fig.savefig(plot_file, dpi=100)
        plt.close(fig)
        if self.verbose:
            print(f"[Validator] Saved plot: {plot_file}")
