# Motion Validator - Usage Guide

Validate your GMR retargeted robot motions against 5 key checks.

## Quick Start

```bash
# Validate a single motion
python scripts/validate_motion.py --motion motion.pkl --robot booster_t1_29dof_inspire_custom

# Validate all motions in a directory
python scripts/validate_motion.py --motion-dir outputs/ --robot booster_t1_29dof_inspire_custom

# With target keypoints for tracking error analysis
python scripts/validate_motion.py --motion motion.pkl --robot booster_t1_29dof_inspire_custom --keypoints keypoints.npz

# Generate diagnostic plots
python scripts/validate_motion.py --motion motion.pkl --robot booster_t1_29dof_inspire_custom --plots

# Save to custom output directory
python scripts/validate_motion.py --motion motion.pkl --robot booster_t1_29dof_inspire_custom --output results/
```

## Supported Robots

- `unitree_g1` - Unitree G1 (29 DOF)
- `unitree_h1` - Unitree H1 (19 DOF)
- `booster_t1` - Booster T1 (13 DOF)
- `booster_t1_29dof_inspire_custom` - Booster T1 with hands (29 DOF)

To add a new robot, edit `general_motion_retargeting/validation_config.py`.

## Input File Formats

### Motion File (`.pkl`) - REQUIRED

A pickle file containing:

```python
{
    "fps": float,              # Frames per second
    "dof_pos": np.ndarray,     # (T, n_joints) - REQUIRED
    "root_pos": np.ndarray,    # (T, 3) - optional
    "root_rot": np.ndarray,    # (T, 4) in xyzw format - optional
    "foot_poses": dict,        # optional: {'foot_name': (T, 7), ...}
}
```

**Minimum required**: `dof_pos` (joint positions in radians)

### Target Keypoints (Optional)

For keypoint tracking error analysis. Supported formats:

**NPZ format** (recommended):
```python
np.savez('keypoints.npz',
    pelvis=np.array([[x,y,z], ...]),           # (T, 3)
    head=np.array([[x,y,z], ...]),             # (T, 3)
    left_hand=np.array([[x,y,z], ...]),        # (T, 3)
    right_hand=np.array([[x,y,z], ...]),       # (T, 3)
    left_foot=np.array([[x,y,z], ...]),        # (T, 3)
    right_foot=np.array([[x,y,z], ...]),       # (T, 3)
)
```

**CSV format**:
```
frame,keypoint_name,x,y,z
0,pelvis,0.0,0.0,1.0
0,head,0.0,0.1,1.8
...
```

## Validation Checks

1. **Joint Limits** - Checks if joints stay within safe bounds (±0.05 rad margin)
2. **Keypoint Tracking** - Position error for end-effectors (if keypoints provided)
3. **Self-Collision** - Detects disallowed body pair collisions
4. **Foot Ground Contact** - Checks foot penetration and flatness during stance
5. **Jitter & Temporal Artifacts** - Analyzes velocity and acceleration smoothness

## Output Files

For each motion, generates:

- **`{clip}_summary.txt`** - Quick-scan summary with status and worst joints
- **`{clip}_validation.json`** - Complete machine-readable results
- **`{clip}_joint_limits.csv`** - Per-joint limit violations
- **`{clip}_jitter.csv`** - Velocity and acceleration statistics
- **`{clip}_keypoint_errors.csv`** - Tracking errors (if keypoints provided)
- **`{clip}_joints.png`** - Plots (if matplotlib installed)

### Example Summary Output

```
=== VALIDATION SUMMARY ===
Clip: Sprint1_stageii
Duration: 2.21s (66 frames @ 29.9 Hz)
Status: ✓ PASS

=== JOINT LIMITS ===
✓ All joints within safe limits

=== JITTER STATISTICS (Worst 5 joints) ===
right_little_1_joint           vel_p99=11.619 acc_p99=318.22
Left_Knee_Pitch                vel_p99=11.436 acc_p99=260.94
right_little_2_joint           vel_p99=10.388 acc_p99=214.78
Trunk                          vel_p99=8.935 acc_p99=180.04
Left_Ankle_Pitch               vel_p99=8.416 acc_p99=224.97
```

## Command Line Options

```
--motion MOTION              Path to single .pkl motion file
--motion-dir MOTION_DIR      Directory with .pkl files for batch validation
--robot {unitree_g1,...}     Robot model (REQUIRED)
--keypoints KEYPOINTS        Path to target keypoints (CSV or NPZ)
--output OUTPUT              Output directory (default: validation_output)
--plots                      Generate diagnostic plots
--quiet                      Suppress verbose output
```

## Programmatic Usage

```python
from motion_validator import MotionValidator
from validation_config import get_robot_config
import numpy as np

# Get robot config
config = get_robot_config('booster_t1_29dof_inspire_custom')

# Create validator
validator = MotionValidator(
    robot_xml='assets/booster_t1_29dof_inspire_custom/robot.xml',
    keypoint_map=config['keypoint_map'],
    collision_pairs=config['collision_pairs'],
    foot_names=config['foot_names'],
    verbose=True,
)

# Load motion data
motion_data = np.load('motion.pkl')
joint_positions = motion_data['dof_pos']  # (T, n_joints)
root_pose = motion_data['root_pos']       # (T, 7)

# Run validation
result = validator.validate(
    motion_file='motion.pkl',
    joint_positions=joint_positions,
    root_pose=root_pose,
    fps=30.0,
    joint_limits=config['joint_limits'],
)

# Save reports
validator.save_reports(result, output_dir='results/')

# Access results
print(f"All checks passed: {result.all_passed}")
print(f"Joint limits OK: {result.joint_limits_passed}")
```

## Interpretation

### Joint Limits
- ✓ PASS: All joints within safe bounds
- ✗ FAIL: One or more joints violated limits

### Keypoint Tracking
- Mean error < 0.05m: Excellent
- Mean error < 0.10m: Good
- Mean error > 0.15m: Check IK quality

### Self-Collision
- ✓ PASS: No disallowed collisions
- ✗ FAIL: Collision detected

### Foot Ground
- ✓ PASS: Proper contact, no penetration
- ✗ FAIL: Foot penetration > 0.01m or non-flat contact

### Jitter (Worst Joints)
- Lower values: Smoother motion
- vel_p99 < 5 rad/s: Smooth
- vel_p99 > 10 rad/s: Dynamic/aggressive motion

## Troubleshooting

### "Robot XML not found"
- Check robot name spelling
- Verify assets directory exists
- Use absolute path if needed

### "No .pkl files found"
- Check directory path
- Ensure motion files have .pkl extension

### MuJoCo warnings
- Optional; collision detection will be skipped
- Install with: `pip install mujoco`

### Matplotlib warnings
- Optional; plots will be skipped
- Install with: `pip install matplotlib`

## See Also

- `VALIDATOR_README.md` - Full API documentation
- `general_motion_retargeting/validation_config.py` - Add new robots here
- `general_motion_retargeting/motion_validator.py` - Core validation engine
