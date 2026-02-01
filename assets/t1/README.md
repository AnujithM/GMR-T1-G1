# T1 Robot Model Assets

This directory contains the T1 robot model files for both training (URDF) and deployment (MuJoCo XML).

## Files

| File | Purpose |
|------|---------|
| `t1_robot.urdf` | Training model (Isaac Sim/Lab) |
| `t1_robot.xml` | Deployment model (MuJoCo) |
| `meshes/` | Visual meshes for robot body |
| `include/` | XML include files (actuators, sensors, assets) |

## Model Alignment

The URDF and XML are aligned on dynamics-critical properties:

| Property | URDF | XML | Match |
|----------|------|-----|-------|
| Joint limits (29 DOF) | Calibrated values | Calibrated values | ✓ |
| Joint axes | Same orientation | Same orientation | ✓ |
| Non-finger body masses | FALCON baseline | Updated to match URDF | ✓ |
| Hand mass | 0.62 kg (lumped) | 0.62 kg (lumped) | ✓ |
| Hand inertia | Scaled FALCON | Scaled FALCON | ✓ |
| Kinematic chain | FALCON structure | Same structure | ✓ |

## Expected Differences

| Property | URDF | XML | Reason |
|----------|------|-----|--------|
| Finger inertia | **Zero** | 0.001 kg | MuJoCo requires non-zero mass for jointed bodies |
| Finger joints | Fixed | Revolute | URDF uses lumped model; XML allows articulation |
| `left/right_inspire_hand` | Not present | Present | XML uses intermediate transform body for mesh alignment |

## Unit Tests

Run parity tests from `minerva_bringup/` root:

```bash
python -m pytest minerva_bringup/assets/t1/tests/test_urdf_xml_parity.py -v
```

### Test Coverage

| Test | Description |
|------|-------------|
| `test_urdf_parses` | URDF is valid XML |
| `test_urdf_link_count` | 54 links (30 body + 24 finger) |
| `test_xml_loads` | XML loads in MuJoCo |
| `test_left/right_hand_urdf/xml` | Hand mass = 0.62 kg |
| `test_urdf_fingers_zero` | Finger mass = 0 in URDF |
| `test_xml_fingers_negligible` | Finger mass < 0.01 kg in XML |
| `test_main_joint_count` | 29 actuated DOF |
| `test_joint_limits_match` | All 29 limits match |
| `test_non_finger_masses_match` | All body masses match |

## Hand Inertia Details

The hand uses a **lumped inertia model** where all finger mass is concentrated at `left_hand_link` / `right_hand_link`:

```
Hand mass: 0.62 kg (real hardware weight)
Inertia:   Scaled from FALCON baseline (preserves off-diagonal terms)
           Left:  ixx=0.003908, iyy=0.000646, izz=0.004010
           Right: ixx=0.003950, iyy=0.000630, izz=0.004031
```

Individual finger links exist for visualization but have zero (URDF) or negligible (XML) inertia.
