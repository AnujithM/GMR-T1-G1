"""
Validation configurations for different robots.

Maps robots to their joint limits, keypoint mappings, and collision pairs.
"""

import json
from pathlib import Path

# Joint limits in radians for common humanoid robots
# These are approximate safe ranges; adjust based on actual robot specs

UNITREE_G1_LIMITS = {
    'left_hip_roll': (-0.52, 0.52),
    'left_hip_yaw': (-0.52, 0.52),
    'left_hip_pitch': (-1.57, 0.52),
    'left_knee': (0, 2.79),
    'left_ankle': (-0.52, 0.52),
    'right_hip_roll': (-0.52, 0.52),
    'right_hip_yaw': (-0.52, 0.52),
    'right_hip_pitch': (-1.57, 0.52),
    'right_knee': (0, 2.79),
    'right_ankle': (-0.52, 0.52),
    'left_shoulder_roll': (-2.09, 2.09),
    'left_shoulder_pitch': (-2.09, 2.09),
    'left_shoulder_yaw': (-2.09, 2.09),
    'left_elbow': (-2.09, 2.09),
    'right_shoulder_roll': (-2.09, 2.09),
    'right_shoulder_pitch': (-2.09, 2.09),
    'right_shoulder_yaw': (-2.09, 2.09),
    'right_elbow': (-2.09, 2.09),
}

UNITREE_H1_LIMITS = {
    'left_hip_roll': (-0.87, 0.87),
    'left_hip_yaw': (-0.87, 0.87),
    'left_hip_pitch': (-1.57, 0.87),
    'left_knee': (0, 2.79),
    'left_ankle_pitch': (-0.67, 0.67),
    'left_ankle_roll': (-0.67, 0.67),
    'right_hip_roll': (-0.87, 0.87),
    'right_hip_yaw': (-0.87, 0.87),
    'right_hip_pitch': (-1.57, 0.87),
    'right_knee': (0, 2.79),
    'right_ankle_pitch': (-0.67, 0.67),
    'right_ankle_roll': (-0.67, 0.67),
    'torso': (-0.87, 0.87),
    'left_shoulder_roll': (-2.09, 2.09),
    'left_shoulder_pitch': (-1.57, 1.57),
    'left_elbow': (-2.09, 2.09),
    'right_shoulder_roll': (-2.09, 2.09),
    'right_shoulder_pitch': (-1.57, 1.57),
    'right_elbow': (-2.09, 2.09),
}

BOOSTER_T1_LIMITS = {
    'left_hip_roll': (-1.57, 1.57),
    'left_hip_pitch': (-1.57, 1.57),
    'left_knee': (-2.09, 0),
    'left_ankle_pitch': (-0.87, 0.87),
    'left_ankle_roll': (-0.87, 0.87),
    'right_hip_roll': (-1.57, 1.57),
    'right_hip_pitch': (-1.57, 1.57),
    'right_knee': (-2.09, 0),
    'right_ankle_pitch': (-0.87, 0.87),
    'right_ankle_roll': (-0.87, 0.87),
    'waist_roll': (-0.87, 0.87),
    'waist_pitch': (-0.87, 0.87),
    'waist_yaw': (-0.87, 0.87),
}

BOOSTER_T1_29DOF_INSPIRE_LIMITS = {
    'left_hip_roll': (-1.57, 1.57),
    'left_hip_pitch': (-1.57, 1.57),
    'left_knee': (-2.09, 0),
    'left_ankle_pitch': (-0.87, 0.87),
    'left_ankle_roll': (-0.87, 0.87),
    'right_hip_roll': (-1.57, 1.57),
    'right_hip_pitch': (-1.57, 1.57),
    'right_knee': (-2.09, 0),
    'right_ankle_pitch': (-0.87, 0.87),
    'right_ankle_roll': (-0.87, 0.87),
    'waist_roll': (-0.87, 0.87),
    'waist_pitch': (-0.87, 0.87),
    'waist_yaw': (-0.87, 0.87),
    'left_shoulder_roll': (-2.09, 2.09),
    'left_shoulder_pitch': (-2.09, 2.09),
    'left_shoulder_yaw': (-2.09, 2.09),
    'left_elbow': (-2.09, 2.09),
    'right_shoulder_roll': (-2.09, 2.09),
    'right_shoulder_pitch': (-2.09, 2.09),
    'right_shoulder_yaw': (-2.09, 2.09),
    'right_elbow': (-2.09, 2.09),
}

# Keypoint mappings: which body corresponds to each tracking point
KEYPOINT_MAPS = {
    'unitree_g1': {
        'pelvis': 'pelvis',
        'head': 'head',
        'left_hand': 'left_hand',
        'right_hand': 'right_hand',
        'left_foot': 'left_foot',
        'right_foot': 'right_foot',
    },
    'unitree_h1': {
        'pelvis': 'pelvis',
        'head': 'head',
        'left_hand': 'left_hand',
        'right_hand': 'right_hand',
        'left_foot': 'left_foot',
        'right_foot': 'right_foot',
    },
    'booster_t1': {
        'pelvis': 'Waist',
        'head': 'Head',
        'left_hand': 'LeftHand',
        'right_hand': 'RightHand',
        'left_foot': 'LeftFoot',
        'right_foot': 'RightFoot',
    },
    'booster_t1_29dof_inspire_custom': {
        'pelvis': 'Waist',
        'head': 'Head',
        'left_hand': 'LeftHand',
        'right_hand': 'RightHand',
        'left_foot': 'LeftFoot',
        'right_foot': 'RightFoot',
    },
}

# Self-collision pairs to check: (body1_name, body2_name)
COLLISION_PAIRS = {
    'unitree_g1': [
        ('left_foot', 'right_foot'),
        ('left_hand', 'torso'),
        ('right_hand', 'torso'),
        ('left_thigh', 'torso'),
        ('right_thigh', 'torso'),
    ],
    'unitree_h1': [
        ('left_foot', 'right_foot'),
        ('left_hand', 'torso'),
        ('right_hand', 'torso'),
        ('left_thigh', 'torso'),
        ('right_thigh', 'torso'),
    ],
    'booster_t1': [
        ('LeftFoot', 'RightFoot'),
        ('LeftHand', 'Waist'),
        ('RightHand', 'Waist'),
        ('LeftThigh', 'Waist'),
        ('RightThigh', 'Waist'),
    ],
    'booster_t1_29dof_inspire_custom': [
        ('LeftFoot', 'RightFoot'),
        ('LeftHand', 'Waist'),
        ('RightHand', 'Waist'),
        ('LeftThigh', 'Waist'),
        ('RightThigh', 'Waist'),
    ],
}

# Foot body names for each robot
FOOT_NAMES = {
    'unitree_g1': ['left_foot', 'right_foot'],
    'unitree_h1': ['left_foot', 'right_foot'],
    'booster_t1': ['LeftFoot', 'RightFoot'],
    'booster_t1_29dof_inspire_custom': ['LeftFoot', 'RightFoot'],
}

# Robot XML file paths and joint names
ROBOT_CONFIGS = {
    'unitree_g1': {
        'xml': 'assets/unitree_g1/g1_mocap_29dof.xml',
        'joint_limits': UNITREE_G1_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['unitree_g1'],
        'collision_pairs': COLLISION_PAIRS['unitree_g1'],
        'foot_names': FOOT_NAMES['unitree_g1'],
    },
    'unitree_h1': {
        'xml': 'assets/unitree_h1/h1.xml',
        'joint_limits': UNITREE_H1_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['unitree_h1'],
        'collision_pairs': COLLISION_PAIRS['unitree_h1'],
        'foot_names': FOOT_NAMES['unitree_h1'],
    },
    'booster_t1': {
        'xml': 'assets/booster_t1/T1_serial.xml',
        'joint_limits': BOOSTER_T1_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['booster_t1'],
        'collision_pairs': COLLISION_PAIRS['booster_t1'],
        'foot_names': FOOT_NAMES['booster_t1'],
    },
    'booster_t1_29dof_inspire_custom': {
        'xml': 'assets/booster_t1_29dof_inspire_custom/robot.xml',
        'joint_limits': BOOSTER_T1_29DOF_INSPIRE_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['booster_t1_29dof_inspire_custom'],
        'collision_pairs': COLLISION_PAIRS['booster_t1_29dof_inspire_custom'],
        'foot_names': FOOT_NAMES['booster_t1_29dof_inspire_custom'],
    },
}


def get_robot_config(robot_name: str) -> dict:
    """Get configuration for a specific robot."""
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_name}. Available: {list(ROBOT_CONFIGS.keys())}")
    return ROBOT_CONFIGS[robot_name]
